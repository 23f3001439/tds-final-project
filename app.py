import os
import sqlite3
import traceback
import logging
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Optional
import base64
import numpy as np
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
API_KEY = os.getenv("API_KEY")  # Set this in Render's environment variables
DB_PATH = "knowledge.db"

# Initialize FastAPI app
app = FastAPI()

# CORS middleware (IMPORTANT for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming requests
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded image string

# --- Utility Functions ---

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

async def get_embedding(text: str) -> list:
    if not API_KEY:
        raise ValueError("API_KEY not set in environment variables")

    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "input": text,
        "model": "text-embedding-ada-002"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot / (norm1 * norm2)) if norm1 and norm2 else 0.0

async def process_multimodal_query(text, image_base64=None):
    if image_base64:
        logger.info("Image input received but currently not used in embedding.")
    return await get_embedding(text)

async def find_similar_content(query_embedding, conn, top_k=5, similarity_threshold=0.78):
    def get_chunks(table):
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, content, url, embedding FROM {table} WHERE embedding IS NOT NULL")
        return [dict(row) for row in cursor.fetchall()]

    def parse_embedding(embedding_str):
        return list(map(float, embedding_str.strip("[]").split(",")))

    results = []
    for table in ["markdown_chunks", "discourse_chunks"]:
        for chunk in get_chunks(table):
            chunk_embedding = parse_embedding(chunk["embedding"])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            if similarity >= similarity_threshold:
                results.append({
                    "id": chunk["id"],
                    "content": chunk["content"],
                    "url": chunk["url"],
                    "similarity": similarity,
                    "table": table
                })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

async def enrich_with_adjacent_chunks(conn, results, window=1):
    enriched = []
    for res in results:
        table, chunk_id = res["table"], res["id"]
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, content, url FROM {table} WHERE id BETWEEN ? AND ?", (chunk_id - window, chunk_id + window))
        enriched.extend([dict(row) for row in cursor.fetchall()])
    return enriched

async def generate_answer(query, context_chunks):
    try:
        context_text = "\n\n".join(chunk["content"] for chunk in context_chunks)
        prompt = (
            f"You are an intelligent assistant. Use the following knowledge base to answer the query.\n\n"
            f"Knowledge:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer (include 'Sources:' followed by source URLs if relevant):"
        )
        url = "https://aipipe.org/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        raise

def parse_llm_response(response):
    try:
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        answer = parts[0].strip()
        links = []
        if len(parts) > 1:
            lines = parts[1].strip().split("\n")
            for line in lines:
                line = re.sub(r'^\d+\.\s*|^-+\s*', '', line.strip())
                url_match = re.search(r'(https?://[^\s\]]+)', line)
                if url_match:
                    url = url_match.group(1)
                    text_match = re.search(r'"(.*?)"', line)
                    text = text_match.group(1) if text_match else "Source"
                    links.append({"url": url, "text": text})
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return {"answer": "Error parsing response.", "links": []}

# --- API ROUTES ---

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        if not API_KEY:
            return JSONResponse(status_code=500, content={"error": "API_KEY not set."})

        conn = get_db_connection()
        query_embedding = await process_multimodal_query(request.question, request.image)
        relevant = await find_similar_content(query_embedding, conn)

        if not relevant:
            return {"answer": "I couldn't find relevant information.", "links": []}

        enriched = await enrich_with_adjacent_chunks(conn, relevant)
        llm_response = await generate_answer(request.question, enriched)
        result = parse_llm_response(llm_response)

        if not result["links"]:
            urls = set()
            links = []
            for res in relevant[:5]:
                if res["url"] not in urls:
                    urls.add(res["url"])
                    snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                    links.append({"url": res["url"], "text": snippet})
            result["links"] = links

        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embed = cursor.fetchone()[0]
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse,
            "markdown_chunks": markdown,
            "discourse_embeddings": discourse_embed,
            "markdown_embeddings": markdown_embed
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
from fastapi import Request

from fastapi import Request

@app.api_route("/", methods=["GET", "POST"])
async def root(request: Request):
    return {
        "answer": "✅ TDS Final Project backend is live. Use /query endpoint to ask questions.",
        "links": [
            {"url": "https://tds-final-project-deploy.onrender.com/query", "text": "Query Endpoint"},
            {"url": "https://tds-final-project-deploy.onrender.com/health", "text": "Health Check"}
        ]
    }



# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
