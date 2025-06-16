import os
import json
import sqlite3
import argparse
import asyncio
import logging
import aiohttp
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
API_KEY = os.getenv("API_KEY")

# Create SQLite connection
def create_connection(db_file="knowledge.db"):
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
    return None

# Create tables
def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS discourse_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT, topic_id TEXT, topic_title TEXT, post_number INTEGER,
        author TEXT, created_at TEXT, likes INTEGER,
        chunk_index TEXT, content TEXT, url TEXT,
        embedding BLOB
    )""")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS markdown_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_title TEXT, original_url TEXT, downloaded_at TEXT,
        chunk_index TEXT, content TEXT,
        embedding BLOB
    )""")
    conn.commit()

# Dummy placeholders for parsing functions
def process_discourse_files(conn):
    logger.info("Simulating discourse chunk loading")
    # You'd implement the real logic here

def process_markdown_files(conn):
    logger.info("Simulating markdown chunk loading")
    # You'd implement the real logic here

# Embedding logic
def load_chunks_without_embeddings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM discourse_chunks WHERE embedding IS NULL")
    discourse_chunks = cursor.fetchall()
    logger.info(f"Found {len(discourse_chunks)} discourse chunks to embed")

    cursor.execute("SELECT id, content FROM markdown_chunks WHERE embedding IS NULL")
    markdown_chunks = cursor.fetchall()
    logger.info(f"Found {len(markdown_chunks)} markdown chunks to embed")
    return discourse_chunks, markdown_chunks

async def handle_long_text(session, text, record_id, api_key, is_discourse=True, max_retries=3):
    max_chars = 8000
    if len(text) <= max_chars:
        return await embed_text(session, text, record_id, api_key, is_discourse, max_retries)

    logger.info(f"Text exceeds embedding limit for {record_id}: {len(text)} chars. Creating multiple embeddings.")
    overlap = 200
    subchunks = []
    for i in range(0, len(text), max_chars - overlap):
        end = min(i + max_chars, len(text))
        subchunk = text[i:end]
        if subchunk:
            subchunks.append(subchunk)

    logger.info(f"Split into {len(subchunks)} subchunks for embedding")
    results = []
    for i, subchunk in enumerate(subchunks):
        logger.info(f"Embedding subchunk {i+1}/{len(subchunks)} for {record_id}")
        success = await embed_text(
            session,
            subchunk,
            record_id,
            api_key,
            is_discourse,
            max_retries,
            f"part_{i+1}_of_{len(subchunks)}"
        )
        results.append(success)
    return all(results)

async def embed_text(session, text, record_id, api_key, is_discourse=True, max_retries=3, part_id=None):
    retries = 0
    while retries < max_retries:
        try:
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    embedding_blob = json.dumps(embedding).encode()

                    cursor = conn.cursor()
                    if part_id:
                        if is_discourse:
                            cursor.execute("SELECT post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url FROM discourse_chunks WHERE id = ?", (record_id,))
                            original = cursor.fetchone()
                            if original:
                                cursor.execute("""
                                    INSERT INTO discourse_chunks 
                                    (post_id, topic_id, topic_title, post_number, author, created_at, 
                                     likes, chunk_index, content, url, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    original["post_id"], original["topic_id"], original["topic_title"],
                                    original["post_number"], original["author"], original["created_at"],
                                    original["likes"], f"{original['chunk_index']}_{part_id}", text, original["url"], embedding_blob
                                ))
                        else:
                            cursor.execute("SELECT doc_title, original_url, downloaded_at, chunk_index FROM markdown_chunks WHERE id = ?", (record_id,))
                            original = cursor.fetchone()
                            if original:
                                cursor.execute("""
                                    INSERT INTO markdown_chunks
                                    (doc_title, original_url, downloaded_at, chunk_index, content, embedding)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    original["doc_title"], original["original_url"], original["downloaded_at"],
                                    f"{original['chunk_index']}_{part_id}", text, embedding_blob
                                ))
                    else:
                        if is_discourse:
                            cursor.execute("UPDATE discourse_chunks SET embedding = ? WHERE id = ?", (embedding_blob, record_id))
                        else:
                            cursor.execute("UPDATE markdown_chunks SET embedding = ? WHERE id = ?", (embedding_blob, record_id))
                    conn.commit()
                    return True
                elif response.status == 429:
                    error_text = await response.text()
                    logger.warning(f"Rate limit hit, retrying (attempt {retries+1}): {error_text}")
                    await asyncio.sleep(5 * (retries + 1))
                    retries += 1
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to embed (status {response.status}): {error_text}")
                    return False
        except Exception:
            logger.exception("Exception embedding text")
            retries += 1
            await asyncio.sleep(3 * retries)
    logger.error("Failed to embed after maximum retries")
    return False

async def create_embeddings(api_key):
    global conn
    conn = create_connection()
    if not conn:
        return
    discourse_chunks, markdown_chunks = load_chunks_without_embeddings(conn)

    batch_size = 10
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(discourse_chunks), batch_size):
            batch = discourse_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, api_key, True) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Discourse batch {i//batch_size+1}: {sum(results)}/{len(batch)} embedded")
            if i + batch_size < len(discourse_chunks):
                await asyncio.sleep(2)

        for i in range(0, len(markdown_chunks), batch_size):
            batch = markdown_chunks[i:i+batch_size]
            tasks = [handle_long_text(session, text, record_id, api_key, False) for record_id, text in batch]
            results = await asyncio.gather(*tasks)
            logger.info(f"Markdown batch {i//batch_size+1}: {sum(results)}/{len(batch)} embedded")
            if i + batch_size < len(markdown_chunks):
                await asyncio.sleep(2)

    conn.close()
    logger.info("Finished all embeddings")

async def main():
    global CHUNK_SIZE, CHUNK_OVERLAP
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", help="API key for embedding")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    api_key = args.api_key or API_KEY
    if not api_key:
        logger.error("API key is required.")
        return

    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    conn = create_connection()
    if conn:
        create_tables(conn)
        process_discourse_files(conn)
        process_markdown_files(conn)
        conn.close()

    await create_embeddings(api_key)

if __name__ == "__main__":
    asyncio.run(main())
