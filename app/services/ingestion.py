import io
from pypdf import PdfReader
from app.core.db import get_db_connection
from app.services.embeddings import generate_embedding

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += (chunk_size - chunk_overlap)
    return [c for c in chunks if c]

def process_pdf_document(project_name: str, filename: str, file_bytes: bytes):
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1. Ensure the project exists, get its ID
        cur.execute(
            "INSERT INTO projects (name) VALUES (%s) RETURNING id;", 
            (project_name,)
        )
        project_id = cur.fetchone()[0]

        # 2. Track the uploaded document
        cur.execute(
            "INSERT INTO documents (project_id, filename) VALUES (%s, %s) RETURNING id;",
            (project_id, filename)
        )
        document_id = cur.fetchone()[0]

        # 3. Read text from the PDF file
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        # 4. Run the chunking function
        chunks = split_text_into_chunks(full_text, chunk_size=600, chunk_overlap=120)

        # 5. Generate embeddings and save to PostgreSQL
        print(f"Starting vector generation for {len(chunks)} chunks...")
        for chunk_text in chunks:
            # Call the embedding service to get our 768-dimensional array
            vector_embedding = generate_embedding(chunk_text)
            
            # Insert both text content AND the vector array into pgvector
            cur.execute(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (%s, %s, %s);",
                (document_id, chunk_text, vector_embedding)
            )

        conn.commit()
        return {
            "status": "success",
            "project_id": project_id,
            "document_id": document_id,
            "chunks_created": len(chunks),
            "embeddings_generated": True
        }

    except Exception as e:
        conn.rollback()
        print(f"❌ Error ingesting document: {e}")
        raise e
    finally:
        cur.close()
        conn.close()