import os
import google.generativeai as genai
from app.core.db import get_db_connection
from app.services.embeddings import generate_embedding
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Use the current generation fast/cheap model for the Student
student_model = genai.GenerativeModel('gemini-2.5-flash')

def retrieve_relevant_chunks(question: str, project_id: int, limit: int = 3) -> list[str]:
    """
    1. Converts the user's question into a vector.
    2. Uses pgvector's cosine distance operator (<=>) to find the closest chunks.
    """
    # Convert the question into a 768-dimensional vector
    question_vector = generate_embedding(question)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # SQL query using pgvector for similarity search
        query = """
            SELECT c.content 
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.project_id = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
        """
        # Note: psycopg requires vectors to be formatted as strings for insertion
        vector_str = "[" + ",".join(map(str, question_vector)) + "]"
        
        cur.execute(query, (project_id, vector_str, limit))
        results = cur.fetchall()
        
        # Return a list of just the text content
        return [row[0] for row in results]
    
    except Exception as e:
        print(f"❌ Retrieval Error: {e}")
        raise e
    finally:
        cur.close()
        conn.close()

def generate_student_answer(question: str, project_id: int) -> dict:
    """
    The core RAG function. Retrieves context and generates an answer.
    """
    # 1. Retrieve the top 3 relevant chunks
    relevant_chunks = retrieve_relevant_chunks(question, project_id)
    
    # Combine chunks into a single string for the prompt
    context_text = "\n\n---\n\n".join(relevant_chunks)
    
    # 2. Build the strict prompt for the Student model
    prompt = f"""
    You are an expert answering questions based strictly on the provided context.
    
    Context:
    {context_text}
    
    Question: {question}
    
    Instructions:
    1. Answer the question using ONLY the information in the context above.
    2. If the answer cannot be found in the context, explicitly say: "I cannot answer this based on the provided documents."
    3. Be concise and direct.
    
    Answer:
    """
    
    try:
        # 3. Generate the answer using Gemini Flash
        response = student_model.generate_content(prompt)
        answer_text = response.text
        
        return {
            "question": question,
            "answer": answer_text,
            "context_used": relevant_chunks
        }
        
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        raise e