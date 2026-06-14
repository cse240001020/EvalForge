import os
import math
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure the Gemini SDK with your API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def generate_embedding(text: str) -> list[float]:
    """
    Sends a text chunk to Google's Gemini embedding model.
    Truncates the 3072D vector down to 768D to match our database.
    """
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",  # <-- The new active model
            content=text,
            task_type="retrieval_document"
        )
        
        # 1. Get the raw array and slice off the first 768 numbers
        raw_embedding = response['embedding']
        truncated = raw_embedding[:768]
        
        # 2. Normalize the vector using standard math so cosine similarity still works
        magnitude = math.sqrt(sum(x * x for x in truncated))
        normalized_vector = [x / magnitude for x in truncated]
        
        return normalized_vector

    except Exception as e:
        print(f"❌ Gemini Embedding Error: {e}")
        raise e