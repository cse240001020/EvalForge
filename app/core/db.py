import os
import psycopg
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """
    Creates and returns a raw connection to the PostgreSQL database.
    """
    try:
        conn = psycopg.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        raise e