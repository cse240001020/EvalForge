from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from app.core.db import get_db_connection
from app.services.ingestion import process_pdf_document
from app.services.generation import generate_student_answer 
from app.services.evaluation import evaluate_answer
from app.services.history import save_evaluation_history
app = FastAPI(title="EvalForge API", version="1.0.0")

# 1. Pydantic Models must be defined at the top
class QuestionRequest(BaseModel):
    project_id: int
    question: str


# 2. Health Check Endpoint
@app.get("/")
def health_check():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        db_status = "Connected"
    except Exception:
        db_status = "Disconnected/Error"

    return {"status": "healthy", "database": db_status}


# 3. Document Ingestion Endpoint
@app.post("/upload")
async def upload_document(
    project_name: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        file_bytes = await file.read()
        result = process_pdf_document(
            project_name=project_name,
            filename=file.filename,
            file_bytes=file_bytes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    1. Generates an answer using the Student (Gemini 2.5 Flash).
    2. Evaluates the answer using the Judge (Gemini 2.5 Flash).
    3. Permanently logs the complete transaction into PostgreSQL history.
    """
    try:
        # Step 1: The Student generates the response text
        student_result = generate_student_answer(
            question=request.question,
            project_id=request.project_id
        )
        
        # Step 2: The Judge scores the response
        evaluation_result = evaluate_answer(
            question=student_result["question"],
            context_chunks=student_result["context_used"],
            student_answer=student_result["answer"]
        )
        
        # Step 3: Log everything to your database tables
        save_evaluation_history(
            project_id=request.project_id,
            question=request.question,
            answer_text=student_result["answer"],
            model_name="gemini-2.5-flash",
            eval_data=evaluation_result
        )
        
        # Step 4: Return unified payload back to UI
        return {
            "project_id": request.project_id,
            "question": student_result["question"],
            "student_answer": student_result["answer"],
            "evaluation": evaluation_result,
            "saved_to_history": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))