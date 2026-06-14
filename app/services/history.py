from app.core.db import get_db_connection

def save_evaluation_history(project_id: int, question: str, answer_text: str, model_name: str, eval_data: dict):
    """
    Saves the RAG session into the database across three linked tables:
    1. test_cases
    2. responses
    3. evaluations (Now upgraded for the Adaptive Cascade)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 1. Log the Question into test_cases
        cur.execute(
            """
            INSERT INTO test_cases (project_id, question) 
            VALUES (%s, %s) 
            RETURNING id;
            """,
            (project_id, question)
        )
        test_case_id = cur.fetchone()[0]
        
        # 2. Log the Student's Answer into responses
        cur.execute(
            """
            INSERT INTO responses (test_case_id, model_name, answer_text) 
            VALUES (%s, %s, %s) 
            RETURNING id;
            """,
            (test_case_id, model_name, answer_text)
        )
        response_id = cur.fetchone()[0]
        
        # 3. Determine if the system passed your quality threshold
        grounding = eval_data.get("grounding_score", 0)
        relevance = eval_data.get("relevance_score", 0)
        has_passed = grounding >= 80 and relevance >= 80
        
        # 4. Log the Cascade's Scores into evaluations (NEW SCHEMA)
        cur.execute(
            """
            INSERT INTO evaluations (
                response_id, grounding_score, relevance_score, 
                completeness_score, feedback, passed,
                highest_level_used, logical_consistency_score, escalation_reason
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """,
            (
                response_id, 
                grounding, 
                relevance, 
                100,  # Defaulting completeness to 100 for now
                eval_data.get("feedback", "No feedback provided"), 
                has_passed,
                # New Adaptive Cascade Fields:
                eval_data.get("highest_level_used", "Level 1 (Flash)"),
                eval_data.get("logical_consistency_score", None),
                eval_data.get("escalation_reason", None)
            )
        )
        
        # Commit all inserts together safely
        conn.commit()
        return {"status": "persisted", "test_case_id": test_case_id, "response_id": response_id}
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Failed to log history to database: {e}")
        raise e
    finally:
        cur.close()
        conn.close()