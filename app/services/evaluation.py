import os
import json
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

load_dotenv()
# Note: In 2026, the updated SDK uses google.genai.Client() rather than genai.configure()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define our model string constants
FLASH_MODEL = "gemini-3.5-flash"
PRO_MODEL = "gemini-3.1-pro-preview"

def parse_json_safely(raw_text: str) -> dict:
    """Helper to strip markdown blocks from LLM responses."""
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def evaluate_answer(question: str, context_chunks: list[str], student_answer: str) -> dict:
    """
    Adaptive Cascade Evaluator
    Level 1: 3.5 Flash (Fast Relevance/Grounding Check)
    Level 2: 3.1 Pro (Deep Logical Audit if Flash flags an issue)
    """
    context_text = "\n---\n".join(context_chunks)
    
    # ---------------------------------------------------------
    # LEVEL 1: The Sorter (gemini-3.5-flash)
    # ---------------------------------------------------------
    l1_prompt = f"""
    You are a fast evaluation triage agent. 
    Question: {question}
    Context: {context_text}
    Student Answer: {student_answer}
    
    Grade Grounding (0-100) and Relevance (0-100).
    Return EXACTLY as JSON:
    {{"grounding_score": int, "relevance_score": int, "hallucination_risk": "Low" | "Medium" | "High", "feedback": "string"}}
    """
    
    try:
        # We don't need deep thinking for Level 1, just a fast read.
        l1_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
        )
        l1_response = client.models.generate_content(
            model=FLASH_MODEL, 
            contents=l1_prompt,
            config=l1_config
        )
        l1_data = parse_json_safely(l1_response.text)
        
        grounding = l1_data.get("grounding_score", 0)
        relevance = l1_data.get("relevance_score", 0)
        risk = l1_data.get("hallucination_risk", "High")
        
        # EARLY EXIT: If the answer is basically perfect, stop here and save money!
        if grounding >= 90 and relevance >= 90 and risk == "Low":
            print("🟢 Level 1 Passed: No escalation needed.")
            l1_data["highest_level_used"] = "Level 1 (Flash)"
            l1_data["logical_consistency_score"] = None
            l1_data["escalation_reason"] = None
            return l1_data
            
        print(f"🟡 Anomaly Detected (Grounding: {grounding}, Risk: {risk}). Escalating to Level 2...")
        escalation_reason = l1_data.get("feedback", "Scores dropped below threshold.")
        
    except Exception as e:
        print(f"⚠️ Level 1 Failed to execute: {e}. Escalating to Level 2...")
        escalation_reason = f"Level 1 API Failure: {str(e)}"
        # We initialize empty data so L2 has a clean slate
        l1_data = {}


    # ---------------------------------------------------------
    # LEVEL 2: The Auditor / Supreme Court (gemini-3.1-pro-preview)
    # ---------------------------------------------------------
    # We only reach this code if Level 1 exited early or failed.
    
    l2_prompt = f"""
    You are a Master AI Auditor. A previous evaluation flagged an anomaly.
    Question: {question}
    Context: {context_text}
    Student Answer: {student_answer}
    Escalation Reason: {escalation_reason}
    
    Perform a deep, logical contrast check. 
    1. Recalculate Grounding (0-100) and Relevance (0-100).
    2. Provide a 'logical_consistency_score' (0-100) detailing if the student's answer contradicts itself.
    
    Return EXACTLY as JSON:
    {{"grounding_score": int, "relevance_score": int, "logical_consistency_score": int, "hallucination_risk": "Low" | "Medium" | "High", "feedback": "Detailed forensic explanation"}}
    """
    
    try:
        # We trigger Extended Thinking (HIGH budget) to ensure maximum intelligence
        l2_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH)
        )
        l2_response = client.models.generate_content(
            model=PRO_MODEL, 
            contents=l2_prompt,
            config=l2_config
        )
        l2_data = parse_json_safely(l2_response.text)
        
        # Package the final escalated payload
        l2_data["highest_level_used"] = "Level 2 (Pro Extended)"
        l2_data["escalation_reason"] = escalation_reason
        return l2_data
        
    except Exception as e:
        print(f"❌ Level 2 API Error: {e}")
        return {
            "grounding_score": 0,
            "relevance_score": 0,
            "logical_consistency_score": 0,
            "hallucination_risk": "High",
            "highest_level_used": "Level 2 Failed",
            "escalation_reason": escalation_reason,
            "feedback": f"Critical Pipeline Failure: {str(e)}"
        }