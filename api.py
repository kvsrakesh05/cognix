from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid

from llm_extractor import extract_symptoms
from adaptive_questions import get_next_question
import medical_assessment
from routing import route_case


app = FastAPI()


# ============================================================
# TEMPORARY IN-MEMORY SESSIONS
# ============================================================
# Used for our prototype.
# Later this can be replaced by a real database/session store.

sessions = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class StartAssessmentRequest(BaseModel):
    complaint: str


class AnswerRequest(BaseModel):
    session_id: str
    answer: object


# ============================================================
# HELPER — FIND FIELD FOR CURRENT QUESTION
# ============================================================

def find_field(question):

    q = question.lower()

    if "when did" in q:
        return "onset"

    if "how severe" in q:
        return "severity"

    if "spread to" in q:
        return "radiation"

    if "difficulty breathing" in q:
        return "breathlessness"

    if "sweating" in q:
        return "sweating"

    if "dizzy" in q:
        return "dizziness"

    if "nauseous" in q:
        return "nausea"

    if "racing, pounding" in q:
        return "palpitations"

    if "difficulty speaking" in q:
        return "speech_difficulty"

    if "drooping" in q:
        return "facial_drooping"

    if "sudden changes in your vision" in q:
        return "vision_change"

    if "severe headache" in q:
        return "severe_headache"

    if "fever" in q:
        return "fever"

    if "are you coughing" in q:
        return "cough"

    if "wheezing" in q:
        return "wheezing"

    if "bluish or grey" in q:
        return "blue_lips"

    if "where exactly" in q:
        return "location"

    if "vomiting" in q:
        return "vomiting"

    if "diarrhea" in q:
        return "diarrhea"

    if "blood in your stool" in q:
        return "blood_in_stool"

    if "swelling" in q:
        return "swelling"

    if "after an injury" in q:
        return "injury"

    if "movement make the pain worse" in q:
        return "movement"

    if "pain or burning when urinating" in q:
        return "pain"

    if "urinate more often" in q:
        return "frequency"

    if "blood in your urine" in q:
        return "blood"

    if "pain in your back or side" in q:
        return "back_pain"

    return None


# ============================================================
# HELPER — BUILD TEXT FOR FINAL ML MODEL
# ============================================================

def symptoms_to_text(symptoms):

    parts = []

    for key, value in symptoms.items():

        if value is None or value is False:
            continue

        if value is True:
            parts.append(key.replace("_", " "))

        else:
            parts.append(
                f"{key.replace('_', ' ')} {value}"
            )

    return " ".join(parts)


# ============================================================
# START ASSESSMENT
# ============================================================

@app.post("/start")
def start_assessment(request: StartAssessmentRequest):

    complaint = request.complaint

    # ---------------------------------------------
    # LLM extraction
    # ---------------------------------------------

    symptoms = extract_symptoms(complaint)

    # ---------------------------------------------
    # Initial ML domain prediction
    # ---------------------------------------------

    predicted_domain = medical_assessment.domain_model.predict(
        [complaint]
    )[0]

    # ---------------------------------------------
    # Create session
    # ---------------------------------------------

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "original_complaint": complaint,
        "symptoms": symptoms,
        "domain": predicted_domain
    }

    # ---------------------------------------------
    # First adaptive question
    # ---------------------------------------------

    next_question = get_next_question(
        symptoms,
        predicted_domain
    )

    # ---------------------------------------------
    # No questions needed
    # ---------------------------------------------

    if next_question is None:

        final_result = medical_assessment.assess_complaint(
            complaint + " " + symptoms_to_text(symptoms)
        )

        routing_result = route_case(final_result)

        return {
            "session_id": session_id,
            "status": "complete",
            "symptoms": symptoms,
            "assessment": final_result,
            "routing": routing_result
        }

    return {
        "session_id": session_id,
        "status": "questioning",
        "domain": predicted_domain,
        "symptoms": symptoms,
        "question": next_question
    }


# ============================================================
# ANSWER QUESTION
# ============================================================

@app.post("/answer")
def answer_question(request: AnswerRequest):

    session_id = request.session_id

    # ---------------------------------------------
    # Check session
    # ---------------------------------------------

    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Assessment session not found"
        )

    session = sessions[session_id]

    symptoms = session["symptoms"]
    domain = session["domain"]

    # ---------------------------------------------
    # Find the question currently being asked
    # ---------------------------------------------

    current_question = get_next_question(
        symptoms,
        domain
    )

    if current_question is None:

        raise HTTPException(
            status_code=400,
            detail="No unanswered question remains"
        )

    # ---------------------------------------------
    # Determine which field the answer belongs to
    # ---------------------------------------------

    field = find_field(current_question)

    if field is None:

        raise HTTPException(
            status_code=500,
            detail="Could not determine question field"
        )

    # ---------------------------------------------
    # Store answer
    # ---------------------------------------------

    symptoms[field] = request.answer

    # ---------------------------------------------
    # Get next question
    # ---------------------------------------------

    next_question = get_next_question(
        symptoms,
        domain
    )

    # ---------------------------------------------
    # More questions remain
    # ---------------------------------------------

    if next_question is not None:

        return {
            "session_id": session_id,
            "status": "questioning",
            "question": next_question,
            "symptoms": symptoms
        }

    # ========================================================
    # FINAL ASSESSMENT
    # ========================================================

    final_text = (
        session["original_complaint"]
        + " "
        + symptoms_to_text(symptoms)
    )

    # ---------------------------------------------
    # Final ML + safety assessment
    # ---------------------------------------------

    assessment_result = medical_assessment.assess_complaint(
        final_text
    )

    # ---------------------------------------------
    # Routing
    # ---------------------------------------------

    routing_result = route_case(
        assessment_result
    )

    return {
        "session_id": session_id,
        "status": "complete",
        "symptoms": symptoms,
        "assessment": assessment_result,
        "routing": routing_result
    }


# ============================================================
# BASIC HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Acute Risk Assessment API"
    }