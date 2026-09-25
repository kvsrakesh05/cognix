from llm_extractor import extract_symptoms
from adaptive_questions import get_next_question
import medical_assessment
from routing import route_case


# ============================================================
# 1. PATIENT COMPLAINT
# ============================================================

complaint = (
    "I have had severe chest pain since morning. "
    "It is spreading to my left arm and I am sweating."
)

print("\nPATIENT COMPLAINT")
print("-----------------------------")
print(complaint)


# ============================================================
# 2. LLM EXTRACTION
# ============================================================

symptoms = extract_symptoms(complaint)

print("\nLLM EXTRACTED SYMPTOMS")
print("-----------------------------")

for key, value in symptoms.items():
    print(f"{key}: {value}")


# ============================================================
# 3. INITIAL DOMAIN PREDICTION
# ============================================================

predicted_domain = medical_assessment.domain_model.predict([complaint])[0]

print("\nINITIAL ML DOMAIN")
print("-----------------------------")
print(predicted_domain)


# ============================================================
# 4. ADAPTIVE QUESTIONING
# ============================================================

answers = {
    "onset": "since morning",
    "severity": 8,
    "radiation": "left arm",
    "breathlessness": True,
    "sweating": True,
    "dizziness": False,
    "nausea": True,
    "palpitations": False
}

print("\nADAPTIVE QUESTIONING")
print("-----------------------------")


while True:

    question = get_next_question(
        symptoms,
        predicted_domain
    )

    if question is None:
        print("\nAssessment questions complete.")
        break

    print("\nQuestion:", question)

    question_lower = question.lower()

    if "when did" in question_lower:
        field = "onset"

    elif "how severe" in question_lower:
        field = "severity"

    elif "spread" in question_lower:
        field = "radiation"

    elif "difficulty breathing" in question_lower:
        field = "breathlessness"

    elif "sweating" in question_lower:
        field = "sweating"

    elif "dizzy" in question_lower:
        field = "dizziness"

    elif "nauseous" in question_lower:
        field = "nausea"

    elif "racing, pounding" in question_lower:
        field = "palpitations"

    else:
        print("No simulated answer available.")
        break

    symptoms[field] = answers[field]

    print("Patient answer:", answers[field])


# ============================================================
# 5. FINAL ML + SAFETY ASSESSMENT
# ============================================================

symptom_text = []

for key, value in symptoms.items():

    if value is True:
        symptom_text.append(key.replace("_", " "))

    elif value is False or value is None:
        continue

    else:
        symptom_text.append(
            key.replace("_", " ") + " " + str(value)
        )

final_complaint = complaint + " " + " ".join(symptom_text)

result = medical_assessment.assess_complaint(final_complaint)


print("\nFINAL ASSESSMENT")
print("-----------------------------")
print("Urgency:", result["urgency"])
print("Domain:", result["domain"])
print("Action:", result["action"])
print("Safety override:", result["safety_override"])
print("Reason:", result["reason"])


# ============================================================
# 6. ROUTING
# ============================================================

routing_result = route_case(result)


print("\nROUTING RESULT")
print("-----------------------------")
print("Flag GP:", routing_result["flag_gp"])
print("Flag Specialist:", routing_result["flag_specialist"])
print("Department:", routing_result["department"])
print("Priority:", routing_result["priority"])
print("Action:", routing_result["action"])
print("Safety override:", routing_result["safety_override"])


# ============================================================
# 7. FINAL SYMPTOMS
# ============================================================

print("\nFINAL STRUCTURED SYMPTOMS")
print("-----------------------------")

for key, value in symptoms.items():
    print(f"{key}: {value}")