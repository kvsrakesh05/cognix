import ollama
import json


def extract_symptoms(complaint):

    prompt = f"""
Extract the medical symptoms explicitly mentioned in this patient statement.

Return ONLY JSON.
Do not diagnose.
Do not guess.
Use null for anything not mentioned.

Fields:

chest_pain
one_sided_weakness
breathlessness
stomach_pain
onset
severity
radiation
sweating
dizziness
nausea
speech_difficulty
facial_drooping
vision_change
fever

Patient statement:
{complaint}
"""

    response = ollama.chat(
        model="qwen3:4b-instruct",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
        )

    return json.loads(response["message"]["content"])
