import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_parquet(
    "hf://datasets/sweatSmile/medical-symptom-triage-csv/data/train-00000-of-00001.parquet"
)

df = df[
    ["symptom_description", "urgency_level", "primary_specialty"]
].dropna()


# ============================================================
# 2. CREATE DOMAIN LABELS
# ============================================================

domain_mapping = {
    "Cardiology": "Cardiovascular",
    "Neurology": "Neurological",
    "Pulmonology": "Respiratory",
    "Gastroenterology": "Gastrointestinal",
    "Dermatology": "Dermatological",
    "Orthopedics": "Musculoskeletal",
    "Urology": "Urinary",
    "Endocrinology": "Endocrine",
    "Mental Health": "Mental Health"
}

df["domain"] = df["primary_specialty"].map(domain_mapping)

# Remove Emergency Medicine because it is an urgency/routing
# category rather than a clinical domain we want the model to predict.
df = df.dropna(subset=["domain"])


# ============================================================
# 3. TRAIN URGENCY MODEL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["symptom_description"],
    df["urgency_level"],
    test_size=0.2,
    random_state=42,
    stratify=df["urgency_level"]
)

urgency_model = Pipeline([
    ("text_to_numbers", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight={
            "Emergency": 3,
            "Urgent": 1,
            "Observation": 1,
            "Routine": 1
        }
    ))
])

urgency_model.fit(X_train, y_train)

urgency_predictions = urgency_model.predict(X_test)


# ============================================================
# 4. TRAIN DOMAIN MODEL
# ============================================================

X_train_domain, X_test_domain, y_train_domain, y_test_domain = train_test_split(
    df["symptom_description"],
    df["domain"],
    test_size=0.2,
    random_state=42,
    stratify=df["domain"]
)

domain_model = Pipeline([
    ("text_to_numbers", TfidfVectorizer()),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

domain_model.fit(X_train_domain, y_train_domain)

domain_predictions = domain_model.predict(X_test_domain)


# ============================================================
# 5. SAFETY LAYER
# ============================================================

def safety_check(complaint, predicted_urgency, predicted_domain):

    text = complaint.lower()

    emergency_signs = [
        "severe chest pain",
        "crushing chest pain",
        "difficulty breathing",
        "struggling to breathe",
        "cannot breathe",
        "can't breathe",
        "one side of my body",
        "difficulty speaking",
        "unconscious",
        "severe bleeding"
    ]

    for sign in emergency_signs:

        if sign in text:

            # Cardiovascular red flags
            if (
                "chest pain" in text
                or "sweating" in text
                or "left arm" in text
                or "jaw" in text
            ):
                predicted_domain = "Cardiovascular"

            # Neurological red flags
            elif (
                "one side of my body" in text
                or "difficulty speaking" in text
            ):
                predicted_domain = "Neurological"

            # Respiratory red flags
            elif (
                "difficulty breathing" in text
                or "struggling to breathe" in text
                or "cannot breathe" in text
                or "can't breathe" in text
            ):
                predicted_domain = "Respiratory"

            return {
                "urgency": "Emergency",
                "domain": predicted_domain,
                "action": "Immediate medical assistance",
                "safety_override": True,
                "reason": sign
            }

    return {
        "urgency": predicted_urgency,
        "domain": predicted_domain,
        "action": "Continue with normal assessment",
        "safety_override": False,
        "reason": None
    }


# ============================================================
# 6. COMPLETE ACUTE ASSESSMENT FUNCTION
# ============================================================

def assess_complaint(complaint):

    predicted_urgency = urgency_model.predict(
        [complaint]
    )[0]

    predicted_domain = domain_model.predict(
        [complaint]
    )[0]

    final_result = safety_check(
        complaint,
        predicted_urgency,
        predicted_domain
    )

    return final_result


# ============================================================
# 7. ONLY RUN TESTS WHEN THIS FILE IS RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # Urgency model evaluation
    # -----------------------------

    print("URGENCY MODEL")
    print("-----------------------------")
    print(
        "Accuracy:",
        accuracy_score(y_test, urgency_predictions)
    )

    print(
        classification_report(
            y_test,
            urgency_predictions,
            zero_division=0
        )
    )


    # -----------------------------
    # Domain model evaluation
    # -----------------------------

    print("DOMAIN MODEL")
    print("-----------------------------")
    print(
        "Accuracy:",
        accuracy_score(
            y_test_domain,
            domain_predictions
        )
    )

    print(
        classification_report(
            y_test_domain,
            domain_predictions,
            zero_division=0
        )
    )


    # -----------------------------
    # Test complaints
    # -----------------------------

    test_complaints = [

        "I have severe chest pain with sweating and trouble breathing",

        "I suddenly have weakness on one side of my body and difficulty speaking",

        "I am coughing badly and struggling to breathe",

        "I have severe stomach pain and vomiting"
    ]


    for complaint in test_complaints:

        result = assess_complaint(complaint)

        print("\n========================================")
        print("Complaint:", complaint)
        print("Urgency:", result["urgency"])
        print("Domain:", result["domain"])
        print("Action:", result["action"])
        print("Safety override:", result["safety_override"])
        print("Reason:", result["reason"])