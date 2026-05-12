import pandas as pd
import numpy as np
import joblib
import os
os.makedirs("models", exist_ok=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# LOAD DATA
# -------------------------------

# Clean score
df = pd.read_csv("data/student_learning_data.csv")
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna()

# -------------------------------
# SUBJECT → TOPIC ORDER
# -------------------------------
NEXT_TOPIC_MAP = {
    "OOP": "Functions",
    "Functions": "Loops",
    "Loops": "File Handling",
    "File Handling": None,

    "Queue": "Stack",
    "Stack": "Arrays",
    "Arrays": "Linked List",
    "Linked List": None,

    "Processes": "Scheduling",
    "Scheduling": "Threads",
    "Threads": None,

    "SQL": "Joins",
    "Joins": "Normalization",
    "Normalization": None,

    "Routing": "OSI Model",
    "OSI Model": "TCP/IP",
    "TCP/IP": None
}

# -------------------------------
# CREATE TARGET COLUMN
# -------------------------------
df["next_topic"] = df["topic"].map(NEXT_TOPIC_MAP)

# Remove last topics (no next topic)
df = df[df["next_topic"].notna()]

# -------------------------------
# FIXED ENCODING
# -------------------------------

topic_encoder = LabelEncoder()
subject_encoder = LabelEncoder()

# Combine ALL topics (IMPORTANT FIX)
all_topics = pd.concat([df["topic"], df["next_topic"]]).unique()

topic_encoder.fit(all_topics)

# Now transform safely
df["topic_enc"] = topic_encoder.transform(df["topic"])
df["next_topic_enc"] = topic_encoder.transform(df["next_topic"])

# Subject encoding
df["subject_enc"] = subject_encoder.fit_transform(df["subject"])

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df[["score", "time_spent", "attempts", "topic_enc", "subject_enc"]]
y = df["next_topic_enc"]

# -------------------------------
# TRAIN MODEL
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X, y)

# -------------------------------
# SAVE MODELS
# -------------------------------
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(topic_encoder, "models/topic_encoder.pkl")
joblib.dump(subject_encoder, "models/subject_encoder.pkl")

print("✅ Model trained and saved successfully!")