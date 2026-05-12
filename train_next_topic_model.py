import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("data/next_topic_dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------------
# LABEL ENCODERS
# -----------------------------------

subject_encoder = LabelEncoder()
topic_encoder = LabelEncoder()
style_encoder = LabelEncoder()
target_encoder = LabelEncoder()

# -----------------------------------
# ENCODE CATEGORICAL COLUMNS
# -----------------------------------

df["subject"] = subject_encoder.fit_transform(df["subject"])

df["current_topic"] = topic_encoder.fit_transform(
    df["current_topic"]
)

df["learning_style"] = style_encoder.fit_transform(
    df["learning_style"]
)


# TARGET LABEL
df["next_topic"] = target_encoder.fit_transform(
    df["next_topic"]
)

# -----------------------------------
# FEATURES (X)
# -----------------------------------

X = df[
    [
        "subject",
        "current_topic",
        "score",
        "time_spent",
        "attempts",
        "learning_style"
    ]
]

# -----------------------------------
# TARGET (y)
# -----------------------------------

y = df["next_topic"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------------
# RANDOM FOREST MODEL
# -----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------

rf_model.fit(X_train, y_train)

print("\nModel Training Completed")

# -----------------------------------
# PREDICTIONS
# -----------------------------------

y_pred = rf_model.predict(X_test)

# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# -----------------------------------
# SAVE MODEL
# -----------------------------------

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

# -----------------------------------
# SAVE ENCODERS
# -----------------------------------

joblib.dump(
    subject_encoder,
    "models/subject_encoder.pkl"
)

joblib.dump(
    topic_encoder,
    "models/topic_encoder.pkl"
)

joblib.dump(
    style_encoder,
    "models/style_encoder.pkl"
)

joblib.dump(
    target_encoder,
    "models/target_encoder.pkl"
)

print("\nModel and Encoders Saved Successfully")