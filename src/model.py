import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
def load_data():
    
    return pd.read_csv("../data/student_learning_data.csv")


# -------------------------------
# 1️⃣ K-MEANS CLUSTERING
# -------------------------------
def train_kmeans(df):
    features = df[['score', 'time_spent', 'attempts']]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features)

    # Map clusters to labels
    cluster_map = {
        0: "Beginner",
        1: "Intermediate",
        2: "Advanced"
    }

    df['level'] = df['cluster'].map(cluster_map)

    joblib.dump(kmeans, "../models/kmeans.pkl")

    print("✅ KMeans model trained and saved")
    return df


# -------------------------------
# 2️⃣ RANDOM FOREST CLASSIFIER
# -------------------------------
def train_classifier(df):
    # Features
    X = df[['score', 'time_spent', 'attempts']]

    # Target → topic prediction
    y = df['topic']

    # Encode topic
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    joblib.dump(le, "../models/topic_encoder.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("✅ Classification Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "../models/random_forest.pkl")

    print("✅ Random Forest model saved")


# -------------------------------
# MAIN PIPELINE
# -------------------------------
if __name__ == "__main__":
    df = load_data()

    df = train_kmeans(df)
    train_classifier(df)

    # Save updated dataset
    df.to_csv("../data/processed_data.csv", index=False)

    print("✅ Processed data saved!")