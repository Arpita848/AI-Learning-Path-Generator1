import pandas as pd
import joblib
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

# -------------------------------
# CREATE MODELS FOLDER
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/student_learning_data.csv")  # adjust path if needed

df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df.dropna()

# -------------------------------
# ENCODE LEARNING STYLE
# -------------------------------

learning_style_map = {
    "Visual": 0,
    "Audio-Visual": 1,
    "Kinesthetic": 2
}

df["learning_style_encoded"] = df["learning_style"].map(learning_style_map)

# -------------------------------
# FEATURES
# -------------------------------

X = df[[
    "score",
    "time_spent",
    "attempts",
    "learning_style_encoded"
]]
# -------------------------------
# SCALE DATA (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# TRAIN KMEANS
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

pickle.dump(kmeans, open("kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# -------------------------------
# CLUSTER SUMMARY
# -------------------------------
df["cluster"] = kmeans.predict(X_scaled)


df.to_csv("data/student_clustered.csv", index=False)

cluster_summary = df.groupby("cluster")[[
    "score",
    "time_spent",
    "attempts"
]].mean()

print("\nCluster Summary:")
print(cluster_summary)

# -------------------------------
# FIND CLUSTER MEAN SCORES
# -------------------------------
cluster_means = df.groupby("cluster")["score"].mean().sort_values()

print("\nCluster Means:")
print(cluster_means)


# -------------------------------
# CREATE CORRECT MAPPING
# -------------------------------
sorted_clusters = cluster_means.index.tolist()

cluster_labels = {
    sorted_clusters[0]: "Struggling Learner",
    sorted_clusters[1]: "Consistent Learner",
    sorted_clusters[2]: "Fast Learner"
}

level_map = {
    sorted_clusters[0]: "Weak",
    sorted_clusters[1]: "Intermediate",
    sorted_clusters[2]: "Strong"
}
df["learner_group"] = df["cluster"].map(cluster_labels)

# SAVE UPDATED DATASET
df.to_csv("data/student_clustered.csv", index=False)

CLUSTER_PROFILES = {
    sorted_clusters[0]: {
        "type": "Struggling Learner",
        "insight": "Needs conceptual clarity and guided practice",
        "strategy": "Focus on basics, watch tutorials, solve easy questions"
    },

    sorted_clusters[1]: {
        "type": "Consistent Learner",
        "insight": "Understands concepts but needs more consistency",
        "strategy": "Practice medium-level problems and revise regularly"
    },

    sorted_clusters[2]: {
        "type": "Advanced Learner",
        "insight": "Learns quickly and performs efficiently",
        "strategy": "Work on projects and advanced problem-solving"
    }
}

print("\nLevel Mapping:")
print(level_map)

joblib.dump(level_map, "models/level_map.pkl")

# -------------------------------
# SAVE MODELS
# -------------------------------
joblib.dump(kmeans, "models/kmeans.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(level_map, "models/level_map.pkl")
joblib.dump(CLUSTER_PROFILES, "models/cluster_profiles.pkl")

print("KMeans + Level Mapping + Profiles saved!")