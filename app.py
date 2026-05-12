from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


# -------------------------------
# LOAD MODELS
# -------------------------------
kmeans = joblib.load("models/kmeans.pkl")
rf_model = joblib.load("models/random_forest.pkl")

subject_encoder = joblib.load("models/subject_encoder.pkl")
topic_encoder = joblib.load("models/topic_encoder.pkl")
style_encoder = joblib.load("models/style_encoder.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

scaler = joblib.load("models/scaler.pkl")
level_map = joblib.load("models/level_map.pkl")
cluster_profiles = joblib.load("models/cluster_profiles.pkl")

def generate_weekly_plan(topic, next_topic, score):

    LAST_TOPICS = [
        "File Handling",
        "Linked List",
        "Threads",
        "Normalization",
        "TCP/IP"
    ]

    # -----------------------------------
    # HIGH PERFORMERS (80+)
    # -----------------------------------

    if score >= 80:

        # LAST TOPIC OF SUBJECT
        if topic in LAST_TOPICS:

            subject_name = {
                "File Handling": "Python",
                "Linked List": "DSA",
                "Threads": "OS",
                "Normalization": "DBMS",
                "TCP/IP": "CN"
            }

            subject_full = subject_name.get(topic)

            plan = [
                f"Day 1: Study advanced concepts of {topic}",
                f"Day 2: Revise important concepts from all {subject_full} topics",
                "Day 3: Solve quizzes and practice exercises",
                "Day 4: Take a subject-level mock test and analyze mistakes",
                f"Day 5: Practice advanced-level applications of {subject_full}",
                "Bonus Task: Build mini-projects or solve advanced applications"
            ]

        # NORMAL HIGH SCORE
        else:

            plan = [
                f"Day 1: Study advanced concepts of {topic}",
                f"Day 2: Start learning basics of {next_topic}",
                f"Day 3: Practice beginner-level problems on {next_topic}",
                f"Day 4: Revise important concepts of {next_topic}",
                f"Day 5: Solve quizzes and exercises on {next_topic}",
                "Bonus Task: Build mini-projects or solve advanced applications"
            ]

    # -----------------------------------
    # MEDIUM PERFORMERS (45–79)
    # -----------------------------------

    elif score >= 45:

        plan = [
            f"Day 1: Revise important concepts of {topic}",
            f"Day 2: Solve practice questions and exercises on {topic}",
            f"Day 3: Take a topic-wise mock test for {topic}",
            f"Day 4: Start learning basics of {next_topic}",
            f"Day 5: Practice beginner-level problems on {next_topic}"
        ]

    # -----------------------------------
    # LOW PERFORMERS (<45)
    # -----------------------------------

    else:

        plan = [
            f"Day 1: Learn fundamentals and basics of {topic}",
            f"Day 2: Practice beginner-level questions on {topic}",
            f"Day 3: Revise core concepts and notes of {topic}",
            f"Day 4: Solve guided quizzes and exercises on {topic}",
            f"Day 5: Take a revision test and analyze mistakes in {topic}",
            "Bonus Task: Spend at least 30 minutes revising mistakes daily"
        ]

    return plan

# -------------------------------
# API ROUTE
# -------------------------------
@app.route("/get_learning_path", methods=["POST"])
def get_learning_path():
    data = request.json

    score = data["score"]
    time_spent = data["time_spent"]
    attempts = data["attempts"]
    topic = data["topic"]
    subject = data["subject"]

    learning_style = data["learning_style"]

    # -------------------------------
    # LEARNING STYLE ENCODING
    # -------------------------------

    learning_style_map = {
        "Visual": 0,
        "Audio-Visual": 1,
        "Kinesthetic": 2
    }

    learning_style_encoded = learning_style_map.get(
        learning_style,
        1
    )


    subject_encoded = subject_encoder.transform([subject])[0]

    current_topic_encoded = topic_encoder.transform([topic])[0]

    learning_style_map = {
        "Visual": 0,
        "Audio-Visual": 1,
        "Kinesthetic": 2
    }

    learning_style_encoded = learning_style_map.get(
        learning_style,
        1
    )
    # -------------------------------
    # 1️⃣ K-MEANS → LEVEL
    # -------------------------------
    X_input = pd.DataFrame([{
        "score": score,
        "time_spent": time_spent,
        "attempts": attempts,
        "learning_style_encoded": learning_style_encoded
    }])
    X_scaled = scaler.transform(X_input)
    cluster = kmeans.predict(X_scaled)[0]

    # IMPORTANT: adjust mapping after testing
    level = level_map.get(cluster, "Intermediate")

    profile = cluster_profiles.get(cluster, {})

    # -------------------------------
    # 2️⃣ RANDOM FOREST → NEXT TOPIC
    # -------------------------------
    rf_input = pd.DataFrame([{
        "subject": subject_encoded,
        "current_topic": current_topic_encoded,
        "score": score,
        "time_spent": time_spent,
        "attempts": attempts,
        "learning_style": learning_style_encoded
    }])
    pred = rf_model.predict(rf_input)[0]

    next_topic = target_encoder.inverse_transform(
        [pred]
    )[0]
    # -------------------------------
    # 3️⃣ RULE + ML COMBINED LOGIC
    # -------------------------------
    if level == "Weak":
        action = f"""
        • Study {topic} from beginner level  
        • Revise fundamentals  
        • Practice basic questions  
        • Spend more time on this topic  
        """

        difficulty = "Easy"

    elif level == "Intermediate":
        action = f"""
        • Practice more questions on {topic}  
        • Solve medium-level problems  
        • Strengthen concepts  
        """

        difficulty = "Medium"

    elif level == "Strong":
        if topic in ["OOP", "Functions", "Loops", "Arrays", "Linked List"]:
            action = f"""
            • Build a project using {topic}  
            • Apply concepts practically  
            """
        else:
            action = f"""
            • Study advanced concepts of {topic}  
            """

        difficulty = "Hard"
    else:
            action = "Continue practicing consistently"
            difficulty = "Medium"
    
    weekly_plan = generate_weekly_plan(
        topic,
        next_topic,
        score
    )

    return jsonify({
        "level": level,

        "learner_type": profile.get("type"),
        "learning_insight": profile.get("insight"),
        "learning_strategy": profile.get("strategy"),

        "next_topic": next_topic,
        "recommended_action": action,
        "difficulty": difficulty,
        "explanation": f"Student classified as {level} using K-Means and next topic predicted using Random Forest.",
        
        "plan": weekly_plan
    })
if __name__ == "__main__":
    app.run(port=5001, debug=True)