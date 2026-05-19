import streamlit as st

st.set_page_config(
    page_title="AI Learning Path Generator", 
    layout="wide",
    initial_sidebar_state="expanded"

)

import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import copy
import joblib
from openai import OpenAI
from datetime import datetime
import os
from dotenv import load_dotenv





df = pd.read_csv("data/student_clustered.csv")
if "quiz_generated" not in st.session_state:
    st.session_state.quiz_generated = False
# -------------------------------
# 📚 SUBJECT → TOPIC MAPPING
# -------------------------------
SUBJECT_TOPICS = {
    "Python": ["OOP", "Functions", "Loops", "File Handling"],
    "DSA": ["Queue", "Stack", "Arrays", "Linked List"],
    "OS": ["Processes", "Scheduling", "Threads"],
    "DBMS": ["SQL", "Joins", "Normalization"],
    "CN": ["Routing", "OSI Model", "TCP/IP"]
}

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
#  STUDENT DATA FUNCTIONS
# -------------------------------
STUDENT_FILE = "data/student_data.csv"

def load_students():

    try:
        df = pd.read_csv(STUDENT_FILE)

    except FileNotFoundError:

        df = pd.DataFrame(columns=[
            "student_id",
            "subject",
            "topic",
            "score",
            "time_spent",
            "attempts",
            "learning_style"
        ])

        # CREATE FILE AUTOMATICALLY
        df.to_csv(STUDENT_FILE, index=False)

    return df
df = load_students()
df["score"] = pd.to_numeric(df["score"], errors="coerce")

students_df = df.copy()   #  NOW df exists
students_df["score"] = pd.to_numeric(students_df["score"], errors="coerce")
def save_students(df):
    df.to_csv(STUDENT_FILE, index=False)
def generate_ai_quiz(
    topic,
    difficulty,
    learning_style
):

    prompt = f"""
    Generate 5 multiple choice questions.

    Topic: {topic}

    Difficulty: {difficulty}

    Learning Style: {learning_style}

    Return ONLY valid JSON.

    Format:

    [
      {{
        "question": "Question here",
        "options": [
          "Option A",
          "Option B",
          "Option C",
          "Option D"
        ],
        "answer": "Correct Option"
      }}
    ]
    """

    try:
            st.error("API key missing. Check .env file")
            return []

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            quiz_text = response.json()["choices"][0]["message"]["content"]
        else:
            st.error(response.text)
            return []

        # Remove markdown if AI returns ```json
        quiz_text = (
            quiz_text
            .replace("```json", "")
            .replace("```", "")
            .replace("json", "", 1)
            .strip()
        )

        start = quiz_text.find("[")
        end = quiz_text.rfind("]") + 1

        quiz_text = quiz_text[start:end]

        quiz_data = json.loads(quiz_text)

        return quiz_data

    except Exception as e:

        st.error(f"""
        Quiz generation failed.

        Error:
        {str(e)}
        """)

        return []
st.markdown("""
<style>

/* -----------------------------------
   🌈 MAIN APP BACKGROUND
----------------------------------- */

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #edf4ff 0%,
        #f8fbff 40%,
        #fffbea 100%
    );
    background-attachment: fixed;
    color: #1e293b;
}

/* Remove white main container */

.main {
    background: transparent !important;
}

.block-container {
    background: transparent !important;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* -----------------------------------
   🧊 GLASSMORPHISM CARDS
----------------------------------- */

div[data-testid="stVerticalBlock"] > div:has(div.stMetric),
div[data-testid="stExpander"],
div[data-testid="stDataFrame"],
div[data-testid="element-container"] {

    background: rgba(255, 255, 255, 0.28) !important;

    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);

    border-radius: 18px;

    border: 1px solid rgba(255,255,255,0.35);

    padding: 18px;

    box-shadow:
        0 8px 32px rgba(31, 38, 135, 0.08);

    margin-bottom: 16px;
}

/* -----------------------------------
   📚 HEADINGS
----------------------------------- */

h1, h2, h3 {
    color: #1e3a5f !important;
    font-weight: 700 !important;
}

/* -----------------------------------
   📝 TEXT
----------------------------------- */

p, label, div {
    color: #334155;
}

/* -----------------------------------
   🎛 SIDEBAR
----------------------------------- */

section[data-testid="stSidebar"] {

    background: linear-gradient(
        180deg,
        rgba(219,234,254,0.95),
        rgba(239,246,255,0.92)
    );

    backdrop-filter: blur(12px);

    border-right: 1px solid rgba(255,255,255,0.4);
}

/* Sidebar text */

section[data-testid="stSidebar"] * {
    color: #1e293b !important;
}

/* -----------------------------------
   🧾 INPUTS
----------------------------------- */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {

    background: rgba(255,255,255,0.65) !important;

    border-radius: 12px !important;

    border: 1px solid rgba(255,255,255,0.4) !important;

    backdrop-filter: blur(6px);
}

/* -----------------------------------
   📊 METRICS
----------------------------------- */

[data-testid="stMetric"] {

    background: rgba(255,255,255,0.25);

    border-radius: 16px;

    padding: 15px;

    backdrop-filter: blur(10px);
}

[data-testid="stMetricValue"] {
    color: #2563eb !important;
    font-weight: 700;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

/* -----------------------------------
   📑 TABS
----------------------------------- */

button[data-baseweb="tab"] {

    background: rgba(255,255,255,0.25) !important;

    backdrop-filter: blur(10px);

    border-radius: 12px 12px 0 0 !important;

    margin-right: 6px;

    padding: 10px 18px !important;

    font-size: 17px !important;

    font-weight: 600 !important;

    color: #334155 !important;
}

/* Active tab */

button[aria-selected="true"] {

    background: rgba(191,219,254,0.6) !important;

    color: #2563eb !important;
}

/* -----------------------------------
   🔘 BUTTONS
----------------------------------- */

.stButton > button {

    background: linear-gradient(
        90deg,
        #60a5fa,
        #3b82f6
    );

    color: white;

    border: none;

    border-radius: 14px;

    height: 3em;

    font-weight: 600;

    transition: 0.3s ease;

    box-shadow:
        0 6px 18px rgba(59,130,246,0.25);
}

/* Hover effect */

.stButton > button:hover {

    transform: translateY(-2px);

    opacity: 0.95;
}

/* -----------------------------------
   📈 PROGRESS BAR
----------------------------------- */

div[data-testid="stProgressBar"] > div {

    background: linear-gradient(
        90deg,
        #60a5fa,
        #3b82f6
    ) !important;
}

/* -----------------------------------
   📋 DATAFRAME
----------------------------------- */

[data-testid="stDataFrame"] {

    overflow: hidden;
}

/* -----------------------------------
   🎯 ALERTS
----------------------------------- */

.stSuccess,
.stInfo,
.stWarning,
.stError {

    border-radius: 14px;

    backdrop-filter: blur(8px);
}

/* -----------------------------------
   ✨ Smooth Animation
----------------------------------- */

* {
    transition: all 0.2s ease;
}
/* -----------------------------------
   🎨 SELECTBOX / DROPDOWN FIX
----------------------------------- */

/* Selected value */

div[data-baseweb="select"] > div {

    background: rgba(255,255,255,0.75) !important;

    color: #1e293b !important;

    border-radius: 12px !important;

    border: 1px solid rgba(255,255,255,0.45) !important;
}

/* Dropdown popup menu */

div[role="listbox"] {

    background: #f8fbff !important;

    border-radius: 14px !important;

    border: 1px solid #dbeafe !important;

    backdrop-filter: blur(10px);
}

/* Dropdown options */

div[role="option"] {

    background: #f8fbff !important;

    color: #1e293b !important;

    font-weight: 500;
}

/* Hovered option */

div[role="option"]:hover {

    background: #dbeafe !important;

    color: #2563eb !important;
}

/* Selected option */

div[aria-selected="true"] {

    background: #bfdbfe !important;

    color: #1d4ed8 !important;

    font-weight: 600;
}

/* Placeholder text */

div[data-baseweb="select"] span {

    color: #334155 !important;
}

/* Input labels */

label {

    color: #1e293b !important;

    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)



df["score"] = pd.to_numeric(df["score"], errors="coerce")
if os.path.exists("data/quiz_data.json"):
    with open("data/quiz_data.json") as f:
        quiz_data = json.load(f)
else:
    quiz_data = []

if "analyze" not in st.session_state:
    st.session_state.analyze = False

#  Initialize globally (FIX)
weak_topics = pd.Series(dtype=float)
strong_topics = pd.Series(dtype=float)

# -------------------------------
# CONFIG
# -------------------------------
API_URL = "http://127.0.0.1:5001"

if "analyze" not in st.session_state:
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []

    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    st.session_state.analyze = False
# -------------------------------
# SIDEBAR
# -------------------------------

# -------------------------------
# 🔐 LOGIN + REGISTRATION SYSTEM
# -------------------------------

USER_FILE = "data/users.csv"
COMMON_PASSWORD = "IBM"

def load_users():
    try:
        return pd.read_csv(USER_FILE)
    except FileNotFoundError:
        users = pd.DataFrame(columns=[
            "student_id",
            "name",
            "password",
            "learning_style"
        ])
        users.to_csv(USER_FILE, index=False)
        return users

def save_users(users):
    users.to_csv(USER_FILE, index=False)

users_df = load_users()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "student_id" not in st.session_state:
    st.session_state.student_id = None

if "student_name" not in st.session_state:
    st.session_state.student_name = ""

if not st.session_state.logged_in:

    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    .block-container {
        max-width: 560px;
        padding-top: 7rem;
    }

    .app-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #2563eb;
        margin-bottom: 8px;
    }

    .app-subtitle {
        text-align: center;
        font-size: 17px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .login-note {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="app-title">Your Personal Learning App</div>
    <div class="app-subtitle">
        Login or register to start your personalized learning journey
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        login_id = st.text_input(
            "Student ID",
            placeholder="Enter your student ID",
            key="login_id"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            key="login_password"
        )

        if st.button("Login", use_container_width=True):

            try:
                login_id_int = int(login_id)

                user_match = users_df[
                    (users_df["student_id"] == login_id_int) &
                    (users_df["password"] == login_password)
                ]

                # fallback for old users using IBM password
                old_user_exists = not students_df[
                    students_df["student_id"] == login_id_int
                ].empty

                if not user_match.empty:
                    st.session_state.logged_in = True
                    st.session_state.student_id = login_id_int
                    st.session_state.student_name = user_match.iloc[0]["name"]
                    st.rerun()

                elif old_user_exists and login_password == COMMON_PASSWORD:
                    st.session_state.logged_in = True
                    st.session_state.student_id = login_id_int
                    st.session_state.student_name = f"Student {login_id_int}"
                    st.rerun()

                else:
                    st.error("Invalid Student ID or Password")

            except ValueError:
                st.error("Student ID must be numeric")

    with register_tab:
        reg_name = st.text_input(
            "Full Name",
            placeholder="Enter your name",
            key="reg_name"
        )

        reg_id = st.text_input(
            "Create Student ID",
            placeholder="Example: 101",
            key="reg_id"
        )

        reg_password = st.text_input(
            "Create Password",
            type="password",
            placeholder="Create password",
            key="reg_password"
        )

        reg_style = st.selectbox(
            "Preferred Learning Style",
            ["Visual", "Audio-Visual", "Kinesthetic"],
            key="reg_style"
        )

        if st.button("Register", use_container_width=True):

            try:
                reg_id_int = int(reg_id)

                already_exists = not users_df[
                    users_df["student_id"] == reg_id_int
                ].empty

                if already_exists:
                    st.error("This Student ID already exists. Please login.")

                elif reg_name.strip() == "" or reg_password.strip() == "":
                    st.error("Name and password are required.")

                else:
                    new_user = {
                        "student_id": reg_id_int,
                        "name": reg_name,
                        "password": reg_password,
                        "learning_style": reg_style
                    }

                    users_df = pd.concat(
                        [users_df, pd.DataFrame([new_user])],
                        ignore_index=True
                    )

                    save_users(users_df)

                    st.session_state.logged_in = True
                    st.session_state.student_id = reg_id_int
                    st.session_state.student_name = reg_name

                    st.success("Registration successful.")
                    st.rerun()

            except ValueError:
                st.error("Student ID must be numeric")

    st.markdown("""
    <div class="login-note">
        Existing demo users can login using password: IBM
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# -------------------------------
# LOGGED-IN USER DETAILS
# -------------------------------

login_success = st.session_state.logged_in
student_id = st.session_state.student_id

# -------------------------------
# STUDENT HISTORY
# -------------------------------

if student_id:

    student_history = students_df[
        students_df["student_id"] == student_id
    ].copy()

else:

    student_history = pd.DataFrame()

st.sidebar.markdown(f"""
<h2>Student Dashboard</h2>
<p>Welcome, <b>{st.session_state.student_name or 'Student'}</b></p>
<hr>
""", unsafe_allow_html=True)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.student_id = None
    st.session_state.student_name = ""
    st.rerun()
score = st.sidebar.slider(
    "Score",
    0,
    100,
    50
)

time_spent = st.sidebar.slider(
    "Time Spent (mins)",
    10,
    180,
    60
)

attempts = st.sidebar.slider(
    "Attempts",
    1,
    5,
    2
)

learning_style = st.sidebar.selectbox(
    "Learning Style",
    ["Visual", "Audio-Visual", "Kinesthetic"]
)

subject = st.sidebar.selectbox(
    "Subject",
    list(SUBJECT_TOPICS.keys())
)

topic = st.sidebar.selectbox(
    "Topic",
    ["All Topics"] + SUBJECT_TOPICS[subject]
)

if topic == "All Topics":

    st.sidebar.info(
        "You are analyzing subject-level performance"
    )

# -------------------------------
# BUTTONS
# -------------------------------

if st.sidebar.button(
    "Analyze Student",
    key="analyze_btn"
):

    st.session_state.analyze = True

if st.sidebar.button(
    "Reset",
    key="reset_btn"
):

    st.session_state.analyze = False

    st.rerun()

# -------------------------------
# TITLE
# -------------------------------
st.markdown("""
<h1 style='text-align: center; color: #2563eb;'>
AI Personalized Learning Path Generator
</h1>
<p style='text-align: center; color: gray;'>
Smart AI system for adaptive learning & performance tracking
</p>
""", unsafe_allow_html=True)

topic_scores = pd.Series(dtype=float)

# -------------------------------
# Topic Performance (USE USER DATA ONLY)
# -------------------------------
st.subheader(" Topic Performance")

# ALWAYS DEFINE filtered_df FIRST
if not students_df.empty and student_id:

    if topic == "All Topics":
        filtered_df = students_df[
            (students_df["student_id"] == student_id) &
            (students_df["subject"] == subject)
        ]
    else:
        filtered_df = students_df[
            (students_df["student_id"] == student_id) &
            (students_df["subject"] == subject) &
            (students_df["topic"] == topic)
        ]
else:
    filtered_df = pd.DataFrame()

# SAFE TO USE
if not filtered_df.empty:
    topic_performance = filtered_df.groupby("topic")["score"].mean()
    st.bar_chart(topic_performance.sort_values(ascending=False))
else:
    st.info("No data available yet")
# -------------------------------
# LOAD KMEANS MODELS
# -------------------------------
kmeans = joblib.load("models/kmeans.pkl")
scaler = joblib.load("models/scaler.pkl")
# -------------------------------
# CREATE LEARNER GROUPS
# -------------------------------

if not students_df.empty and len(students_df) > 0:

    # Features needed for clustering
    required_cols = [
        "score",
        "time_spent",
        "attempts",
        "learning_style"
    ]

    if all(col in students_df.columns for col in required_cols):

        # Encode learning styles into numbers
        learning_style_map = {
            "Visual": 0,
            "Audio-Visual": 1,
            "Kinesthetic": 2
        }

        students_df["learning_style_encoded"] = (
            students_df["learning_style"]
            .map(learning_style_map)
        )

        # AI clustering features
        X = students_df[[
            "score",
            "time_spent",
            "attempts",
            "learning_style_encoded"
        ]]

        X_scaled = scaler.transform(X)

        students_df["cluster"] = kmeans.predict(X_scaled)

        # -----------------------------------
        # FIND CLUSTER MEAN SCORES
        # -----------------------------------

        cluster_means = (
            students_df
            .groupby("cluster")["score"]
            .mean()
            .sort_values()
        )

        sorted_clusters = cluster_means.index.tolist()

        # Lowest score cluster -> Weak
        # Middle score cluster -> Intermediate
        # Highest score cluster -> Strong

        cluster_name_map = {
            sorted_clusters[0]: "Struggling Learner",
            sorted_clusters[1]: "Consistent Learner",
            sorted_clusters[2]: "Advanced Learner"
        }

        # -----------------------------------
        # APPLY LEARNER LABELS
        # -----------------------------------

        students_df["learner_group"] = (
            students_df["cluster"]
            .map(cluster_name_map)
        )
else:
    st.info("No student data available yet")
# -------------------------------
# 🏆 Topic-wise Student Rank
# -------------------------------


if student_id and not students_df.empty and topic != "All Topics":

    # Filter students for selected topic
    topic_students = students_df[
        (students_df["subject"] == subject) &
        (students_df["topic"] == topic)
    ].copy()

    if not topic_students.empty:

        # Average score per student in that topic
        topic_ranks = (
            topic_students
            .groupby("student_id")["score"]
            .last()
            .sort_values(ascending=False)
            .reset_index()
        )

        # Assign ranks
        topic_ranks["rank"] = range(1, len(topic_ranks) + 1)

        # Find current student's rank
        student_rank_data = topic_ranks[
            topic_ranks["student_id"] == student_id
        ]
        if not student_rank_data.empty:
            student_rank = int(student_rank_data["rank"].values[0])
            total_students = len(topic_ranks)
            # -------------------------------
            # 🎨 Dynamic Rank Color
            # -------------------------------

            if student_rank == 1:
                rank_color = "#16a34a"   # green
            else:
                rank_color = "#eab308"   # yellow

            st.markdown(f"""
            <h3 style='
                font-weight:800;
                margin-bottom:10px;
            '>
            Your Rank in {topic} topic:
            <span style='color:{rank_color};'>
            {student_rank} / {total_students}
            </span>
            </h3>
            """, unsafe_allow_html=True)
            # -------------------------------
            # 📈 Topic Rank Performance Graph
            # -------------------------------

            fig_rank, ax_rank = plt.subplots(figsize=(9,4))

            # X-axis → Rank positions
            x_values = range(1, len(topic_ranks) + 1)

            # Y-axis → Scores
            y_values = topic_ranks["score"]

            # Line graph
            ax_rank.plot(x_values, y_values, marker='o', linewidth=3, label="Score")
            ax_rank.legend()

            # Highlight current student
            student_row = topic_ranks[
                topic_ranks["student_id"] == student_id
            ]

            if not student_row.empty:

                student_x = int(student_row["rank"].values[0])
                student_y = float(student_row["score"].values[0])


                ax_rank.annotate(
                    "You are here",
                    (student_x, student_y),
                    textcoords="offset points",
                    xytext=(0,18),
                    ha='center',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        fc="#e0f2fe",
                        alpha=0.9
                    )
                )

            # Labels
            ax_rank.set_title(
                f"{topic} Leaderboard",
                fontsize=16,
                fontweight='bold'
            )

            ax_rank.set_xlabel(
                "Student Rank",
                fontsize=13,
                fontweight='bold',
                fontstyle='italic'
            )
            ax_rank.set_xticks(list(x_values))
            ax_rank.set_xticklabels(
                [
                    f"Rank {r}\nID {sid}"
                    for r, sid in zip(
                        topic_ranks["rank"],
                        topic_ranks["student_id"]
                    )
                ],
                fontsize=9
            )
            ax_rank.set_ylabel(
                "Latest Topic Score",
                fontsize=13,
                fontweight='bold',
                fontstyle='italic'
            )

            # Grid for modern look
            ax_rank.grid(alpha=0.3)

            # Remove top/right borders
            ax_rank.spines['top'].set_visible(False)
            ax_rank.spines['right'].set_visible(False)

            st.pyplot(fig_rank)

        else:
            st.info("No ranking data available yet")
    else:
        st.warning("No students found for this topic")
elif topic == "All Topics":
    st.info("Select a specific topic to view ranking")

# -------------------------------
# ANALYZE
# -------------------------------
if st.session_state.analyze:
    
    data = {
        "score": score,
        "time_spent": time_spent,
        "attempts": attempts,
        "learning_style": learning_style,
        "subject": subject,
        "topic": topic if topic != "All Topics" else "General"
        }

    # API CALL
    try:
        response = requests.post(f"{API_URL}/get_learning_path", json=data, timeout=5)
    
        if response.status_code == 200:
            result = response.json()
        else:
            result = {}

    except Exception as e:
        st.error(f"API not working: {e}")
        result = {}

    if not result:
        result = {
            "level": "Intermediate" if score >= 40 else "Weak",
            "next_topic": NEXT_TOPIC_MAP.get(topic, "N/A"),
            "recommended_action": "Practice more questions",
            "difficulty": "Medium",
            "explanation": "Fallback logic used (API failed)"
        }
    # -------------------------------
    # SAVE STUDENT DATA
    # -------------------------------
    if student_id:
        
        students_df = students_df[
            ~((students_df["student_id"] == student_id) &
              (students_df["subject"] == subject) &
              (students_df["topic"] == (topic if topic != "All Topics" else "General")))
    ]
        
        new_row = {
            "student_id": student_id,
             "subject": subject,
             "topic": topic if topic != "All Topics" else "General",
             "score": score,
             "time_spent": time_spent,
             "attempts": attempts,
             "learning_style": learning_style
        }

        students_df = pd.concat([students_df, pd.DataFrame([new_row])], ignore_index=True)
        save_students(students_df)

    # REFRESH FILTERED DATA AFTER SAVE
    if student_id:
        if topic == "All Topics":
            filtered_df = students_df[
                (students_df["student_id"] == student_id) &
                (students_df["subject"] == subject)
            ]
        else:
            filtered_df = students_df[
                (students_df["student_id"] == student_id) &
                (students_df["subject"] == subject) &
                (students_df["topic"] == topic)
            ]
    # -------------------------------
    # WEAK TOPIC DETECTION 
    # -------------------------------

    if not filtered_df.empty:
        topic_performance = filtered_df.groupby("topic")["score"].mean()
        weak_topics = topic_performance[topic_performance < 40]
        strong_topics = topic_performance[topic_performance >= 70]
    else:
        weak_topics = pd.Series(dtype=float)
        strong_topics = pd.Series(dtype=float)

    # PRIORITY TOPICS FROM HISTORY
    priority_topics = []

    if not student_history.empty:

        topic_avg_scores = (
            student_history
            .groupby("topic")["score"]
            .mean()
            .sort_values()
        )

        for topic_name, avg_score in topic_avg_scores.items():

            if avg_score < 75:

                priority_topics.append(
                    (topic_name, round(avg_score, 2))
                )
    # -------------------------------
    # KPI CARDS
    # -------------------------------

    # Safe defaults (avoid errors if not defined)
    weak_topics = weak_topics if 'weak_topics' in locals() else []
    strong_topics = strong_topics if 'strong_topics' in locals() else []
    col1, col2, col3 = st.columns(3)

    # Total Records
    col1.markdown(f"""
    <div class="card">
    <h3>Total Records</h3>
    <h2>{len(students_df)}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Weak Topics
    col2.markdown(f"""
    <div class="card">
    <h3>Weak Topics</h3>
    <h2>{len(weak_topics)}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Strong Topics
    col3.markdown(f"""
    <div class="card">
    <h3>Strong Topics</h3>
    <h2>{len(strong_topics)}</h2>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # QUIZ RECOMMENDATION LOGIC
    # -------------------------------
    recommended_quizzes = []

    for weak_topic in weak_topics.index:
        for quiz in quiz_data:
            if quiz["topic"].lower() == weak_topic.lower():
                
                # Dynamic difficulty adjustment
                quiz_copy = copy.deepcopy(quiz)
                if score < 40:
                    quiz_copy["difficulty"] = "Easy"
                    quiz_copy["questions"] = 5
                elif score < 70:
                    quiz_copy["difficulty"] = "Medium"
                    quiz_copy["questions"] = 7
                else:
                    quiz_copy["difficulty"] = "Hard"
                    quiz_copy["questions"] = 10

                recommended_quizzes.append(quiz_copy)
                
    # -------------------------------
    # HEATMAP DATA
    # -------------------------------
    heatmap_data = filtered_df.pivot_table(
        index="topic",
        columns="learning_style",
        values="score",
        aggfunc="mean"
    )
    # -------------------------------
    # TABS
    # -------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "AI Insights",
        "Analytics",
        "Learning Plan",
        "Quiz Arena",
        "Achievements & Badges"
    ])
    # -------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------
    with tab1:
        st.subheader("Student Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Level", result.get("level", "N/A"))
        col2.metric("Score", score)
        col3.metric("Learning Style", learning_style)

        st.subheader("Performance Insight")

        if score < 40:
            st.error("Weak in fundamentals")
        elif score < 70:
            st.warning("Needs practice")
        else:
            st.success("Strong performance")

        st.subheader("Progress")
        st.progress(score / 100)
        st.markdown("<h2 style='text-align:center;'> Your Progress</h2>", unsafe_allow_html=True)

        if not student_history.empty:
            progress = student_history.groupby("topic")["score"].mean()
            st.line_chart(progress)
        else:
            st.info("No past data available")
        student_history = students_df[
            students_df["student_id"] == student_id
        ]

    # -------------------------------
    # TAB 2: AI INSIGHTS
    # -------------------------------
    with tab2:

        st.markdown("""
        <h1 style='text-align:center; color:#2563eb;'>
        AI Student Insights
        </h1>
        """, unsafe_allow_html=True)

        # -----------------------------------
        # AI Learner Performance Overview
        # -----------------------------------

        st.subheader("AI Learner Performance Overview")

        if not students_df.empty:

            group_scores = (
                students_df
                .groupby("learner_group")["score"]
                .mean()
            )

            fig2, ax2 = plt.subplots(figsize=(7,4))

            group_scores.plot(
                kind="bar",
                ax=ax2
            )

            ax2.set_xlabel(
                "Learner Categories",
                fontsize=13,
                fontweight='bold',
                fontstyle='italic'
            )

            ax2.set_ylabel(
                "Average Student Score",
                fontsize=13,
                fontweight='bold',
                fontstyle='italic'
            )
            ax2.set_title("Performance Across Learner Categories")

            st.pyplot(fig2)

        # -----------------------------------
        # AI LEARNING INSIGHT
        # -----------------------------------
        student_data = students_df[
            students_df["student_id"] == student_id
        ].copy() if student_id else pd.DataFrame()

        if student_id and not student_data.empty:

            student_data["score"] = pd.to_numeric(student_data["score"], errors="coerce")

            avg_score = student_data["score"].mean()
            min_score = student_data["score"].min()
            max_score = student_data["score"].max()
            latest_score = student_data.iloc[-1]["score"]

            # clean NaN handling
            if pd.isna(avg_score):
                avg_score = 0
            if pd.isna(min_score):
                min_score = 0

            # AI-style classification (history + consistency)
            if avg_score >= 85 and min_score >= 70 and latest_score >= 75:
                current_group = "Advanced Learner"

            elif avg_score >= 60:
                current_group = "Consistent Learner"

            else:
                current_group = "Struggling Learner"
            # -----------------------------------
            # Smart Override Based on Score
            # -----------------------------------
            if latest_score >= 85:
                current_group = "Advanced Learner"

            elif latest_score >= 50 and current_group == "Struggling Learner":
                current_group = "Consistent Learner"

            st.subheader("AI Learning Insight")

            if current_group == "Advanced Learner":

                st.success("""
                You are currently performing as an 
                Advanced Learner.

                Your consistency and strong scores indicate 
                excellent progress. Continue practicing 
                advanced-level problems to maintain your position.
                """)

            elif current_group == "Consistent Learner":

                st.info("""
                You are currently classified as a 
                Consistent Learner.

                Improving the priority topics below can help 
                you move into the Advanced Learner category.
                """)

            else:

                st.warning("""
                You are currently classified as a 
                Struggling Learner.

                Focus on strengthening the priority topics 
                below to improve your learning performance.
                """)

            # -----------------------------------
            # Priority Topics
            # -----------------------------------

            st.subheader("Priority Topics")

            if priority_topics:

                for topic_name, score_value in priority_topics:

                    st.markdown(f"""
                    <div style="
                        padding:12px;
                        border-radius:12px;
                        margin-bottom:10px;
                        background:rgba(255,255,255,0.25);
                        backdrop-filter:blur(8px);
                        border-left:5px solid #3b82f6;
                    ">
                        <b>{topic_name}</b><br>
                        Current Score: {score_value}
                    </div>
                    """, unsafe_allow_html=True)

            else:

                st.success("No priority topics detected.")

        # -----------------------------------
        # Weak Topics
        # -----------------------------------

        st.subheader("Weak Topics")

        if not weak_topics.empty:

            for topic in weak_topics.index:

                st.warning(
                    f"{topic} needs improvement"
                )

        else:

            st.success("No weak topics detected")

        # -----------------------------------
        # Strong Topics
        # -----------------------------------

        st.subheader("Strong Topics")

        if not strong_topics.empty:

            st.success(", ".join(strong_topics.index))

        else:

            st.info("No strong topics yet")
        
        # -------------------------------
        # WEAK vs STRONG DISTRIBUTION
        # -------------------------------
        st.markdown("<h2 style='text-align:center;'> Weak vs Strong Distribution</h2>", unsafe_allow_html=True)
        weak_count = len(weak_topics)
        strong_count = len(strong_topics)
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(
            ["Weak", "Strong"],
            [weak_count, strong_count],
            color=["#ef4444", "#22c55e"]
        )
        ax.set_title("Weak vs Strong", fontsize=10)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.pyplot(fig)
    # -------------------------------
    # TAB 3: ANALYTICS
    # -------------------------------
    with tab3:
        st.markdown("""
        <h1 style='text-align:center; color:#4f46e5; margin-bottom:20px;'>
        Welcome to Your Analytics Dashboard
        </h1>
        """, unsafe_allow_html=True)
        
        # -------------------------------
        # TOP PERFORMERS
        # -------------------------------
        st.markdown("<h2 style='text-align:center;'>Top 10 Performers</h2>", unsafe_allow_html=True)
        if not students_df.empty:
            top_students = students_df.groupby("student_id")["score"].mean().nlargest(10)
            fig, ax = plt.subplots(figsize=(5,3))
            ax.bar(
                top_students.index.astype(str),
                top_students.values,
                color=["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6"]
            )
            ax.set_title("Top 10 Students", fontsize=10)
            ax.set_xlabel("Student ID", fontsize=8)
            ax.set_ylabel("Score", fontsize=8)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.pyplot(fig)
        else:
            st.info("No student data available")
        # -------------------------------
        # Learning Style Distribution
        # -------------------------------
        st.subheader("Learning Style Distribution")

        if "learning_style" in students_df.columns:

            style_counts = students_df["learning_style"].value_counts()

            # Default: no explode
            explode = [0] * len(style_counts)

            # Highlight user's learning style
            if student_id:

                user_style = students_df[
                    students_df["student_id"] == student_id
                ]["learning_style"].iloc[-1]

                # find index of user style
                if user_style in style_counts.index:
                    idx = list(style_counts.index).index(user_style)
                    explode[idx] = 0.15   # 👈 pop-out effect

            fig3, ax3 = plt.subplots()

            ax3.pie(
                style_counts,
                labels=style_counts.index,
                autopct='%1.1f%%',
                explode=explode,        # 🔥 HERE IS THE MAGIC
                shadow=True,            # optional: makes it more 3D-like
                startangle=90
            )

            st.pyplot(fig3)

            # Optional label under chart
            if student_id:
                st.markdown(
                    f"""
                    <div style="
                        margin-top:10px;
                        padding:10px;
                        border-radius:10px;
                        background:rgba(59,130,246,0.1);
                        border-left:4px solid #3b82f6;
                    ">
                    You are here: <b>{user_style}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            st.warning("No learning_style data available")
        
        # -------------------------------
        # SUBJECT PERFORMANCE
        # -------------------------------
        st.markdown("<h2 style='text-align:center;'> Subject-wise Performance</h2>", unsafe_allow_html=True)

        if not students_df.empty:
            students_df["score"] = pd.to_numeric(students_df["score"], errors="coerce")
            subject_avg = students_df.groupby(["student_id", "subject"])["score"].mean().unstack()

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.bar_chart(subject_avg.fillna(0))
            safe_df = subject_avg.fillna(0)
            # Ensure all values are numeric
            safe_df = safe_df.apply(pd.to_numeric, errors="coerce")
            st.dataframe(
                safe_df.style.background_gradient(cmap="Purples")
            )
        else:
            st.info("No subject data available")
        
        st.markdown("<h2 style='text-align:center;'>Overall score by students</h2>", unsafe_allow_html=True)
        avg_scores = students_df.groupby("student_id")["score"].mean()
        st.bar_chart(avg_scores)
        st.markdown("<h2 style='text-align:center;'>Your Past Records</h2>", unsafe_allow_html=True)
        if not student_history.empty:
            st.dataframe(
                student_history.style
                .background_gradient(cmap="Oranges")
            )
        else:
            st.warning("No records yet")
        
    # -----------------------------------
    # AI LEARNING PATH GENERATOR
    # -----------------------------------
    def generate_ai_learning_plan(student_df):
        latest = student_df.iloc[-1]
        learning_style = latest["learning_style"]
        learner_group = latest["learner_group"]
        # -----------------------------------
        # Identify Weak Topics
        # -----------------------------------
        topic_scores = (
            student_df.groupby("topic")["score"]
            .mean()
            .sort_values()
        )
        weak_topics = topic_scores.index.tolist()
        lowest_score = topic_scores.iloc[0]

        # Weak topic = lowest scoring topic
        weak_topic = weak_topics[0]
        # Remove strong topics if needed
        priority_topics = [
            topic
            for topic in weak_topics
            if topic_scores[topic] < 75
        ]

        # If student is good in all topics
        if len(priority_topics) == 0:

            priority_topics = weak_topics[:3]
        # -------------------------------
        # Personalized Recommendations
        # -------------------------------
        recommendations = []
        # -----------------------------------
        # Score-Based Adaptive Learning
        # -----------------------------------
        average_score = student_df["score"].mean()
        # LOW PERFORMERS
        if average_score < 40:
            recommendations = [
                f"Start learning {weak_topic} from basic concepts",
                "Study theory before problem solving",
                "Spend 1 hour daily on concept building",
                "Revise fundamentals regularly",
                "Solve beginner-level questions first"
            ]
        # MEDIUM PERFORMERS
        elif average_score < 70:

            recommendations = [
                f"Practice medium-level problems in {weak_topic}",
                "Focus on accuracy improvement",
                "Revise important concepts daily",
                "Take topic-wise quizzes regularly"
            ]

        # HIGH PERFORMERS
        else:

            recommendations = [
                f"Practice advanced questions in {weak_topic}",
                "Attempt mock tests regularly",
                "Focus on speed and accuracy",
                "Move to next topic after 2-3 days of revision"
            ]

        # -----------------------------------
        # Learning Style Adaptation
        # -----------------------------------

        if learning_style == "Visual":

            recommendations.append(
                "Use diagrams and flowcharts"
            )

        elif learning_style == "Audio-Visual":

            recommendations.append(
                "Watch animated lectures and tutorials"
            )

        elif learning_style == "Kinesthetic":

            recommendations.append(
                "Practice interactive exercises daily"
            )

        # -------------------------------
        # AI Cluster-Based Suggestions
        # -------------------------------

        if learner_group == "Needs Support":

            recommendations.append(
                "Spend 30 minutes daily on revision"
            )
        elif learner_group == "Consistent Learners":

            recommendations.append(
                "Practice medium-level questions regularly"
            )
        elif learner_group == "High Performers":
            recommendations.append(
                "Try advanced-level problem solving"
            )
        # -------------------------------
        # Weekly Plan
        # -------------------------------
        weekly_plan = {
            "Monday": "Learn core concepts",
            "Tuesday": "Practice beginner problems",
            "Wednesday": "Revise important formulas",
            "Thursday": "Solve quizzes and exercises",
            "Friday": "Topic revision and mock test"
        }
        # -----------------------------------
        # ML-BASED NEXT TOPIC PREDICTION
        # -----------------------------------

        latest_topic = latest["topic"]

        latest_subject = latest["subject"]

        latest_score = latest["score"]

        latest_time_spent = latest["time_spent"]

        latest_attempts = latest["attempts"]

        latest_learning_style = latest["learning_style"]

        try:

            prediction_payload = {
                "score": int(latest_score),
                "time_spent": int(latest_time_spent),
                "attempts": int(latest_attempts),
                "learning_style": latest_learning_style,
                "subject": latest_subject,
                "topic": latest_topic
            }

            response = requests.post(
                f"{API_URL}/get_learning_path",
                json=prediction_payload,
                timeout=5
            )

            if response.status_code == 200:

                prediction_result = response.json()

                next_topic = prediction_result.get(
                    "next_topic",
                    weak_topic
                )
            else:

                next_topic = weak_topic
        except:
            next_topic = weak_topic
        return {
            "weak_topic": weak_topic,
            "priority_topics": priority_topics,
            "learning_style": learning_style,
            "learner_group": learner_group,
            "recommendations": recommendations,
            "weekly_plan": weekly_plan,
            "next_topic": next_topic,
            "average_score": average_score,
        }
    # -----------------------------------
    # AI RESOURCE RECOMMENDATION ENGINE
    # -----------------------------------
    def generate_ai_resources(
        weak_topic,
        learning_style,
        learner_group
    ):
        resources = []
        videos = []

        # -----------------------------------
        # Visual Learners
        # -----------------------------------

        if learning_style == "Visual":

            resources.extend([
                f"Visual notes for {weak_topic}",
                f"Flowcharts for {weak_topic}",
                f"Diagram-based explanations"
            ])

            videos.extend([
                f"{weak_topic} visual tutorial",
                f"{weak_topic} animated explanation"
            ])

        elif learning_style == "Audio-Visual":

            resources.extend([
                f"Lecture videos for {weak_topic}",
                f"Animated concept explanations",
                f"Recorded summaries"
            ])

            videos.extend([
                f"{weak_topic} lecture",
                f"{weak_topic} explained step by step"
            ])

        else:

            resources.extend([
                f"Hands-on exercises for {weak_topic}",
                f"Interactive worksheets",
                f"Practice-based activities"
            ])

            videos.extend([
                f"{weak_topic} practice questions",
                f"{weak_topic} solved exercises"
            ])

        return {
            "resources": resources,
            "videos": videos
        }

    # -----------------------------------
    # AI QUIZ RECOMMENDATION ENGINE
    # -----------------------------------

    def generate_ai_quizzes(
        weak_topics,
        average_score,
        learning_style,
        learner_group
    ):

        quizzes = []

        for topic in weak_topics[:3]:

            # -----------------------------------
            # Difficulty Based on Score
            # -----------------------------------

            if average_score < 40:

                difficulty = "Beginner"
                questions = 5
                focus = "Concept Building"

            elif average_score < 70:

                difficulty = "Intermediate"
                questions = 10
                focus = "Practice & Accuracy"

            else:

                difficulty = "Advanced"
                questions = 15
                focus = "Mock Tests & Speed"

            # -----------------------------------
            # Learning Style Adaptation
            # -----------------------------------

            if learning_style == "Visual":

                style_tip = "Use diagram-based questions"

            elif learning_style == "Audio-Visual":

                style_tip = "Watch explanation before solving"

            else:

                style_tip = "Solve interactive exercises"

            # -----------------------------------
            # AI Quiz Object
            # -----------------------------------

            quizzes.append({

                "topic": topic,

                "difficulty": difficulty,

                "questions": questions,

                "focus": focus,

                "style_tip": style_tip,

                "link":
                    f"https://www.google.com/search?q={topic}+quiz"
            })

        return quizzes

    # -------------------------------
    #  TAB 4: LEARNING PLAN
    # -------------------------------
    with tab4:
        # -----------------------------------
        # AI Personalized Learning Plan
        # -----------------------------------

        student_data = students_df[
            students_df["student_id"] == student_id
        ]

        ai_plan = generate_ai_learning_plan(student_data)
    
        # -----------------------------------
        # AI Analysis
        # -----------------------------------

        st.write(f"### Student ID: {student_id}")

        st.write("### AI Analysis")

        st.write(f"• Weak Topic: {ai_plan['weak_topic']}")
        st.write(f"• Learning Style: {ai_plan['learning_style']}")
        st.write(f"• Behavior Group: {ai_plan['learner_group']}")

        st.markdown(f"""
        <h3 style='color:#2563eb;'>
        Recommended Next Topic:
        <b>{ai_plan['next_topic']}</b>
        </h3>
        """, unsafe_allow_html=True)
        # -----------------------------------
        # Personalized Learning Path
        # -----------------------------------

        st.write("### Personalized Learning Path")

        for rec in ai_plan["recommendations"]:
            st.write(f"• {rec}")

        # -----------------------------------
        # Weekly Learning Plan
        # -----------------------------------

        st.write("### Weekly Learning Plan")

        for day, task in ai_plan["weekly_plan"].items():
            st.write(f"**{day}:** {task}")

        # -----------------------------------
        # Priority Topics
        # -----------------------------------

        st.write("### Priority Topics")

        if len(ai_plan["priority_topics"]) > 0:

            for topic in ai_plan["priority_topics"]:
                st.write(f"• {topic}")

        else:
            st.write("No weak topics detected")

        
        # -----------------------------------
        # AI SMART RESOURCES
        # -----------------------------------

        resource_data = generate_ai_resources(
            ai_plan["weak_topic"],
            ai_plan["learning_style"],
            ai_plan["learner_group"]
        )

        # -----------------------------------
        # Study Resources
        # -----------------------------------

        st.write("### Recommended Resources")

        for resource in resource_data["resources"]:
            st.write(f"• {resource}")

        # -----------------------------------
        # Recommended Videos
        # -----------------------------------

        st.write("### Recommended Videos")

        video_links = {
            "Arrays": "https://www.youtube.com/watch?v=QJNwK2uJyGs",
            "Linked List": "https://www.youtube.com/watch?v=58YbpRDc4yw",
            "Stack": "https://www.youtube.com/watch?v=wjI1WNcIntg",
            "Queue": "https://www.youtube.com/watch?v=XuCbpw6Bj1U",
            "Tree": "https://www.youtube.com/watch?v=oSWTXtMglKE",
            "Graph": "https://www.youtube.com/watch?v=pcKY4hjDrxk"
        }
        weak_topic = ai_plan["weak_topic"]

        if weak_topic in video_links:

           st.video(video_links[weak_topic])

        else:

            youtube_link = (
                "https://www.youtube.com/results?search_query="
                + weak_topic.replace(" ", "+")
           )

            st.markdown(
                f"[▶ Watch videos on {weak_topic}]({youtube_link})"
            )

    # -------------------------------
    # TAB 5: Quiz Arena
    # -------------------------------
    with tab5:

        st.title("Quiz Arena")

        st.write(
            "AI-generated adaptive quizzes"
        )
        # -----------------------------------
        # QUIZ SETTINGS
        # -----------------------------------

        quiz_topic = st.selectbox(
            "Select Topic",
            SUBJECT_TOPICS[subject],
            key="quiz_topic"
        )

        # Difficulty logic
        if score < 40:
            difficulty = "Easy"

        elif score < 70:
            difficulty = "Medium"

        else:
            difficulty = "Hard"

        st.info(
            f"AI Difficulty Level: {difficulty}"
        )

        # -----------------------------------
        # GENERATE QUIZ
        # -----------------------------------

        if st.button("Generate AI Quiz"):

            with st.spinner(
                "Generating personalized quiz..."
            ):

                st.session_state.quiz_questions = generate_ai_quiz(
                    quiz_topic,
                    difficulty,
                    learning_style
                )

                st.session_state.quiz_generated = True

        # -----------------------------------
        # DISPLAY QUIZ
        # -----------------------------------

        if (
            st.session_state.quiz_generated
            and st.session_state.quiz_questions
        ):

            st.success("Quiz Ready")

            user_answers = []

            for i, q in enumerate(
                st.session_state.quiz_questions
            ):

                st.markdown(
                    f"### Q{i+1}. {q['question']}"
                )

                selected = st.radio(
                    "Choose answer",
                    q["options"],
                    key=f"quiz_{i}"
                )

                user_answers.append(selected)

            # -----------------------------------
            # SUBMIT QUIZ
            # -----------------------------------

            if st.button("Submit Quiz"):

                correct = 0

                for i, q in enumerate(
                    st.session_state.quiz_questions
                ):

                    if (
                        user_answers[i]
                        == q["answer"]
                    ):

                        correct += 1

                final_score = (
                    correct
                    / len(st.session_state.quiz_questions)
                ) * 100

                st.metric(
                    "Quiz Score",
                    f"{final_score}%"
                )

                if final_score >= 70:

                    st.success(
                        "Excellent performance!"
                    )

                else:

                    st.warning(
                        "Practice more questions"
                    )

    # -------------------------------
    # TAB 6: ACHIEVEMENTS & BADGES
    # -------------------------------

    with tab6:

        st.subheader("Achievements & Badges")

        student_badge_data = students_df[
            students_df["student_id"] == student_id
        ]

        # -----------------------------------
        # Best Topic Detection
        # -----------------------------------

        if not student_badge_data.empty:
            best_topic_row = student_badge_data.loc[
                student_badge_data["score"].idxmax()
            ]
        else:
            st.warning("No data for badges")
            st.stop()

        best_topic = best_topic_row["topic"]
        best_score = best_topic_row["score"]

        # -----------------------------------
        # Achievement Section
        # -----------------------------------

        st.write("## Achievement Status")

        if best_score >= 95:

            st.balloons()

            st.success(
                f"🏅 Gold Top Performer in {best_topic} "
            )

        elif best_score >= 75:

            st.success(
                f"🥈 Silver Performer in {best_topic}"
            )

        elif best_score >= 50:

            st.info(
                f"Good Progress in {best_topic}"
            )

        else:

            st.warning(
                f"Keep Improving in {best_topic}"
            )

        # -----------------------------------
        # Badge System
        # -----------------------------------

        st.write("## Earned Badges")

        badges = []

        # Gold Badge
        if best_score >= 95:
            badges.append("🥇 Gold Excellence Badge")

        # Silver Badge
        if best_score >= 75:
            badges.append("🥈 Silver Achievement Badge")

        # Consistency Badge
        average_score = student_badge_data["score"].mean()

        if average_score >= 70:
            badges.append("Consistency Master Badge")

        # Hardworking Badge
        avg_attempts = student_badge_data["attempts"].mean()

        if avg_attempts >= 3:
            badges.append("Hardworking Learner Badge")

        # Dedicated Learner Badge
        avg_time = student_badge_data["time_spent"].mean()

        if avg_time >= 60:
            badges.append("Dedicated Learner Badge")

        # -----------------------------------
        # Display Badges
        # -----------------------------------

        if badges:

            for badge in badges:
                st.write(f"• {badge}")

        else:

            st.info("No badges earned yet")

        # -----------------------------------
        # Performance Summary
        # -----------------------------------

        st.write("## Performance Summary")

        st.write(f"• Best Topic: {best_topic}")
        st.write(f"• Highest Score: {best_score}")
        st.write(f"• Average Score: {round(average_score, 2)}")

        # -----------------------------------
        # Motivational Insights
        # -----------------------------------

        st.write("##  Motivation")

        if average_score >= 85:

            st.success(
                "Excellent consistency! Start solving advanced mock tests."
            )

        elif average_score >= 60:

            st.info(
                "You are improving steadily. Focus on weak topics regularly."
            )

        else:

            st.warning(
                "Build strong fundamentals before moving to advanced topics."
            )