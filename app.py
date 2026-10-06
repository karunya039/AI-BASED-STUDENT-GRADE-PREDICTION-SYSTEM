"""

AI-Based Student Grade Prediction System

----------------------------------------

Run with:  streamlit run app.py



This single file contains the whole web app:

  1. Page setup + custom CSS

  2. Helper functions (load model, predict, AI suggestions)

  3. The Streamlit user interface

"""



import os

from pathlib import Path



import joblib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import requests

import seaborn as sns

import streamlit as st

from dotenv import load_dotenv



# ---------------------------------------------------------------------------

# 1. BASIC SETTINGS

# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "student_grade_model.pkl"

PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

DATASET_PATH = BASE_DIR / "data" / "student_data.csv"



# Load values from the .env file (GEMINI_API_KEY, etc.)

load_dotenv(BASE_DIR / ".env")



GRADES = ["A+", "A", "B+", "B", "C", "D", "F"]



PERFORMANCE_LEVELS = {

    "A+": "Outstanding",

    "A": "Excellent",

    "B+": "Very Good",

    "B": "Good",

    "C": "Average",

    "D": "Needs Improvement",

    "F": "Needs Significant Improvement",

}



GRADE_COLORS = {

    "A+": "#16a34a",

    "A": "#22c55e",

    "B+": "#0ea5e9",

    "B": "#3b82f6",

    "C": "#f59e0b",

    "D": "#f97316",

    "F": "#ef4444",

}



DEFAULT_FEATURES = [

    "study_hours",

    "attendance",

    "assignment_marks",

    "internal_marks",

    "previous_exam_marks",

    "participation",

    "sleep_hours",

    "previous_grade",

]



st.set_page_config(

    page_title="AI-Based Student Grade Prediction System",

    page_icon="🎓",

    layout="wide",

)



# ---------------------------------------------------------------------------

# 2. CUSTOM CSS (modern education / AI theme)

# ---------------------------------------------------------------------------

CUSTOM_CSS = """

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght\@300;400;500;600;700;800&display=swap');



html, body, [class*="css"], .stApp {

    font-family: 'Poppins', sans-serif;

}

.stApp {

    background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 45%, #f1f5f9 100%);

}

#MainMenu, footer {visibility: hidden;}

.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}



/* ---------- Header ---------- */

.hero {

    background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 50%, #7c3aed 100%);

    padding: 2.4rem 2rem;

    border-radius: 24px;

    color: #ffffff;

    text-align: center;

    box-shadow: 0 18px 40px rgba(79, 70, 229, 0.35);

    margin-bottom: 1.6rem;

}

.hero h1 {

    color: #ffffff !important;

    font-size: 2.3rem;

    font-weight: 800;

    margin: 0 0 0.5rem 0;

    padding: 0;

    line-height: 1.25;

}

.hero p {

    font-size: 1.05rem;

    font-weight: 300;

    margin: 0;

    opacity: 0.95;

}



/* ---------- Dashboard cards ---------- */

.metric-card {

    background: #ffffff;

    border-radius: 20px;

    padding: 1.3rem 1.2rem;

    box-shadow: 0 8px 24px rgba(30, 41, 59, 0.08);

    border-top: 5px solid #4f46e5;

    text-align: center;

    transition: transform 0.25s ease, box-shadow 0.25s ease;

    height: 100%;

}

.metric-card:hover {

    transform: translateY(-6px);

    box-shadow: 0 16px 32px rgba(79, 70, 229, 0.22);

}

.metric-title {color: #64748b; font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;}

.metric-value {color: #1e293b; font-size: 2rem; font-weight: 700; line-height: 1.2;}

.metric-sub {color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;}



/* ---------- Section headings ---------- */

.section-title {

    font-size: 1.15rem;

    font-weight: 600;

    color: #1e3a8a;

    margin-bottom: 0.6rem;

}



/* ---------- Input containers (rounded cards) ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {

    background: #ffffff;

    border-radius: 20px !important;

    border: 1px solid #e2e8f0 !important;

    box-shadow: 0 6px 20px rgba(30, 41, 59, 0.06);

    padding: 0.4rem 0.6rem;

    transition: box-shadow 0.25s ease;

}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {

    box-shadow: 0 12px 28px rgba(79, 70, 229, 0.14);

}



/* ---------- Input fields ---------- */

.stTextInput input, .stNumberInput input {

    border-radius: 12px !important;

    border: 1.5px solid #cbd5e1 !important;

    background: #f8fafc !important;

    padding: 0.55rem 0.8rem !important;

    transition: all 0.2s ease;

}

.stTextInput input:focus, .stNumberInput input:focus {

    border-color: #4f46e5 !important;

    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.18) !important;

    background: #ffffff !important;

}

div[data-baseweb="select"] > div {

    border-radius: 12px !important;

    border: 1.5px solid #cbd5e1 !important;

    background: #f8fafc !important;

}

label, .stSlider label, .stSelectbox label {

    font-weight: 500 !important;

    color: #334155 !important;

}



/* ---------- Buttons ---------- */

.stButton > button {

    width: 100%;

    background: linear-gradient(135deg, #4f46e5, #7c3aed);

    color: #ffffff;

    font-size: 1.25rem;

    font-weight: 600;

    padding: 0.9rem 1.5rem;

    border: none;

    border-radius: 16px;

    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.38);

    transition: all 0.25s ease;

}

.stButton > button:hover {

    transform: translateY(-3px) scale(1.01);

    box-shadow: 0 16px 32px rgba(79, 70, 229, 0.5);

    color: #ffffff;

    border: none;

}

.stButton > button:active {transform: translateY(0);}



/* ---------- Prediction result card ---------- */

.result-card {

    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);

    border-radius: 24px;

    padding: 2rem 1.5rem;

    text-align: center;

    color: #ffffff;

    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);

    margin: 1.2rem 0;

    animation: fadeUp 0.6s ease;

}

.result-label {font-size: 1.1rem; letter-spacing: 3px; font-weight: 500; opacity: 0.85;}

.result-grade {font-size: 5.5rem; font-weight: 800; line-height: 1.1; margin: 0.4rem 0;}

.result-level {

    display: inline-block;

    background: rgba(255, 255, 255, 0.15);

    padding: 0.4rem 1.2rem;

    border-radius: 50px;

    font-weight: 500;

    margin-top: 0.3rem;

}

@keyframes fadeUp {

    from {opacity: 0; transform: translateY(18px);}

    to {opacity: 1; transform: translateY(0);}

}



/* ---------- Info boxes ---------- */

.info-box {

    background: #ffffff;

    border-left: 5px solid #4f46e5;

    border-radius: 14px;

    padding: 1rem 1.2rem;

    box-shadow: 0 6px 18px rgba(30, 41, 59, 0.06);

    color: #334155;

    margin-bottom: 0.8rem;

}

.provider-badge {

    display: inline-block;

    background: #eef2ff;

    color: #4338ca;

    font-weight: 600;

    padding: 0.35rem 1rem;

    border-radius: 50px;

    margin-bottom: 0.8rem;

}



/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"] {

    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);

}

section[data-testid="stSidebar"] * {color: #e0e7ff !important;}



/* ---------- Responsive ---------- */

@media (max-width: 768px) {

    .hero h1 {font-size: 1.5rem;}

    .hero {padding: 1.5rem 1rem;}

    .result-grade {font-size: 4rem;}

    .metric-value {font-size: 1.6rem;}

}

</style>

"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)





# ---------------------------------------------------------------------------

# 3. HELPER FUNCTIONS

# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)

def load_artifacts():

    """Load the saved model and preprocessing objects (cached for speed)."""

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor





def get_accuracy(preprocessor):

    """Read the real test accuracy saved by train_model.py (if available)."""

    if isinstance(preprocessor, dict) and "accuracy" in preprocessor:

        try:

            return float(preprocessor["accuracy"])

        except (TypeError, ValueError):

            return None

    return None





def predict_grade(model, preprocessor, values):

    """

    Turn the student's inputs into model features and predict the grade.

    Returns (grade, confidence_percent).



    preprocessor.pkl is expected to be a dictionary with:

      feature_columns        -> list of feature names in training order

      numeric_columns        -> list of numeric feature names

      scaler                 -> fitted StandardScaler for numeric columns

      previous_grade_encoder -> fitted encoder for the previous_grade column

      target_encoder         -> fitted LabelEncoder for the final grade

      accuracy               -> test accuracy (0 to 1)

    """

    if not isinstance(preprocessor, dict):

        raise ValueError("preprocessor.pkl has an unexpected format. Please run: python train_model.py")



    feature_columns = preprocessor.get("feature_columns", DEFAULT_FEATURES)

    numeric_columns = preprocessor.get(

        "numeric_columns", [c for c in feature_columns if c != "previous_grade"]

    )



    # Put the inputs in a one-row table

    data = pd.DataFrame([values])



    # Encode the categorical column (Previous Grade)

    grade_encoder = preprocessor["previous_grade_encoder"]

    data["previous_grade"] = grade_encoder.transform(data["previous_grade"].astype(str))



    # Scale the numeric columns

    scaler = preprocessor.get("scaler")

    if scaler is not None:

        data[numeric_columns] = scaler.transform(data[numeric_columns])



    # Keep columns in the same order as training

    data = data[feature_columns]



    # Predict

    prediction = np.asarray(model.predict(data)).ravel()[0]

    target_encoder = preprocessor.get("target_encoder")

    if target_encoder is not None:

        grade = str(target_encoder.inverse_transform([int(prediction)])[0])

    else:

        grade = str(prediction)



    # Confidence (probability of the predicted class)

    confidence = None

    try:

        confidence = float(np.max(model.predict_proba(data))) * 100

    except Exception:

        pass



    return grade, confidence





def build_explanation(grade, v):

    """Write a simple explanation using the student's own inputs."""

    strengths, weaknesses = [], []



    if v["attendance"] >= 85:

        strengths.append("high attendance")

    elif v["attendance"] < 75:

        weaknesses.append("low attendance")



    if v["study_hours"] >= 5:

        strengths.append("good daily study time")

    elif v["study_hours"] < 3:

        weaknesses.append("few study hours")



    if v["assignment_marks"] >= 75:

        strengths.append("strong assignment marks")

    elif v["assignment_marks"] < 50:

        weaknesses.append("weak assignment marks")



    if v["internal_marks"] >= 75:

        strengths.append("strong internal marks")

    elif v["internal_marks"] < 50:

        weaknesses.append("weak internal marks")



    if v["previous_exam_marks"] >= 75:

        strengths.append("a good previous exam score")

    elif v["previous_exam_marks"] < 50:

        weaknesses.append("a low previous exam score")



    if v["sleep_hours"] < 6:

        weaknesses.append("too little sleep")



    text = f"The model predicts grade **{grade}** ({PERFORMANCE_LEVELS.get(grade, '')}). "

    if strengths:

        text += "Helping factors: " + ", ".join(strengths) + ". "

    if weaknesses:

        text += "Factors holding the grade back: " + ", ".join(weaknesses) + ". "

    if not strengths and not weaknesses:

        text += "The student's inputs are close to average in most areas. "

    return text





def rule_based_suggestions(grade, v):

    """Simple suggestions that work without any internet or AI service."""

    tips = []

    if v["attendance"] < 75:

        tips.append("Raise attendance above 75% - regular classes make topics much easier to follow.")

    if v["study_hours"] < 3:

        tips.append("Increase self-study to at least 3-4 focused hours per day, split into short sessions.")

    if v["assignment_marks"] < 60:

        tips.append("Submit assignments on time and ask teachers for feedback on lost marks.")

    if v["internal_marks"] < 60:

        tips.append("Revise class notes weekly and practise internal test questions to improve internal marks.")

    if v["previous_exam_marks"] < 60:

        tips.append("Go through previous exam mistakes and practise those topics again.")

    if v["participation"] < 5:

        tips.append("Participate more in class discussions and group activities.")

    if v["sleep_hours"] < 6:

        tips.append("Sleep 7-8 hours daily - good rest improves memory and concentration.")

    elif v["sleep_hours"] > 10:

        tips.append("Try to keep sleep around 7-9 hours to leave enough time for study.")



    if grade in ("D", "F"):

        tips.append("Meet your mentor or subject teachers this week and make a simple study plan.")

    elif grade in ("A+", "A"):

        tips.append("Great work! Keep your routine and try advanced problems or help classmates to deepen learning.")



    if not tips:

        tips.append("Keep up your current routine and revise regularly to stay consistent.")

    return tips





def gemini_ready():

    """True if a real Gemini API key is stored in .env."""

    key = os.getenv("GEMINI_API_KEY", "").strip()

    return bool(key) and key != "your_api_key_here"





def ollama_host():

    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")





def ollama_ready():

    """True if the local Ollama server is running."""

    try:

        r = requests.get(f"{ollama_host()}/api/tags", timeout=2)

        return r.status_code == 200

    except Exception:

        return False





def build_prompt(name, grade, v):

    return (

        "You are a friendly academic advisor. A student's details are below.\n"

        f"Name: {name}\n"

        f"Predicted grade: {grade}\n"

        f"Study hours per day: {v['study_hours']}\n"

        f"Attendance: {v['attendance']}%\n"

        f"Assignment marks: {v['assignment_marks']}/100\n"

        f"Internal marks: {v['internal_marks']}/100\n"

        f"Previous exam marks: {v['previous_exam_marks']}/100\n"

        f"Participation score: {v['participation']}/10\n"

        f"Sleep hours: {v['sleep_hours']}\n"

        f"Previous grade: {v['previous_grade']}\n\n"

        "Give 5 short, simple, personalized study suggestions as a bullet list. "

        "Use easy English. Do not add any introduction."

    )
def ask_gemini(prompt):
    """Call Gemini and retry temporary server errors."""
    import time

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model_name}:generateContent"
    )

    headers = {
        "x-goog-api-key": os.getenv("GEMINI_API_KEY", "").strip(),
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=40
            )

            if response.status_code == 503:
                if attempt < 2:
                    time.sleep(3)
                    continue

            response.raise_for_status()

            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except requests.exceptions.RequestException:
            if attempt == 2:
                raise

            time.sleep(3)

    raise RuntimeError("Gemini service is temporarily unavailable.")

def ask_ollama(prompt):

    """Call the local Ollama server. Raises an error if something goes wrong."""

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")

    body = {"model": model_name, "prompt": prompt, "stream": False}

    response = requests.post(f"{ollama_host()}/api/generate", json=body, timeout=120)

    response.raise_for_status()

    return response.json()["response"].strip()





def get_ai_suggestions(choice, name, grade, v):

    """

    Try the chosen provider and fall back to rule-based tips on any problem.

    Returns (provider_name, suggestion_text, warning_or_None).

    """

    prompt = build_prompt(name, grade, v)

    warning = None



    use_gemini = choice == "Gemini" or (choice == "Auto" and gemini_ready())

    use_ollama = choice == "Ollama" or (choice == "Auto" and not gemini_ready() and ollama_ready())



    if use_gemini:

        if not gemini_ready():

            warning = "Gemini API key not found in .env - using rule-based suggestions."

        else:

            try:

                return "Gemini", ask_gemini(prompt), None

            except Exception as err:
                warning = f"Gemini error: {err}"

    elif use_ollama:

        if not ollama_ready():

            warning = "Ollama is not running - using rule-based suggestions."

        else:

            try:

                return "Ollama", ask_ollama(prompt), None

            except Exception as err:

                warning = f"Ollama could not answer ({type(err).__name__}). Showing rule-based suggestions."



    tips = rule_based_suggestions(grade, v)

    return "Rule-Based Suggestions", "\n".join(f"- {t}" for t in tips), warning





def validate_inputs(name, student_id):

    """Return a list of problems with the inputs (empty list = all good)."""

    problems = []

    if not name.strip():

        problems.append("Please enter the student's name.")

    elif any(ch.isdigit() for ch in name):

        problems.append("Student name should not contain numbers.")

    if not student_id.strip():

        problems.append("Please enter the student ID.")

    return problems





def draw_chart(v):

    """Simple bar chart of the student's academic values (all shown out of 100)."""

    labels = ["Study Hours\n(of 10 h)", "Attendance", "Assignment\nMarks", "Internal\nMarks", "Previous Exam\nMarks"]

    percent = [

        min(v["study_hours"] / 10 * 100, 100),

        v["attendance"],

        v["assignment_marks"],

        v["internal_marks"],

        v["previous_exam_marks"],

    ]

    actual = [f"{v['study_hours']} h", f"{v['attendance']}%", f"{v['assignment_marks']}",

              f"{v['internal_marks']}", f"{v['previous_exam_marks']}"]



    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = sns.barplot(x=labels, y=percent, hue=labels, palette="viridis", legend=False, ax=ax)

    for bar, text in zip(bars.patches, actual):

        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, text,

                ha="center", fontsize=10, fontweight="bold", color="#1e293b")

    ax.set_ylim(0, 115)

    ax.set_ylabel("Score (%)")

    ax.set_title("Student Academic Performance Overview", fontweight="bold", color="#1e3a8a")

    sns.despine()

    fig.tight_layout()

    return fig





def metric_card(title, value, sub, color="#4f46e5"):

    return (

        f'<div class="metric-card" style="border-top-color:{color};">'

        f'<div class="metric-title">{title}</div>'

        f'<div class="metric-value" style="color:{color};">{value}</div>'

        f'<div class="metric-sub">{sub}</div></div>'

    )





# ---------------------------------------------------------------------------

# 4. SIDEBAR

# ---------------------------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    provider_choice = st.selectbox(

        "AI Suggestion Provider",

        ["Auto", "Gemini", "Ollama", "Rule-Based"],

        help="Auto uses Gemini if an API key exists, then Ollama if running, otherwise rule-based tips.",

    )

    st.markdown("---")

    st.markdown("### 📋 System Status")

    st.write("Model:", "✅ Found" if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() else "❌ Missing")

    st.write("Dataset:", "✅ Found" if DATASET_PATH.exists() else "❌ Missing")

    st.write("Gemini key:", "✅ Set" if gemini_ready() else "⚪ Not set")

    st.write("Ollama:", "✅ Running" if ollama_ready() else "⚪ Not running")

    st.markdown("---")

    st.caption("BTech Project • XGBoost + Streamlit")



# ---------------------------------------------------------------------------

# 5. HEADER

# ---------------------------------------------------------------------------

st.markdown(

    """

    <div class="hero">

        <h1>🎓 AI-Based Student Grade Prediction System</h1>

        <p>Predict student academic performance using Artificial Intelligence and Machine Learning.</p>

    </div>

    """,

    unsafe_allow_html=True,

)



# ---------------------------------------------------------------------------

# 6. LOAD MODEL (friendly messages if something is missing)

# ---------------------------------------------------------------------------

model, preprocessor, load_error = None, None, None



if not DATASET_PATH.exists():

    st.warning("📁 Dataset not found at \`data/student_data.csv\`. It is needed to train the model.")



if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()):

    load_error = "missing"

    st.error("🤖 The trained model was not found. Please open the terminal in VS Code and run:")

    st.code("python train_model.py", language="bash")

    st.info("After training finishes, refresh this page.")

else:

    try:

        model, preprocessor = load_artifacts()

    except Exception as err:

        load_error = str(err)

        st.error(f"Could not load the saved model files: {err}")

        st.code("python train_model.py", language="bash")



accuracy = get_accuracy(preprocessor) if preprocessor is not None else None

accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "N/A"



# ---------------------------------------------------------------------------

# 7. DASHBOARD CARDS (filled in after prediction using placeholders)

# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

card1, card2, card3 = col1.empty(), col2.empty(), col3.empty()



card1.markdown(metric_card("📊 Model Accuracy", accuracy_text, "XGBoost test accuracy"), unsafe_allow_html=True)

card2.markdown(metric_card("🤖 AI Prediction", "—", "Click predict to see the grade", "#7c3aed"), unsafe_allow_html=True)

card3.markdown(metric_card("🎓 Student Performance", "—", "Waiting for prediction", "#0ea5e9"), unsafe_allow_html=True)



st.write("")



# ---------------------------------------------------------------------------

# 8. INPUT SECTIONS

# ---------------------------------------------------------------------------

left, right = st.columns(2, gap="large")



with left:

    with st.container(border=True):

        st.markdown('<div class="section-title">👨‍🎓 Student Information</div>', unsafe_allow_html=True)

        student_name = st.text_input("Student Name", placeholder="e.g. Ravi Kumar")

        student_id = st.text_input("Student ID", placeholder="e.g. STU1001")

        study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=16.0, value=4.0, step=0.5)

        attendance = st.slider("Attendance (%)", 0, 100, 80)



    with st.container(border=True):

        st.markdown('<div class="section-title">🧠 Study & Lifestyle</div>', unsafe_allow_html=True)

        sleep_hours = st.slider("Sleep Hours (per day)", 3.0, 12.0, 7.0, step=0.5)



with right:

    with st.container(border=True):

        st.markdown('<div class="section-title">📚 Academic Performance</div>', unsafe_allow_html=True)

        assignment_marks = st.number_input("Assignment Marks (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

        internal_marks = st.number_input("Internal Marks (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

        previous_exam_marks = st.number_input("Previous Exam Marks (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

        participation = st.slider("Participation Score (0-10)", 0, 10, 6)

        previous_grade = st.selectbox("Previous Grade", GRADES, index=3)



st.write("")

predict_clicked = st.button("🔮 Predict Student Grade")



# ---------------------------------------------------------------------------

# 9. PREDICTION

# ---------------------------------------------------------------------------

if predict_clicked:

    problems = validate_inputs(student_name, student_id)



    if problems:

        for p in problems:

            st.warning(f"⚠️ {p}")

    elif model is None or preprocessor is None:

        st.error("The model is not available yet. Please run \`python train_model.py\` first.")

    else:

        values = {

            "study_hours": float(study_hours),

            "attendance": float(attendance),

            "assignment_marks": float(assignment_marks),

            "internal_marks": float(internal_marks),

            "previous_exam_marks": float(previous_exam_marks),

            "participation": float(participation),

            "sleep_hours": float(sleep_hours),

            "previous_grade": previous_grade,

        }



        try:

            with st.spinner("Analysing student data..."):

                grade, confidence = predict_grade(model, preprocessor, values)

        except Exception as err:

            st.error(f"Prediction failed: {err}")

            st.info("Tip: re-train the model with \`python train_model.py\` and try again.")

            st.stop()



        level = PERFORMANCE_LEVELS.get(grade, "Unknown")

        color = GRADE_COLORS.get(grade, "#4f46e5")



        # Update the dashboard cards at the top

        card2.markdown(metric_card("🤖 AI Prediction", grade, "Predicted final grade", color), unsafe_allow_html=True)

        card3.markdown(metric_card("🎓 Student Performance", level, f"{student_name.strip()} ({student_id.strip()})", color),

                       unsafe_allow_html=True)



        # Big result card

        confidence_html = f"<div style='margin-top:0.8rem;opacity:0.85;'>Model confidence: {confidence:.1f}%</div>" if confidence else ""

        st.markdown(

            f"""

            <div class="result-card">

                <div class="result-label">🎯 PREDICTED GRADE</div>

                <div class="result-grade" style="color:{color};">{grade}</div>

                <div class="result-level">Performance Level: {level}</div>

                {confidence_html}

            </div>

            """,

            unsafe_allow_html=True,

        )



        # Explanation

        st.markdown('<div class="section-title">💡 Explanation</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="info-box">{build_explanation(grade, values)}</div>', unsafe_allow_html=True)



        # AI suggestions

        st.markdown('<div class="section-title">🤖 AI Academic Suggestions</div>', unsafe_allow_html=True)

        with st.spinner("Generating suggestions..."):

            chosen = "Rule-Based" if provider_choice == "Rule-Based" else provider_choice

            provider_used, suggestions, warning = get_ai_suggestions(chosen, student_name.strip(), grade, values)



        if warning:

            st.warning(warning)

        st.markdown(f'<div class="provider-badge">AI Provider: {provider_used}</div>', unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(suggestions)



        # Chart

        st.markdown('<div class="section-title">📈 Academic Performance Chart</div>', unsafe_allow_html=True)

        try:

            fig = draw_chart(values)

            st.pyplot(fig)

            plt.close(fig)

        except Exception as err:

            st.warning(f"Could not draw the chart: {err}")
