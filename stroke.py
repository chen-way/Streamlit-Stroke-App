import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
import time
import random

# Load dataset
uploaded = 'stroke_data.csv'
df = pd.read_csv(uploaded)

# Prepare features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Model pipeline
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X, y)

# --- CLEAN, CONSISTENT STYLING ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9e8 !important;
        color: black !important;
    }
    body {
        color: black !important;
        background-color: #f7f9e8 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        border: 1px solid #ccc;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e7f2c9 !important;
        color: black !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: black !important;
    }

    /* Inputs & dropdowns */
    div[data-baseweb="select"],
    div[data-baseweb="input"],
    div[data-baseweb="textarea"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
        border-radius: 6px !important;
    }

    div[data-baseweb="select"] * {
        color: black !important;
        background-color: white !important;
    }

    /* Progress bar */
    .stProgress>div>div {
        background: linear-gradient(to right, #bde6a1, #8ab94f) !important;
    }

    /* Risk text */
    .risk-text {
        font-weight: bold;
        font-size: 18px;
        padding: 5px;
        color: black !important;
    }
    .risk-high { color: red !important; }
    .risk-low { color: green !important; }

    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'exercise_tip' not in st.session_state:
    st.session_state.exercise_tip = None
if 'diet_tip' not in st.session_state:
    st.session_state.diet_tip = None

# Sidebar
st.sidebar.header("📋 Enter Patient Details")

def get_user_input():
    sex = st.sidebar.selectbox('Sex (1=Male, 0=Female)', [1, 0])
    age = st.sidebar.number_input('Age', min_value=0, max_value=150, step=1, value=30)
    hypertension = st.sidebar.selectbox('Hypertension (1=Yes, 0=No)', [1, 0])
    heart_disease = st.sidebar.selectbox('Heart Disease (1=Yes, 0=No)', [1, 0])
    ever_married = st.sidebar.selectbox('Ever Married (1=Yes, 0=No)', [1, 0])
    work_type = st.sidebar.selectbox('Work Type (0=Never Worked, 1=Children, 2=Govt Job, 3=Self-Employed, 4=Private)', [0, 1, 2, 3, 4])
    residence_type = st.sidebar.selectbox('Residence Type (1=Urban, 0=Rural)', [1, 0])
    avg_glucose_level = st.sidebar.number_input('Avg Glucose Level (mg/dL)', min_value=50, max_value=300, step=1, value=100)
    bmi = st.sidebar.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1, value=20.0)
    smoking_status = st.sidebar.selectbox('Smoking Status (1=Smokes, 0=Never Smoked)', [1, 0])

    user_data = {
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'sex': int(sex),
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'ever_married': int(ever_married),
        'work_type': int(work_type),
        'Residence_type': int(residence_type),
        'smoking_status': int(smoking_status)
    }

    user_data_html = f"""
    <div style="background-color:white; padding: 20px; border-radius: 10px; color: black; border: 1px solid #ccc;">
        <h4>User Input Data:</h4>
        <ul>
            {"".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in user_data.items()])}
        </ul>
    </div>
    """
    st.markdown(user_data_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    return user_data

def make_prediction(user_input):
    input_data = pd.DataFrame([user_input])
    prediction_proba = model_pipeline.predict_proba(input_data)[0][1]

    st.progress(min(int(prediction_proba * 100), 100))

    if prediction_proba > 0.3:
        risk_level = "High"
        risk_class = "risk-high"
        advice_message = "**⚠️ Consider making healthy lifestyle changes!<br>Check out the tips below!**"
    else:
        risk_level = "Low"
        risk_class = "risk-low"
        advice_message = ""

    st.markdown(f'<p class="risk-text {risk_class}">Stroke Risk Level: {risk_level}</p>', unsafe_allow_html=True)
    st.write(f'Estimated Probability of Stroke: {prediction_proba:.2%}')
    if advice_message:
        st.markdown(f'<p class="{risk_class}">{advice_message}</p>', unsafe_allow_html=True)

# Tips
diet_tips = [
    "🥦 Eat a balanced diet with plenty of fruits and vegetables.",
    "💧 Drink at least 8 glasses of water daily.",
    "🌾 Reduce processed foods and opt for whole grains.",
    "🐟 Include lean proteins like fish, chicken, and tofu.",
    "⚖️ Limit sugar and saturated fat intake."
]
exercise_tips = [
    "🏃 Aim for at least 30 minutes of moderate exercise daily.",
    "🏋️‍♂️ Include strength training twice a week.",
    "🚶 Take the stairs instead of the elevator.",
    "🧘 Try yoga or stretching for flexibility.",
    "🖥️ Move every hour if you have a desk job."
]

# Main app
def main():
    st.title("Stroke Prediction Model")
    st.markdown("<br>", unsafe_allow_html=True)

    user_input = get_user_input()

    if st.button('🔍 Predict Stroke Risk'):
        with st.spinner('Analyzing data...'):
            time.sleep(1)
            make_prediction(user_input)

    st.header("💡 Health Tips")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🍏 Get a Diet Tip"):
            st.session_state.diet_tip = random.choice(diet_tips)
        if st.session_state.diet_tip:
            st.success(st.session_state.diet_tip)

    with col2:
        if st.button("🏋️ Get an Exercise Tip"):
            st.session_state.exercise_tip = random.choice(exercise_tips)
        if st.session_state.exercise_tip:
            st.success(st.session_state.exercise_tip)

    st.markdown("## 📢 Check this out!")
    st.markdown("##### https://www.health.harvard.edu/womens-health/8-things-you-can-do-to-prevent-a-stroke")

st.sidebar.markdown("---")
st.sidebar.write("⚠️ **Disclaimer:** Educational use only — not medical advice.")

if __name__ == '__main__':
    main()

st.markdown("---")
st.markdown("## 💭 User Reviews")
st.write("⭐ 'This app is very helpful and easy to use!' - Asiyah A.")
st.write("⭐ 'Great insights! Helped me understand my stroke risk better.' - Xiaomeng W.")
st.write("⭐ 'A very well-made and informative application.' - Crystal W.")
