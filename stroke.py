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
from openai import OpenAI
import os

st.write("OPENAI_API_KEY is set:", "OPENAI_API_KEY" in os.environ)

# Securely access the API key from secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

ALLOWED_TOPICS = ["stroke", "bmi", "hypertension", "heart disease", "smoking", "diet", "exercise", "glucose", "risk factors", "cholesterol", "blood pressure"]
def is_medical_question(prompt):
    prompt = prompt.lower()
    return any(topic in prompt for topic in ALLOWED_TOPICS)

def gpt_medical_response(prompt):
    if not is_medical_question(prompt):
        return "❌ I can only answer health-related questions, especially about stroke, hypertension, diet, and exercise."

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and medically-informed assistant. Only answer medical questions related to stroke prevention, diet, exercise, hypertension, BMI, and heart health. Do not answer unrelated topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Load dataset
uploaded = 'stroke_data.csv'  # Replace with actual data path
df = pd.read_csv(uploaded)

# Prepare features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Define the preprocessing steps
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

# Combine preprocessing steps
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

required_columns = [
    'sex', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e5f3fd !important; /* Light Blue Background */
    }
    body {
        background-color: #f4f4f4;
    }
    .stButton>button {
        background-color: #d1e5f4 !important; /* Light Blue Background */
        color: black !important;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #93BCDC !important; /* Lighter blue on hover */
        color: black !important;
    }
    .stSelectbox select {
        background-color: #FDF6E7 !important;  /* Cream background */
        color: black !important;  /* Text color */
    }
    .stSidebar, .stDataFrame, .css-1r6slb0, .css-1v3fvcr {
        background-color: #FDF6E7 !important; /* Lighter blue for contrast */
    }
    header {visibility: hidden;}
    .stProgress>div>div {
        background: linear-gradient(to right, #B3E5FC, #1E5A96) !important;
    }
    .risk-text {
        font-weight: bold;
        font-size: 18px;
        padding: 5px;
    }
    .risk-high { color: red !important; }
    .risk-low { color: green !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state variables
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
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

    # Display data in a light blue box
    user_data_html = f"""
    <div style="background-color:#d1e5f4; padding: 20px; border-radius: 10px; color: black;">
        <h4>User Input Data:</h4>
        <ul>
            {"".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in user_data.items()])}
        </ul>
    </div>
    """

    st.markdown(user_data_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)  # Blank lines for space
    
    return user_data

def make_prediction(user_input):
    input_data = pd.DataFrame([user_input])
    prediction_proba = model_pipeline.predict_proba(input_data)[0][1]
    
    st.progress(min(int(prediction_proba * 100), 100))
    
    if prediction_proba > 0.3:
        risk_level = "High"
        risk_class = "risk-high"
        advice_message = "**⚠️ Consider making healthy lifestyle changes! <br> Please take a look at the health tips below! **"
    else:
        risk_level = "Low"
        risk_class = "risk-low"
        advice_message = ""

    st.markdown(f'<p class="risk-text {risk_class}">Stroke Risk Level: {risk_level}</p>', unsafe_allow_html=True)
    st.write(f'Estimated Probability of Stroke: {prediction_proba:.2%}')

    if risk_level == "High":
        st.markdown(f'<p class="risk-high">{advice_message}</p>', unsafe_allow_html=True)

# Diet & Exercise Tips Section
diet_tips = [
    "🥦 Eat a balanced diet with plenty of fruits and vegetables.",
    "💧 Drink at least 8 glasses of water daily.",
    "🌾 Reduce processed foods and opt for whole grains.",
    "🐟 Include lean proteins like fish, chicken, and tofu.",
    "⚖️ Limit sugar and saturated fat intake."
]

exercise_tips = [
    "🏃 Aim for at least 30 minutes of moderate exercise daily.",
    "🏋️‍♂️ Incorporate strength training twice a week.",
    "🚶 Take the stairs instead of the elevator.",
    "🧘 Try yoga or stretching for flexibility.",
    "🖥️ Get up and move every hour if you have a desk job."
]

# Main app layout
def main():
    
    st.title("Stroke Prediction Model")

    # Adding space between the title and user input using <br>
    st.markdown("<br>"*1, unsafe_allow_html=True)  # Adjust the number for more or less space

    user_input = get_user_input()

    # Prediction section
    if st.button('🔍 Predict Stroke Risk'):
        with st.spinner('Analyzing data...'):
            time.sleep(1)
            st.session_state.prediction_result = make_prediction(user_input)

    # Display prediction if it exists
    if st.session_state.prediction_result is not None:
        st.session_state.prediction_result

    st.header("💡 Health Tips")

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🍏 Get a Diet Tip"):
            st.session_state.diet_tip = random.choice(diet_tips)

        # Always display diet tip if it exists
        if st.session_state.diet_tip:
            st.success(st.session_state.diet_tip)
    
    with col2:
        if st.button("🏋️ Get an Exercise Tip"):
            st.session_state.exercise_tip = random.choice(exercise_tips)

        # Always display exercise tip if it exists
        if st.session_state.exercise_tip:
            st.success(st.session_state.exercise_tip)

    # Check this out! section moved below health tips
    st.markdown("## 📢 Check this out!")
    st.markdown("##### https://www.health.harvard.edu/womens-health/8-things-you-can-do-to-prevent-a-stroke")

st.markdown("---")
st.markdown("## 🤖 Medical Assistant Chatbot")

if prompt := st.chat_input("Ask a medical question related to stroke, diet, or exercise!"):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = gpt_medical_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("⚠️ **Disclaimer:** This tool is for educational purposes and should not replace professional medical advice.")

if __name__ == '__main__':
    main()

# Separator line
st.markdown("---")

# User Reviews Section
st.markdown("## 💭 User Reviews")
st.write("⭐ 'This app is very helpful and easy to use!' - Asiyah A.")
st.write("⭐ 'Great insights! Helped me understand my stroke risk better.' - Xiaomeng W.")
st.write("⭐ 'A very well-made and informative application.' - Crystal W.")
