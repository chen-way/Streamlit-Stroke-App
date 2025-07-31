import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import time
import random

warnings.filterwarnings('ignore')

# Enhanced mobile-friendly page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Alzheimer's Risk Assessment",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

# MOBILE-OPTIMIZED CSS SECTION - Fixed for iOS compatibility
st.markdown(
    """
    <style>
    /* Mobile-first responsive design */
    .stApp {
        background-color: #e5f3fd !important;
    }
    
    .main {
        background-color: #e5f3fd !important;
        padding: 0.5rem !important;
    }
    
    .main-header {
        background-color: #d1e5f4 !important;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    .main-header h1 {
        color: #2d3436 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    .main-header p {
        color: #636e72 !important;
        font-size: 0.9rem !important;
        font-weight: 300 !important;
        margin-bottom: 0 !important;
    }
    
    .feature-container {
        background-color: #FDF6E7 !important;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    .stSelectbox > div > div {
        background-color: #FDF6E7 !important;
        color: black !important;
        border-radius: 8px !important;
        border: 2px solid #93BCDC !important;
        transition: all 0.3s ease !important;
        min-height: 44px !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #d1e5f4 !important;
        box-shadow: 0 0 0 0.2rem rgba(209, 229, 244, 0.25) !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #FDF6E7 !important;
        border: 2px solid #93BCDC !important;
        border-radius: 8px !important;
        color: black !important;
        padding: 8px !important;
        min-height: 44px !important;
    }

    div[data-baseweb="input"] input {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
        color: black !important;
        font-size: 16px !important;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #d1e5f4 !important;
        box-shadow: 0 0 0 0.2rem rgba(209, 229, 244, 0.25) !important;
    }

    div[data-testid="stNumberInput"] > div {
        border: 2px solid #93BCDC !important;
        border-radius: 8px !important;
        background-color: #FDF6E7 !important;
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        flex: 1 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }

    div[data-testid="stNumberInput"] input[type="number"] {
        border: none !important;
        background: transparent !important;
        height: 100% !important;
        width: 100% !important;
        padding: 0 12px !important;
        color: black !important;
        outline: none !important;
        box-shadow: none !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
        font-size: 16px !important;
    }

    div[data-testid="stNumberInput"] button {
        border: none !important;
        background: rgba(147, 188, 220, 0.2) !important;
        height: 100% !important;
        width: 40px !important;
        transition: background-color 0.2s ease !important;
        flex-shrink: 0 !important;
        min-height: 44px !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: rgba(147, 188, 220, 0.4) !important;
    }
    
    .result-high-risk, .result-low-risk, .result-moderate-risk {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .result-high-risk {
        background-color: #ffcccb;
        color: #d63031;
        border-color: #ff7675;
    }
    
    .result-low-risk {
        background-color: #d4edda;
        color: #155724;
        border-color: #28a745;
    }
    
    .result-moderate-risk {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffc107;
    }
    
    .result-high-risk h2, .result-low-risk h2, .result-moderate-risk h2 {
        font-size: 1.3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .result-high-risk h3, .result-low-risk h3, .result-moderate-risk h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .tips-container {
        background-color: #FDF6E7 !important;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #93BCDC;
    }
    
    .tips-container h2 {
        color: #2d3436 !important;
        text-align: center;
        margin-bottom: 1rem !important;
        font-size: 1.2rem !important;
    }
    
    @media (max-width: 768px) {
        .stSidebar {
            display: none !important;
        }
        
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        .stColumns {
            flex-direction: column !important;
        }
        
        .stColumn {
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
        
        .stButton > button {
            width: 100% !important;
            margin-bottom: 0.5rem !important;
        }
        
        .main-header h1 {
            font-size: 1.5rem !important;
        }
        
        .main-header p {
            font-size: 0.8rem !important;
        }
    }
    
    .stButton > button {
        background-color: #d1e5f4 !important;
        color: black !important;
        border-radius: 8px;
        font-size: 16px !important;
        padding: 12px 20px !important;
        transition: background-color 0.3s ease, color 0.3s ease;
        border: none;
        min-height: 44px !important;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background-color: #93BCDC !important;
        color: black !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .metric-container {
        background-color: #d1e5f4;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #93BCDC;
        color: #2d3436;
    }
    
    .stProgress>div>div {
        background: linear-gradient(to right, #B3E5FC, #1E5A96) !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    @media (max-width: 480px) {
        .main {
            padding: 0.25rem !important;
        }
        
        .main-header {
            padding: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        .tips-container {
            padding: 0.75rem;
        }
        
        .feature-container {
            padding: 0.75rem;
        }
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        line-height: 1.2 !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        line-height: 1.4 !important;
        font-size: 0.9rem !important;
    }
    
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stSelectbox, .stNumberInput, .stTextInput {
        margin-bottom: 1rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Load models - with better error handling
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model_compressed.pkl.gz')
        scaler = joblib.load('scaler_compressed.pkl.gz')
        encoders = joblib.load('encoders_compressed.pkl.gz')
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error("Model files not found. Please check file locations.")
        st.error("Looking for: model_compressed.pkl.gz, scaler_compressed.pkl.gz, encoders_compressed.pkl.gz")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("This might be a version compatibility issue or corrupted model file.")
        st.stop()

# Load the models
model, scaler, label_encoders = load_models()

# Feature definitions
CATEGORICAL_OPTIONS = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'South Korea', 'India', 'China', 'Brazil', 'South Africa', 'Australia', 'Russia', 'Mexico', 'Italy'],
    'Gender': ['Male', 'Female'],
    'Education Level': ['No Formal Education', 'Primary Education', 'Secondary Education', "Bachelor's Degree", "Master's Degree", 'Doctorate'],
    'Physical Activity Level': ['Low', 'Moderate', 'High'],
    'Smoking Status': ['Never', 'Former', 'Current'],
    'Alcohol Consumption': ['None', 'Moderate', 'High'],
    'Diabetes': ['Yes', 'No'],
    'Hypertension': ['Yes', 'No'],
    'Cholesterol Level': ['Low', 'Normal', 'High'],
    "Family History of Alzheimer’s": ['Yes', 'No'],
    'Sleep Quality': ['Poor', 'Fair', 'Good', 'Excellent'],
    'Dietary Habits': ['Unhealthy', 'Moderate', 'Healthy'],
    'Employment Status': ['Employed', 'Unemployed', 'Retired', 'Student'],
    'Marital Status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'Genetic Risk Factor (APOE-ε4 allele)': ['Yes', 'No'],
    'Social Engagement Level': ['Low', 'Moderate', 'High'],
    'Income Level': ['Low', 'Middle', 'High'],
    'Urban vs Rural Living': ['Urban', 'Rural'],
    'Air Pollution Exposure': ['Minimal', 'Slight', 'Moderate', 'High', 'Severe']
}

NUMERICAL_FEATURES = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']

feature_names = [
    'Country', 'Age', 'Gender', 'Education Level', 'BMI',
    'Physical Activity Level', 'Smoking Status', 'Alcohol Consumption',
    'Diabetes', 'Hypertension', 'Cholesterol Level',
    "Family History of Alzheimer’s", 'Cognitive Test Score', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels', 'Urban vs Rural Living'
]

def get_user_input():
    user_data = {}
    st.write("### 📝 Health Information Form")
    st.write("*Please fill in all fields for the most accurate assessment*")
    
    feature_groups = {
        "👤 Personal Information": ['Country', 'Age', 'Gender', 'Education Level', 'Employment Status', 'Marital Status', 'Income Level'],
        "🏥 Health Metrics": ['BMI', 'Diabetes', 'Hypertension', 'Cholesterol Level', 'Cognitive Test Score', 'Depression Level'],
        "🏃 Lifestyle Factors": ['Physical Activity Level', 'Smoking Status', 'Alcohol Consumption', 'Sleep Quality', 'Dietary Habits', 'Stress Levels'],
        "🧬 Risk Factors": ["Family History of Alzheimer’s", 'Genetic Risk Factor (APOE-ε4 allele)', 'Social Engagement Level'],
        "🌍 Environment": ['Urban vs Rural Living', 'Air Pollution Exposure']
    }
    
    for group_name, features in feature_groups.items():
        with st.expander(group_name, expanded=True):
            for feature in features:
                if feature in CATEGORICAL_OPTIONS:
                    value = st.selectbox(
                        f"**{feature}**:", 
                        options=CATEGORICAL_OPTIONS[feature],
                        key=feature,
                        help=f"Select your {feature.lower()}"
                    )
                    user_data[feature] = value
                elif feature in NUMERICAL_FEATURES:
                    if feature == 'Age':
                        value = st.number_input(
                            f"**{feature}** (years):", 
                            min_value=10, max_value=120, value=65, step=1, key=feature,
                            help="Your current age in years"
                        )
                    elif feature == 'BMI':
                        value = st.number_input(
                            f"**{feature}** (kg/m²):", 
                            min_value=10.0, max_value=50.0, value=25.0, step=0.1, key=feature,
                            help="Body Mass Index: weight in kg divided by height in meters squared"
                        )
                    elif feature == 'Cognitive Test Score':
                        value = st.number_input(
                            f"**{feature}** (0-30):", 
                            min_value=0, max_value=30, value=25, step=1, key=feature,
                            help="Mini-Mental State Exam score (higher is better)"
                        )
                    elif feature == 'Depression Level':
                        value = st.number_input(
                            f"**{feature}** (0-15, higher = more depressed):", 
                            min_value=0, max_value=15, value=2, step=1, key=feature,
                            help="Depression severity scale (0=none, 15=severe)"
                        )
                    elif feature == 'Stress Levels':
                        value = st.number_input(
                            f"**{feature}** (0-10, higher = more stress):", 
                            min_value=0, max_value=10, value=5, step=1, key=feature,
                            help="Perceived stress level (0=no stress, 10=maximum stress)"
                        )
                    else:
                        value = st.number_input(f"**{feature}**:", key=feature, step=1.0)
                    
                    user_data[feature] = value
    return pd.DataFrame([user_data])

def make_prediction(user_input_df):
    try:
        user_age = user_input_df['Age'].iloc[0]
        
        if user_age < 50:
            st.success("✅ Low Risk Assessment")
            st.write("**Alzheimer's Risk: Very Low**")
            st.write("Age under 50 typically indicates very low risk. Continue healthy lifestyle practices!")
            st.write("---")
            st.write("### 💡 Maintenance Strategies")
            st.success(
                "• Continue current healthy lifestyle practices\n"
                "• Maintain regular physical activity and social engagement\n"
                "• Keep challenging your brain with new activities"
            )
            return "Low"
        
        input_data = pd.DataFrame({
            'Country': [user_input_df['Country'].iloc[0]],
            'Age': [user_input_df['Age'].iloc[0]],
            'Gender': [user_input_df['Gender'].iloc[0]],
            'Education Level': [user_input_df['Education Level'].iloc[0]],
            'BMI': [user_input_df['BMI'].iloc[0]],
            'Physical Activity Level': [user_input_df['Physical Activity Level'].iloc[0]],
            'Smoking Status': [user_input_df['Smoking Status'].iloc[0]],
            'Alcohol Consumption': [user_input_df['Alcohol Consumption'].iloc[0]],
            'Diabetes': [user_input_df['Diabetes'].iloc[0]],
            'Hypertension': [user_input_df['Hypertension'].iloc[0]],
            'Cholesterol Level': [user_input_df['Cholesterol Level'].iloc[0]],
            'Family History of Alzheimer’s': [user_input_df["Family History of Alzheimer’s"].iloc[0]],
            'Cognitive Test Score': [user_input_df['Cognitive Test Score'].iloc[0]],
            'Depression Level': [user_input_df['Depression Level'].iloc[0]],
            'Sleep Quality': [user_input_df['Sleep Quality'].iloc[0]],
            'Dietary Habits': [user_input_df['Dietary Habits'].iloc[0]],
            'Air Pollution Exposure': [user_input_df['Air Pollution Exposure'].iloc[0]],
            'Employment Status': [user_input_df['Employment Status'].iloc[0]],
            'Marital Status': [user_input_df['Marital Status'].iloc[0]],
            'Genetic Risk Factor (APOE-ε4 allele)': [user_input_df['Genetic Risk Factor (APOE-ε4 allele)'].iloc[0]],
            'Social Engagement Level': [user_input_df['Social Engagement Level'].iloc[0]],
            'Income Level': [user_input_df['Income Level'].iloc[0]],
            'Stress Levels': [user_input_df['Stress Levels'].iloc[0]],
            'Urban vs Rural Living': [user_input_df['Urban vs Rural Living'].iloc[0]]
        })
        
        input_encoded = input_data.copy()
        encoding_maps = {}
        for feature, options in CATEGORICAL_OPTIONS.items():
            if feature in ['Diabetes', 'Hypertension', "Family History of Alzheimer’s", 'Genetic Risk Factor (APOE-ε4 allele)']:
                if 'Yes' in options and 'No' in options:
                    encoding_maps[feature] = {'No': 0, 'Yes': 1}
                else:
                    encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
            else:
                encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
        
        for column in input_data.columns:
            if column in CATEGORICAL_OPTIONS:
                original_value = input_data[column].iloc[0]
                if column in encoding_maps and original_value in encoding_maps[column]:
                    input_encoded[column] = encoding_maps[column][original_value]
                else:
                    input_encoded[column] = 0

        for col in input_encoded.columns:
            input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)
        
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features = list(model.feature_names_in_)
            model_input = pd.DataFrame()
            for expected_feature in expected_features:
                if expected_feature in input_encoded.columns:
                    model_input[expected_feature] = input_encoded[expected_feature]
                else:
                    found_match = False
                    for input_col in input_encoded.columns:
                        if (expected_feature.replace("'", "'") == input_col or 
                            expected_feature == input_col.replace("'", "'")):
                            model_input[expected_feature] = input_encoded[input_col]
                            found_match = True
                            break
                    if not found_match:
                        model_input[expected_feature] = 0
            input_encoded = model_input
        
        input_scaled = scaler.transform(input_encoded)
        raw_probabilities = model.predict_proba(input_scaled)[0]
        alzheimers_risk = raw_probabilities[1] * 100
        
        st.write("### 🎯 Your Risk Assessment")
        st.metric("Alzheimer's Risk Assessment", f"{alzheimers_risk:.1f}%", help="Raw model prediction probability")
        
        if alzheimers_risk >= 70:
            st.error("⚠️ High Risk Assessment")
            st.write(f"**Alzheimer's Risk: {alzheimers_risk:.1f}%**")
            st.write("The model indicates elevated risk factors. Please consult healthcare professionals.")
            st.write("### 💡 Recommendations")
            st.error(
                "• Schedule consultation with healthcare provider\n"
                "• Consider neurological evaluation\n"
                "• Implement comprehensive brain-healthy lifestyle changes"
            )
        elif alzheimers_risk >= 30:
            st.warning("🔶 Moderate Risk Assessment")
            st.write(f"**Alzheimer's Risk: {alzheimers_risk:.1f}%**")
            st.write("The model shows moderate risk factors that warrant attention.")
            st.write("### 💡 Recommendations")
            st.warning(
                "• Increase physical activity and cognitive challenges\n"
                "• Adopt brain-healthy diet (Mediterranean/MIND diet)\n"
                "• Improve sleep quality and stress management"
            )
        else:
            st.success("✅ Low Risk Assessment")
            st.write(f"**Alzheimer's Risk: {alzheimers_risk:.1f}%**")
            st.write("Your current health profile indicates lower risk factors.")
            st.write("### 💡 Recommendations")
            st.success(
                "• Continue current healthy lifestyle practices\n"
                "• Maintain regular physical activity and social engagement\n"
                "• Keep challenging your brain with new activities"
            )

        st.write("---")
        st.error(
            "⚠️ MEDICAL DISCLAIMER: This tool provides educational insights only. "
            "Always consult healthcare professionals for medical decisions."
        )
        
        return "Complete"
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

# Header
st.write("# 🧠 Alzheimer's Risk Assessment")
st.write("*AI-powered risk evaluation with personalized insights*")

# Enhanced legal disclaimer at the top
st.warning(
    "⚠️ IMPORTANT MEDICAL DISCLAIMER\n\n"
    "This tool is for EDUCATIONAL PURPOSES ONLY and should never be used for actual medical diagnosis. "
    "The predictions are based on statistical models and should not replace professional medical evaluation. "
    "Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions."
)

with st.expander("🔬 How it works", expanded=False):
    st.write(
        "Our machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations.\n\n"
        "**Key Features:**\n"
        "- Comprehensive health factor analysis\n"
        "- Personalized risk scoring\n"
        "- Evidence-based recommendations\n"
        "- Mobile-friendly interface"
    )

user_input_df = get_user_input()

st.write("")
st.write("### 🎯 Get Your Risk Assessment")

if st.button("🧪 Analyze My Alzheimer's Risk", type="primary", use_container_width=True):
    with st.spinner("🔍 Processing your health data..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            critical_fields = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']
            missing_fields = []
            
            for field in critical_fields:
                if pd.isna(user_input_df[field].iloc[0]) or user_input_df[field].iloc[0] in [0, None, ""]:
                    missing_fields.append(field)
            
            if missing_fields:
                progress_bar.empty()
                st.error(f"⚠️ Missing Required Information: {', '.join(missing_fields)}")
                st.info("Please fill in all fields for an accurate assessment.")
                st.stop()
            
            progress_bar.empty()
            make_prediction(user_input_df)
                
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Error during prediction: {str(e)}")
            st.error("Please check your inputs and try again.")

st.write("---")
st.write("## 📖 Educational Resources")

if 'brain_tip' not in st.session_state:
    st.session_state.brain_tip = None
if 'lifestyle_tip' not in st.session_state:
    st.session_state.lifestyle_tip = None
if 'show_all_tips' not in st.session_state:
    st.session_state.show_all_tips = False

brain_tips = [
    "🧠 Challenge your mind with puzzles, reading, or learning new skills daily.",
    "🎵 Listen to music or learn to play an instrument to boost cognitive function.",
    "🎯 Practice mindfulness and meditation for better brain health.",
    "👥 Stay socially active and maintain meaningful relationships.",
    "🎨 Engage in creative activities like painting, writing, or crafts."
]

lifestyle_tips = [
    "🏃 Aim for at least 150 minutes of moderate exercise weekly.",
    "🥗 Follow a Mediterranean or MIND diet rich in omega-3s.",
    "😴 Get 7-9 hours of quality sleep each night.",
    "🚭 Avoid smoking and limit alcohol consumption.",
    "💧 Stay hydrated and maintain a healthy weight."
]

st.write("### 🧠 Brain Health Tips")
st.info(
    "Key Brain Health Strategies:\n"
    "• Stay Physically Active: Regular exercise increases blood flow to the brain\n"
    "• Challenge Your Mind: Learn new skills, read, solve puzzles\n"
    "• Eat Brain-Healthy Foods: Mediterranean diet rich in omega-3s\n"
    "• Get Quality Sleep: 7-9 hours nightly for memory consolidation\n"
    "• Stay Social: Maintain relationships and community connections"
)

st.write("### ⚠️ Warning Signs to Watch")
st.info(
    "Important Warning Signs:\n"
    "• Memory Loss: Forgetting recently learned information\n"
    "• Planning Problems: Difficulty with familiar tasks\n"
    "• Confusion: Losing track of time or place\n"
    "• Language Issues: Trouble finding the right words\n"
    "• Mood Changes: Depression, anxiety, or personality changes\n\n"
    "If you notice these signs, consult a healthcare professional."
)

st.write("### 💡 Get Personalized Tips")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧠 Brain Tip", use_container_width=True):
        st.session_state.brain_tip = random.choice(brain_tips)
with col2:
    if st.button("🌟 Lifestyle Tip", use_container_width=True):
        st.session_state.lifestyle_tip = random.choice(lifestyle_tips)
with col3:
    if st.button("📋 All Tips", use_container_width=True):
        st.session_state.show_all_tips = not st.session_state.show_all_tips

if st.session_state.brain_tip:
    st.success(f"🧠 **Brain Tip:** {st.session_state.brain_tip}")
if st.session_state.lifestyle_tip:
    st.success(f"🌟 **Lifestyle Tip:** {st.session_state.lifestyle_tip}")

if st.session_state.show_all_tips:
    st.write("---")
    st.write("#### 🧠 All Brain Health Tips")
    for i, tip in enumerate(brain_tips, 1):
        st.write(f"{i}. {tip}")
    st.write("#### 🌟 All Lifestyle Tips")
    for i, tip in enumerate(lifestyle_tips, 1):
        st.write(f"{i}. {tip}")

st.write("---")
st.write("### 🌟 Take Control of Your Brain Health")
st.info(
    "Knowledge is power. Use these insights to make informed decisions about your health and lifestyle. "
    "Remember, many risk factors for Alzheimer's disease are modifiable through healthy choices.\n\n"
    "Useful Resources:\n"
    "- Alzheimer's Association (alz.org)\n"
    "- National Institute on Aging (nia.nih.gov)\n"
    "- Brain Health Research (brainhealthregistry.org)"
)

st.write("---")
st.info(
    "💡 Remember: This tool is for educational purposes only. Always consult with healthcare professionals "
    "for medical advice, diagnosis, or treatment decisions. Early detection and lifestyle modifications can "
    "make a significant difference in brain health outcomes."
)
