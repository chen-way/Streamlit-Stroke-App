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

# SINGLE, CLEAN CSS SECTION - NO CONFLICTS
st.markdown("""
   <style>
:root {
  --bg:#f0f7fc;
  --card:#ffffff;
  --radius:16px;
  --shadow-light:rgba(255,255,255,0.8);
  --shadow-dark:rgba(0,0,0,0.08);
  --primary:#4f8fd9;
  --primary-light:#d6e8fd;
  --accent:#ffb366;
  --text:#2e3d49;
  --muted:#6e7e8f;
  --transition:0.35s cubic-bezier(.4,.2,.2,1);
  --border:1px solid rgba(81, 122, 167, 0.2);
  font-family: "Inter","Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

body, .stApp {
  background: var(--bg) !important;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3 {
  font-weight: 600;
  letter-spacing: -0.5px;
  margin-bottom: 0.35rem;
}

p, li, .stMarkdown {
  line-height: 1.45;
  font-size: 0.95rem;
}

.stApp .block-container {
  padding: 1rem 1rem 2rem !important;
  max-width: 1024px;
  margin: 0 auto;
}

/* Card / container styling */
.card, .stCard, .feature-container, .tips-container {
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: 0 18px 30px -10px var(--shadow-dark), 0 8px 18px -6px var(--shadow-dark), inset 0 0 0 1px rgba(81, 122, 167, 0.08);
  border: var(--border);
  transition: all var(--transition);
}

/* Header */
.main-header {
  background: linear-gradient(135deg, #e0f0fc 0%, #c9e6fa 60%) !important;
  padding: 1rem 1rem !important;
  border-radius: var(--radius);
  text-align: center;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
  border: none !important;
}

.main-header h1 {
  font-size: 2rem !important;
  margin: 0;
}

.main-header p {
  margin: 0.25rem 0 0;
  font-weight: 400;
  color: var(--muted);
}

/* Metric styling */
.stMetric {
  background: var(--primary-light) !important;
  border-radius: 14px !important;
  padding: 0.75rem 1rem !important;
  box-shadow: 0 12px 24px -4px rgba(79, 143, 217, 0.2);
  border: none !important;
}

.metric-container {
  background: white;
  border-radius: 14px;
  padding: 1rem;
  box-shadow: 0 14px 32px -6px rgba(0,0,0,0.08);
  margin-bottom: 0.75rem;
}

/* Buttons */
.stButton > button {
  background: var(--primary);
  color: white !important;
  font-weight: 600;
  border-radius: 12px;
  padding: 14px 24px;
  border: none;
  min-height: 48px;
  letter-spacing: 0.5px;
  box-shadow: 0 14px 28px -6px rgba(79, 143, 217, 0.35);
  transition: all var(--transition);
  cursor: pointer;
}
.stButton > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}
.stButton > button:active {
  transform: translateY(1px);
}

/* Inputs & selects */
div[data-baseweb="input"] > div,
.stSelectbox > div > div,
div[data-testid="stNumberInput"] > div {
  background: #f7fbff !important;
  border-radius: 10px !important;
  border: 2px solid rgba(79, 143, 217, 0.25) !important;
  padding: 10px !important;
  transition: all var(--transition);
}
div[data-baseweb="input"] input,
div[data-testid="stNumberInput"] input {
  font-size: 15px !important;
  background: transparent !important;
  outline: none !important;
}

/* Risk boxes */
.result-high-risk, .result-low-risk, .result-moderate-risk {
  padding: 1rem 1rem;
  border-radius: 14px;
  margin: 1rem 0;
  position: relative;
  font-weight: 500;
}
.result-high-risk {
  background: rgba(255, 102, 102, 0.1);
  border: 1px solid #ff6b6b;
  color: #c0392b;
}
.result-moderate-risk {
  background: rgba(255, 191, 0, 0.1);
  border: 1px solid #ffb366;
  color: #7f6000;
}
.result-low-risk {
  background: rgba(76, 209, 55, 0.1);
  border: 1px solid #28a745;
  color: #1b6e3a;
}

/* Tip badges */
.stInfo, .stSuccess, .stWarning, .stError {
  border-radius: 12px !important;
  padding: 0.85rem 1rem !important;
  font-size: 0.9rem !important;
}

/* Headers inside expanders */
.streamlit-expanderHeader {
  font-size: 1.05rem !important;
  font-weight: 600;
}

/* Footer / resource section */
footer, .footer {
  display: none;
}

/* Subtle floating accent animation */
.pulse-animation {
  animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(79, 143, 217, 0.2); }
  50% { box-shadow: 0 0 25px 5px rgba(79, 143, 217, 0.15); }
  100% { box-shadow: 0 0 0 0 rgba(79, 143, 217, 0.2); }
}

/* Responsive tweaks */
@media (max-width: 900px) {
  .stButton > button {
    width: 100% !important;
  }
  .card, .feature-container {
    padding: 1rem !important;
  }
}
</style>
""", unsafe_allow_html=True)

# Enhanced page configuration with custom styling
st.set_page_config(
    page_title="Alzheimer's Risk Assessment", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Load models
@st.cache_resource
def load_models():
    """Load the trained Random Forest model and preprocessing objects"""
    try:
        model = joblib.load('model_compressed.pkl.gz')
        scaler = joblib.load('scaler_compressed.pkl.gz')
        encoders = joblib.load('encoders_compressed.pkl.gz')
        return model, scaler, encoders
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error("Looking for: model_compressed.pkl.gz, scaler_compressed.pkl.gz, encoders_compressed.pkl.gz")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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
    "Family History of Alzheimer's": ['Yes', 'No'],
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
    "Family History of Alzheimer's", 'Cognitive Test Score', 'Depression Level',
    'Sleep Quality', 'Dietary Habits', 'Air Pollution Exposure',
    'Employment Status', 'Marital Status', 'Genetic Risk Factor (APOE-ε4 allele)',
    'Social Engagement Level', 'Income Level', 'Stress Levels', 'Urban vs Rural Living'
]

def get_user_input():
    user_data = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    feature_count = 0
    for feature in feature_names:
        # Alternate between columns
        current_col = col1 if feature_count % 2 == 0 else col2
        
        with current_col:
            if feature in CATEGORICAL_OPTIONS:
                # Categorical features with dropdowns
                value = st.selectbox(
                    f"**{feature}**:", 
                    options=CATEGORICAL_OPTIONS[feature],
                    key=feature
                )
                user_data[feature] = value
            elif feature in NUMERICAL_FEATURES:
                # Numerical features with appropriate ranges and step sizes
                if feature == 'Age':
                    value = st.number_input(
                        f"**{feature}** (years):", 
                        min_value=10, max_value=120, value=65, step=1, key=feature
                    )
                elif feature == 'BMI':
                    value = st.number_input(
                        f"**{feature}** (kg/m²):", 
                        min_value=10.0, max_value=50.0, value=25.0, step=0.1, key=feature
                    )
                elif feature == 'Cognitive Test Score':
                    value = st.number_input(
                        f"**{feature}** (0-30):", 
                        min_value=0, max_value=30, value=25, step=1, key=feature
                    )
                elif feature == 'Depression Level':
                    value = st.number_input(
                        f"**{feature}** (0-15, higher = more depressed):", 
                        min_value=0, max_value=15, value=2, step=1, key=feature
                    )
                elif feature == 'Stress Levels':
                    value = st.number_input(
                        f"**{feature}** (0-10, higher = more stress):", 
                        min_value=0, max_value=10, value=5, step=1, key=feature
                    )
                else:
                    value = st.number_input(f"**{feature}**:", key=feature, step=1.0)
                
                user_data[feature] = value
        
        feature_count += 1
    
    return pd.DataFrame([user_data])

def make_prediction(user_input_df):
    try:
        # Get user's age for safety check
        user_age = user_input_df['Age'].iloc[0]
        
        # Under 50 automatic low risk check
        if user_age < 50:
            st.markdown(f"""
            <div class="result-low-risk">
                <h2>✅ Low Risk Assessment</h2>
                <h3>Alzheimer's Risk: Very Low</h3>
                <p>Age under 50 typically indicates very low risk. Continue healthy lifestyle practices!</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Maintenance Strategies")
            st.success("""
            **Maintenance Strategies:**
            • Continue current healthy lifestyle practices
            • Maintain regular physical activity and social engagement
            • Keep challenging your brain with new activities
            """)
            return "Low"
        
        # Create input dataframe
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
            'Family History of Alzheimer’s': [user_input_df["Family History of Alzheimer's"].iloc[0]],
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
        
        # FIXED: Manual encoding with proper Yes/No handling
        input_encoded = input_data.copy()
        encoding_maps = {}
        for feature, options in CATEGORICAL_OPTIONS.items():
            if feature in ['Diabetes', 'Hypertension', "Family History of Alzheimer's", 'Genetic Risk Factor (APOE-ε4 allele)']:
                # For Yes/No features, ensure 'No'=0 and 'Yes'=1 (standard medical encoding)
                if 'Yes' in options and 'No' in options:
                    encoding_maps[feature] = {'No': 0, 'Yes': 1}
                else:
                    encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
            else:
                encoding_maps[feature] = {option: idx for idx, option in enumerate(options)}
        
        # Encode categorical variables
        for column in input_data.columns:
            if column in CATEGORICAL_OPTIONS:
                original_value = input_data[column].iloc[0]
                if column in encoding_maps and original_value in encoding_maps[column]:
                    input_encoded[column] = encoding_maps[column][original_value]
                else:
                    input_encoded[column] = 0

        
        # Ensure all columns are numeric
        for col in input_encoded.columns:
            input_encoded[col] = pd.to_numeric(input_encoded[col], errors='coerce').fillna(0)
        
        # Handle feature name matching
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features = list(model.feature_names_in_)
            model_input = pd.DataFrame()
            for expected_feature in expected_features:
                if expected_feature in input_encoded.columns:
                    model_input[expected_feature] = input_encoded[expected_feature]
                else:
                    # Handle apostrophe variations
                    found_match = False
                    for input_col in input_encoded.columns:
                        if (expected_feature.replace("'", "'") == input_col or 
                            expected_feature.replace("'", "'") == input_col or
                            expected_feature == input_col.replace("'", "'") or
                            expected_feature == input_col.replace("'", "'")):
                            model_input[expected_feature] = input_encoded[input_col]
                            found_match = True
                            break
                    if not found_match:
                        model_input[expected_feature] = 0
            input_encoded = model_input
        
        # Scale and predict
        input_scaled = scaler.transform(input_encoded)
        raw_probabilities = model.predict_proba(input_scaled)[0]
        alzheimers_risk = raw_probabilities[1] * 100
        
        # Display main risk metric
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric("Alzheimer's Risk Assessment", f"{alzheimers_risk:.1f}%", 
                     help="Raw model prediction probability")
        
        # Risk interpretation based on actual model output
        if alzheimers_risk >= 70:  # High risk
            st.markdown(f"""
            <div class="result-high-risk">
                <h2>⚠️ High Risk Assessment</h2>
                <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                <p>The model indicates elevated risk factors. Please consult healthcare professionals.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            st.error("""
            **High Priority Actions:**
            • Schedule consultation with healthcare provider
            • Consider neurological evaluation
            • Implement comprehensive brain-healthy lifestyle changes
            """)
            
        elif alzheimers_risk >= 30:  # Moderate risk
            st.markdown(f"""
            <div class="result-moderate-risk">
                <h2>🔶 Moderate Risk Assessment</h2>
                <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                <p>The model shows moderate risk factors that warrant attention.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            st.warning("""
            **Moderate Priority Actions:**
            • Increase physical activity and cognitive challenges
            • Adopt brain-healthy diet (Mediterranean/MIND diet)
            • Improve sleep quality and stress management
            """)
            
        else:  # Low risk
            st.markdown(f"""
            <div class="result-low-risk">
                <h2>✅ Low Risk Assessment</h2>
                <h3>Alzheimer's Risk: {alzheimers_risk:.1f}%</h3>
                <p>Your current health profile indicates lower risk factors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            st.success("""
            **Maintenance Strategies:**
            • Continue current healthy lifestyle practices
            • Maintain regular physical activity and social engagement
            • Keep challenging your brain with new activities
            """)

        # Legal disclaimer
        st.markdown("---")
        st.error("""
        ⚠️ **MEDICAL DISCLAIMER:** This tool provides educational insights only. 
        Always consult healthcare professionals for medical decisions.
        """)
        
        return "Complete"
        
    except Exception as e:
        st.error(f"❌ **Error during prediction:** {str(e)}")
        return None

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Alzheimer's Risk Assessment</h1>
    <p>Advanced AI-powered risk evaluation with personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Enhanced legal disclaimer at the top
st.markdown("""
<div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
    <h4 style="color: #856404; margin-top: 0;">⚠️ IMPORTANT MEDICAL DISCLAIMER</h4>
    <p style="color: #856404; margin: 0;">
        <strong>This tool is for EDUCATIONAL PURPOSES ONLY</strong> and should never be used for actual medical diagnosis. 
        The predictions are based on statistical models and should not replace professional medical evaluation. 
        Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Information section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.expander("🔬 How it works", expanded=False):
        st.markdown("""
        Our advanced machine learning model analyzes 24 comprehensive health factors to provide personalized risk assessment and evidence-based recommendations.
        """)

# Get user input
user_input_df = get_user_input()

# === PREDICTION SECTION ===
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #d1e5f4; padding: 1rem; border-radius: 10px; margin: 1rem 0; border: 1px solid #93BCDC;">
    <h3 style="color: #2d3436; text-align: center; margin: 0;">🎯 Risk Assessment</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🧪 Analyze My Alzheimer's Risk", type="primary", use_container_width=True):
        
        with st.spinner("🔍 Processing your health data..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            try:
                # Check for missing critical values
                critical_fields = ['Age', 'BMI', 'Cognitive Test Score', 'Depression Level', 'Stress Levels']
                missing_fields = []
                
                for field in critical_fields:
                    if pd.isna(user_input_df[field].iloc[0]) or user_input_df[field].iloc[0] in [0, None, ""]:
                        missing_fields.append(field)
                
                if missing_fields:
                    progress_bar.empty()
                    st.error(f"⚠️ **Missing Required Information:** {', '.join(missing_fields)}")
                    st.info("Please fill in all fields for an accurate assessment.")
                    st.stop()
                
                # Clear progress bar
                progress_bar.empty()
                
                # Make prediction
                make_prediction(user_input_df)
                    
            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ **Error during prediction:** {str(e)}")
                st.error("Please check your inputs and try again.")

# Educational content
st.markdown("---")
st.markdown("## 📖 Educational Resources")

# Initialize session state variables for tips
if 'brain_tip' not in st.session_state:
    st.session_state.brain_tip = None
if 'lifestyle_tip' not in st.session_state:
    st.session_state.lifestyle_tip = None
if 'show_all_tips' not in st.session_state:
    st.session_state.show_all_tips = False

# Brain health and lifestyle tips
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

# First row: Brain Health Tips and Warning Signs (equal height)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #FDF6E7; padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid #93BCDC; height: 400px; overflow-y: auto;">
        <h3 style="color: #2d3436; text-align: center; margin-bottom: 1.5rem;">🧠 Brain Health Tips</h3>
        <ul>
            <li><strong>Stay Physically Active:</strong> Regular exercise increases blood flow to the brain</li>
            <li><strong>Challenge Your Mind:</strong> Learn new skills, read, solve puzzles</li>
            <li><strong>Eat Brain-Healthy Foods:</strong> Mediterranean diet rich in omega-3s</li>
            <li><strong>Get Quality Sleep:</strong> 7-9 hours nightly for memory consolidation</li>
            <li><strong>Stay Social:</strong> Maintain relationships and community connections</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #FDF6E7; padding: 2rem; border-radius: 20px; margin: 2rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border: 1px solid #93BCDC; height: 400px; overflow-y: auto;">
        <h3 style="color: #2d3436; text-align: center; margin-bottom: 1.5rem;">⚠️ Warning Signs to Watch</h3>
        <ul>
            <li><strong>Memory Loss:</strong> Forgetting recently learned information</li>
            <li><strong>Planning Problems:</strong> Difficulty with familiar tasks</li>
            <li><strong>Confusion:</strong> Losing track of time or place</li>
            <li><strong>Language Issues:</strong> Trouble finding the right words</li>
            <li><strong>Mood Changes:</strong> Depression, anxiety, or personality changes</li>
        </ul>
        <p><strong>If you notice these signs, consult a healthcare professional.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Second row: Interactive Tips Section
st.markdown("### 💡 Get Personalized Tips")
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🧠 Get Brain Tip", use_container_width=True):
        st.session_state.brain_tip = random.choice(brain_tips)

with col_b:
    if st.button("🌟 Get Lifestyle Tip", use_container_width=True):
        st.session_state.lifestyle_tip = random.choice(lifestyle_tips)

with col_c:
    if st.button("📋 Show All Tips", use_container_width=True):
        st.session_state.show_all_tips = not st.session_state.show_all_tips

# Display random tips
if st.session_state.brain_tip:
    st.success(f"🧠 **Brain Tip:** {st.session_state.brain_tip}")
if st.session_state.lifestyle_tip:
    st.success(f"🌟 **Lifestyle Tip:** {st.session_state.lifestyle_tip}")

# Show all tips section
if st.session_state.show_all_tips:
    st.markdown("---")
    col_tips1, col_tips2 = st.columns(2)
    
    with col_tips1:
        st.markdown("#### 🧠 All Brain Health Tips")
        for i, tip in enumerate(brain_tips, 1):
            st.write(f"{i}. {tip}")
    
    with col_tips2:
        st.markdown("#### 🌟 All Lifestyle Tips")
        for i, tip in enumerate(lifestyle_tips, 1):
            st.write(f"{i}. {tip}")

# Footer with additional resources
st.markdown("---")

# Resources section with proper styling - all content in one HTML block
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="background-color: #d1e5f4; padding: 2rem; border-radius: 15px; text-align: center; border: 1px solid #93BCDC; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #2d3436; margin-bottom: 1rem;">🌟 Take Control of Your Brain Health</h3>
        <p style="color: #636e72; margin-bottom: 1.5rem;">
            Knowledge is power. Use these insights to make informed decisions about your health and lifestyle. 
            Remember, many risk factors for Alzheimer's disease are modifiable through healthy choices.
        </p>
        <h4 style="color: #2d3436; margin-bottom: 1rem;">📚 Useful Resources:</h4>
        <p style="color: #636e72; margin-bottom: 0.5rem;">• <strong>Alzheimer's Association:</strong> <a href="https://alz.org" target="_blank" style="color: #007bff; text-decoration: none;">alz.org</a></p>
        <p style="color: #636e72; margin-bottom: 0.5rem;">• <strong>National Institute on Aging:</strong> <a href="https://nia.nih.gov" target="_blank" style="color: #007bff; text-decoration: none;">nia.nih.gov</a></p>
        <p style="color: #636e72; margin-bottom: 0;">• <strong>Brain Health Research:</strong> <a href="https://brainhealthregistry.org" target="_blank" style="color: #007bff; text-decoration: none;">brainhealthregistry.org</a></p>
    </div>
    """, unsafe_allow_html=True)

# Final disclaimer
st.markdown("---")
st.info("""
💡 **Remember:** This tool is for educational purposes only. Always consult with healthcare professionals 
for medical advice, diagnosis, or treatment decisions. Early detection and lifestyle modifications can 
make a significant difference in brain health outcomes.
""")
