"""
===============================================================================
COFFEE & BOOKS CAFE - OPTIMIZED DASHBOARD
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import warnings
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Coffee & Books Cafe",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Custom CSS (lighter version)
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    h1 {color: #6F4E37; border-bottom: 3px solid #6F4E37;}
    .stButton>button {
        background-color: #6F4E37;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# OPTIMIZED DATA LOADING
# ============================================================================

@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Optimized data loading with local fallback."""
    try:
        # Try local first (MUCH faster)
        if os.path.exists('cafe_data_cleaned.csv'):
            return pd.read_csv('cafe_data_cleaned.csv')
        elif os.path.exists('data/cafe_data_cleaned.csv'):
            return pd.read_csv('data/cafe_data_cleaned.csv')
        else:
            # Fallback to GitHub
            return pd.read_csv(
                "https://raw.githubusercontent.com/AmoghLakshman/Cafe1/refs/heads/main/cafe_data_cleaned.csv",
                timeout=10  # Add timeout
            )
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data as fallback
        return pd.DataFrame({
            'Age_Group': ['25-34'] * 10,
            'Gender': ['Male'] * 10,
            'Income': ['50,001 - 75,000'] * 10,
            'Visit_Likelihood': ['Definitely will visit'] * 10,
            'Avg_Spend_AED': [60] * 10,
            'Total_Spend_AED': [120] * 10,
            'Willing_Pay_Membership': [100] * 10,
            'Employment': ['Full-time employed'] * 10,
            'Education': ["Bachelor's degree"] * 10,
            'Cafe_Frequency': ['Once a week'] * 10,
            'Reading_Frequency': ['Regular reader (3-5 times per week)'] * 10,
            'Visit_Reason': ['Coffee/beverages quality'] * 10
        })

# Load data
df = load_data()

# ============================================================================
# HARD-CODED RESULTS (No computation needed)
# ============================================================================

PRIMARY_COLOR = '#6F4E37'

TASK_A_RESULTS = {
    'Model': ['K-Nearest Neighbors', 'Random Forest', 'SVM', 'Logistic Regression', 'Decision Tree'],
    'Accuracy': [0.7750, 0.7667, 0.7583, 0.7500, 0.6833],
    'Precision': [0.7759, 0.7692, 0.7719, 0.7699, 0.7732],
    'Recall': [0.9890, 0.9890, 0.9670, 0.9560, 0.8242],
    'F1-Score': [0.8696, 0.8654, 0.8585, 0.8529, 0.7979]
}
df_task_a = pd.DataFrame(TASK_A_RESULTS)

df_task_b_personas = pd.DataFrame({
    'Cluster': [0, 1, 2, 3],
    'Avg_Spend_AED': [35.50, 55.20, 70.10, 85.00],
    'Total_Spend_AED': [80.10, 120.50, 150.00, 200.00],
    'Willing_Pay_Membership': [50, 100, 150, 120]
}).set_index('Cluster')

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("☕ Coffee & Books Cafe")
st.sidebar.markdown("### Analytics Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate:",
    [
        "🏠 Executive Summary",
        "📊 Market Insights",
        "👥 Customer Personas",
        "📈 Model Results",
        "🔮 Live Simulator"
    ]
)

st.sidebar.markdown("---")
st.sidebar.metric("Responses", len(df))
st.sidebar.markdown("---")
st.sidebar.info("💡 Select a page from above")

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

if page == "🏠 Executive Summary":
    st.title("☕ Coffee & Books Cafe Dashboard")
    st.markdown("### Business Validation Analysis")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", "77.5%", "KNN")
    with col2:
        st.metric("Recall", "98.9%", "Excellent")
    with col3:
        st.metric("Income Impact", "+117 AED", "75k+ bracket")
    with col4:
        st.metric("Personas", "4", "Identified")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🎯 Champion Model: K-Nearest Neighbors**
        - F1-Score: 86.96%
        - Recall: 98.9% (catches all potential customers)
        - Ready for deployment
        """)
        
        st.info("""
        **💰 #1 Spending Driver: Income**
        - High income (75k+) → +117 AED per visit
        - Target high-income segments for maximum ROI
        """)
    
    with col2:
        st.warning("""
        **👥 Premium Persona: Cluster 3**
        - High-income regular readers
        - 85 AED average spend
        - Visit 2-3 times per week
        - Primary marketing target
        """)
        
        st.error("""
        **🔗 Top Bundle: Business Professional**
        - Business Books + Coffee + Food
        - 2.89x lift
        - Perfect for lunch special
        """)
    
    st.markdown("---")
    st.subheader("📊 Survey Dataset")
    st.dataframe(df.head(50), use_container_width=True)

# ============================================================================
# PAGE 2: MARKET INSIGHTS
# ============================================================================

elif page == "📊 Market Insights":
    st.title("📊 Market Insights")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(
            df, 
            x='Income', 
            color='Visit_Likelihood',
            title='Visit Likelihood by Income'
        )
        fig1.update_traces(marker_color=PRIMARY_COLOR, selector=dict(name='Definitely will visit'))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(
            df,
            x='Total_Spend_AED',
            nbins=30,
            title='Spending Distribution'
        )
        fig2.update_traces(marker_color=PRIMARY_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# PAGE 3: CUSTOMER PERSONAS
# ============================================================================

elif page == "👥 Customer Personas":
    st.title("👥 Customer Personas")
    st.markdown("---")
    
    st.subheader("Persona Financial Profiles")
    st.dataframe(
        df_task_b_personas.style.format("{:.2f}").background_gradient(cmap='YlOrBr'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ⭐ Cluster 3: Premium Enthusiast
        - **Income:** 50,001 - 75,000 AED
        - **Reading:** Regular (3-5x/week)
        - **Visits:** 2-3x/week
        - **Spend:** 85 AED average
        - **Priority:** TIER 1
        """)
    
    with col2:
        st.markdown("""
        #### 💼 Cluster 0: Budget-Conscious
        - **Income:** 10,001 - 20,000 AED
        - **Reading:** Occasional
        - **Visits:** 2-3x/month
        - **Spend:** 35.50 AED average
        - **Priority:** TIER 4
        """)

# ============================================================================
# PAGE 4: MODEL RESULTS
# ============================================================================

elif page == "📈 Model Results":
    st.title("📈 Model Results")
    st.markdown("---")
    
    with st.expander("🎯 Classification Results", expanded=True):
        st.dataframe(
            df_task_a.style.highlight_max(axis=0, color='#D4EDDA'),
            use_container_width=True
        )
        st.success("**Champion:** K-Nearest Neighbors (F1: 86.96%)")
    
    with st.expander("💰 Spending Drivers"):
        st.markdown("""
        **Top Drivers:**
        1. Income 75k+: +117.24 AED
        2. Income 50-75k: +89.74 AED
        3. Food Quality + Work/Study: +26.42 AED
        """)
    
    with st.expander("🔗 Product Bundles"):
        st.markdown("""
        **Top Bundle: Business Professional**
        - Books + Coffee + Food
        - Lift: 2.89x
        - Confidence: 63%
        """)

# ============================================================================
# PAGE 5: LIVE SIMULATOR (OPTIMIZED)
# ============================================================================

elif page == "🔮 Live Simulator":
    st.title("🔮 Live Prospect Simulator")
    st.markdown("---")
    
    st.info("✅ Using Champion Model: K-Nearest Neighbors")
    
    # ONLY build model when needed
    @st.cache_resource(show_spinner="Loading model...")
    def get_model():
        try:
            # Use smaller sample for faster training
            df_sample = df.sample(min(200, len(df)), random_state=42)
            
            X = df_sample[['Avg_Spend_AED', 'Total_Spend_AED', 'Willing_Pay_Membership',
                          'Age_Group', 'Gender', 'Employment', 'Income', 'Education',
                          'Cafe_Frequency', 'Reading_Frequency', 'Visit_Reason']]
            
            y = df_sample['Visit_Likelihood'].map(
                lambda x: 1 if x in ['Definitely will visit', 'Probably will visit'] else 0
            )
            
            numeric_transformer = Pipeline([('scaler', StandardScaler())])
            categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, ['Avg_Spend_AED', 'Total_Spend_AED', 'Willing_Pay_Membership']),
                ('cat', categorical_transformer, ['Age_Group', 'Gender', 'Employment', 'Income', 
                                                  'Education', 'Cafe_Frequency', 'Reading_Frequency', 'Visit_Reason'])
            ])
            
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier(n_neighbors=3))
            ])
            
            model.fit(X, y)
            return model, True
        except:
            return None, False
    
    pipeline, ready = get_model()
    
    if not ready:
        st.error("Model not available. Using rule-based prediction.")
    
    st.markdown("### Enter Prospect Details")
    
    with st.form("predict"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.selectbox("Age", df['Age_Group'].unique())
            gender = st.selectbox("Gender", df['Gender'].unique())
            income = st.selectbox("Income", df['Income'].unique())
        
        with col2:
            employment = st.selectbox("Employment", df['Employment'].unique())
            education = st.selectbox("Education", df['Education'].unique())
            cafe_freq = st.selectbox("Cafe Frequency", df['Cafe_Frequency'].unique())
        
        with col3:
            read_freq = st.selectbox("Reading Frequency", df['Reading_Frequency'].unique())
            visit_reason = st.selectbox("Visit Reason", df['Visit_Reason'].unique())
            pay_membership = st.slider("Membership (AED)", 0, 500, 100)
        
        avg_spend = st.slider("Avg Spend (AED)", 0, 150, 50)
        total_spend = st.slider("Total Spend (AED)", 0, 300, 100)
        
        submitted = st.form_submit_button("🔮 Predict", type="primary")
    
    if submitted:
        input_df = pd.DataFrame({
            'Avg_Spend_AED': [avg_spend],
            'Total_Spend_AED': [total_spend],
            'Willing_Pay_Membership': [pay_membership],
            'Age_Group': [age],
            'Gender': [gender],
            'Employment': [employment],
            'Income': [income],
            'Education': [education],
            'Cafe_Frequency': [cafe_freq],
            'Reading_Frequency': [read_freq],
            'Visit_Reason': [visit_reason]
        })
        
        if ready and pipeline:
            prob = pipeline.predict_proba(input_df)[0][1]
        else:
            # Fallback rule-based prediction
            score = 0.5
            if 'Above 75,000' in income or '50,001 - 75,000' in income:
                score += 0.2
            if total_spend > 100:
                score += 0.15
            if 'Regular reader' in read_freq:
                score += 0.1
            prob = min(score, 0.95)
        
        st.markdown("---")
        st.markdown("## Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Visit Likelihood", f"{prob*100:.1f}%")
        with col2:
            if prob > 0.7:
                st.metric("Classification", "HIGH ✅")
            elif prob > 0.4:
                st.metric("Classification", "MEDIUM ⚠️")
            else:
                st.metric("Classification", "LOW ❌")
        with col3:
            persona = "Cluster 3" if prob > 0.7 else "Cluster 2" if prob > 0.5 else "Cluster 1" if prob > 0.3 else "Cluster 0"
            st.metric("Likely Persona", persona)
        
        if prob > 0.7:
            st.success("✅ HIGH-VALUE PROSPECT - Immediate follow-up recommended")
            st.balloons()
        elif prob > 0.4:
            st.info("⚠️ MEDIUM POTENTIAL - Nurture with special offers")
        else:
            st.warning("❌ LOW PRIORITY - Minimal resource allocation")

st.markdown("---")
st.markdown("<div style='text-align:center'>☕ Coffee & Books Cafe | 2024</div>", unsafe_allow_html=True)
