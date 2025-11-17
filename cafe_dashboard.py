"""
===============================================================================
COFFEE & BOOKS CAFE - PROFESSIONAL ANALYTICS DASHBOARD
===============================================================================
A comprehensive multi-page Streamlit dashboard for analyzing survey data
and validating the Coffee & Books Cafe business concept.

Author: [Your Name]
Date: 2024
Version: 1.0.0
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import warnings
import time
from datetime import datetime

# ============================================================================
# 0. PAGE CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="Coffee & Books Cafe | Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Analytics Dashboard for Coffee & Books Cafe Business Validation"
    }
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    
    /* Header styling */
    h1 {
        color: #6F4E37;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #6F4E37;
    }
    
    h2 {
        color: #8B6F47;
        margin-top: 20px;
    }
    
    h3 {
        color: #6F4E37;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5F5F0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #6F4E37;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #5A3D2B;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Custom card styling */
    .custom-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# 1. DATA LOADING & CACHING
# ============================================================================

DATA_URL = "https://raw.githubusercontent.com/AmoghLakshman/Cafe1/refs/heads/main/cafe_data_cleaned.csv"

@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads and caches the cleaned survey data from GitHub.
    
    Returns:
        pd.DataFrame: Cleaned survey data or None if loading fails
    """
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please check your internet connection and the GitHub URL.")
        return None

# Load data with spinner
with st.spinner('Loading survey data...'):
    df = load_data()
    
if df is None:
    st.stop()

# ============================================================================
# 2. HARD-CODED RESULTS (Single Source of Truth)
# ============================================================================

# Color palette
PRIMARY_COLOR = '#6F4E37'
SECONDARY_COLOR = '#D2B48C'
ACCENT_COLOR = '#8B6F47'

# Task A: Classification Model Results
TASK_A_RESULTS = {
    'Model': [
        'K-Nearest Neighbors', 
        'Random Forest', 
        'Support Vector Machine (SVM)', 
        'Logistic Regression', 
        'Decision Tree'
    ],
    'Accuracy': [0.7750, 0.7667, 0.7583, 0.7500, 0.6833],
    'Precision': [0.7759, 0.7692, 0.7719, 0.7699, 0.7732],
    'Recall': [0.9890, 0.9890, 0.9670, 0.9560, 0.8242],
    'F1-Score': [0.8696, 0.8654, 0.8585, 0.8529, 0.7979]
}
df_task_a = pd.DataFrame(TASK_A_RESULTS)

# Task B: Customer Personas (Clustering Results)
# *** THIS IS THE FIX: Replaced placeholder data with your REAL results ***
TASK_B_PERSONAS_NUMERIC = {
    'Cluster': [0, 1, 2, 3],
    'Avg_Spend_AED': [26.92, 39.78, 58.78, 69.30],
    'Total_Spend_AED': [46.91, 73.88, 142.43, 168.79],
    'Willing_Pay_Membership': [67.36, 247.29, 8.06, 367.73]
}
df_task_b_personas = pd.DataFrame(TASK_B_PERSONAS_NUMERIC).set_index('Cluster')

TASK_B_PERSONAS_CATEGORICAL = {
    'Cluster 0': {
        'Income': '10,001 - 20,000 AED',
        'Reading Frequency': 'Occasional reader (1-2 times per week)',
        'Cafe Visits': '2-3 times per month',
        'Profile': '💼 Budget-Conscious Casual'
    },
    'Cluster 1': {
        'Income': '20,001 - 35,000 AED',
        'Reading Frequency': 'Regular reader (3-5 times per week)',
        'Cafe Visits': 'Once a week',
        'Profile': '📚 Middle-Income Bookworm'
    },
    'Cluster 2': {
        'Income': '50,001 - 75,000 AED',
        'Reading Frequency': 'Occasional reader (1-2 times per week)',
        'Cafe Visits': '2-3 times per week',
        'Profile': '💰 Affluent Social Visitor'
    },
    'Cluster 3': {
        'Income': '50,001 - 75,000 AED',
        'Reading Frequency': 'Regular reader (3-5 times per week)',
        'Cafe Visits': '2-3 times per week',
        'Profile': '⭐ Premium Reading Enthusiast'
    }
}

# Task C: Spending Drivers (Regression)
TASK_C_DRIVERS = {
    'Feature': [
        'Income_Above 75,000',
        'Income_50,001 - 75,000',
        'Income_Less than 5,000',
        'Income_5,000 - 10,000',
        'Visit_Reason_Food quality|Work/study...',
        'Income_10,001 - 20,000',
        'Income_35,001 - 50,000',
        'Visit_Reason_Coffee/beverages quality|Food quality'
    ],
    'Coefficient (AED)': [117.24, 89.74, -46.20, -39.10, 26.42, -16.69, 14.16, -11.61]
}
df_task_c = pd.DataFrame(TASK_C_DRIVERS)

# Task D: Association Rules (Market Basket Analysis)
TASK_D_RULES = {
    'antecedents': [
        'Non-caffeinated beverages only, Flavored Coffee..., International cuisine...',
        'Non-caffeinated beverages only, Non-Fiction - Business/Self-Help...',
        'Flavored Coffee..., Non-Fiction - Business/Self-Help, Non-caffeinated...',
        'Flavored Coffee..., Non-Fiction - Business/Self-Help, International...',
        'Fiction - Literary, Childrens/Young Adult',
        'Pastries (croissants, muffins), Non-Fiction - Biography/Memoir',
        'Breakfast items, International cuisine options',
        'Flavored Coffee (Vanilla, Caramel, Hazelnut), Arabic/Turkish Coffee',
        'Religious/Spiritual, Childrens/Young Adult',
        'Other, Childrens/Young Adult'
    ],
    'consequents': [
        'Non-Fiction - Business/Self-Help',
        'Flavored Coffee (Vanilla, Caramel, Hazelnut)',
        'International cuisine options',
        'Non-caffeinated beverages only',
        'Light snacks (cookies, biscuits)',
        'Flavored Coffee (Vanilla, Caramel, Hazelnut)',
        'Flavored Coffee (Vanilla, Caramel, Hazelnut)',
        'Pastries (croissants, muffins)',
        'No food, just beverages',
        'Desserts (cakes, brownies)'
    ],
    'support': [0.0200, 0.0200, 0.0200, 0.0200, 0.0250, 0.0383, 0.0317, 0.0333, 0.0300, 0.0217],
    'confidence': [0.6316, 0.7059, 0.7500, 0.6667, 0.5357, 0.5476, 0.5429, 0.5556, 0.6000, 0.5200],
    'lift': [2.8927, 2.5514, 2.5281, 2.3669, 2.0344, 1.9793, 1.9621, 1.9048, 1.8947, 1.8795]
}
df_task_d = pd.DataFrame(TASK_D_RULES)

# ============================================================================
# 3. SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.image("https://img.icons8.com/emoji/96/000000/hot-beverage-emoji.png", width=80)
st.sidebar.title("☕ Coffee & Books Cafe")
st.sidebar.markdown("### Professional Analytics Dashboard")
st.sidebar.markdown("---")

# Navigation with emojis
page = st.sidebar.radio(
    "📍 Navigate to:",
    [
        "🏠 Executive Summary",
        "📊 Market Insights (EDA)",
        "👥 Customer Personas",
        "📈 ML Model Results",
        "🔮 Live Prospect Simulator"
    ]
)

st.sidebar.markdown("---")

# Dataset info
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.metric("Total Responses", len(df))
st.sidebar.metric("Features", len(df.columns))

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the navigation menu to explore different sections of this analysis.")
st.sidebar.markdown("---")
st.sidebar.markdown("**📧 Contact:** [Your Friend's Email]")
st.sidebar.markdown("**👤 Author:** [Your Friend's Name]")
st.sidebar.markdown(f"**📅 Last Updated:** {datetime.now().strftime('%B %Y')}")

# ============================================================================
# 4. PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

if page == "🏠 Executive Summary":
    # Hero Section
    st.markdown("<h1 style='text-align: center;'>☕ Coffee & Books Cafe</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #8B6F47;'>Comprehensive Business Validation Dashboard</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Project Overview
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3>🎯 Project Overview</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
        This dashboard presents a comprehensive analysis of survey data collected to validate a new 
        <strong>Coffee & Books Cafe</strong> concept. Using advanced machine learning techniques, we've 
        identified key customer segments, spending drivers, and strategic product bundles to inform 
        business decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("### 📊 Key Business Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Champion Model Accuracy",
            value="77.5%",
            delta="K-Nearest Neighbors"
        )
    
    with col2:
        st.metric(
            label="Model Recall",
            value="98.9%",
            delta="Exceptional"
        )
    
    with col3:
        st.metric(
            label="Top Income Impact",
            value="+117 AED",
            delta="75k+ Income Bracket"
        )
    
    with col4:
        st.metric(
            label="Valuable Personas",
            value="4 Segments",
            delta="Identified via Clustering"
        )
    
    st.markdown("---")
    
    # Key Findings
    st.markdown("### 🎓 Executive Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
            <h4>🎯 Champion Classification Model</h4>
            <h2 style='color: #2E7D32;'>K-Nearest Neighbors</h2>
            <p><strong>F1-Score:</strong> 86.96%</p>
            <p style='font-size: 14px;'>Our champion model excels at identifying potential customers with 
            an exceptional <strong>98.9% Recall rate</strong>, ensuring we never miss a viable prospect.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800;'>
            <h4>💰 Primary Spending Driver</h4>
            <h2 style='color: #E65100;'>Customer Income Level</h2>
            <p><strong>Impact:</strong> +117.24 AED for 75k+ bracket</p>
            <p style='font-size: 14px;'>Regression analysis reveals that income is the most significant 
            predictor of customer spending, with high-income customers spending substantially more per visit.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
            <h4>👥 Premium Customer Persona</h4>
            <h2 style='color: #1565C0;'>Premium Reading Enthusiast</h2>
            <p><strong>Cluster 3:</strong> Most Valuable Segment</p>
            <p style='font-size: 14px;'>Our clustering analysis identified high-income, regular readers who 
            visit frequently as the most valuable segment. Average spend: <strong>169 AED</strong>, with 
            membership willingness of <strong>368 AED</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #FCE4EC; padding: 20px; border-radius: 10px; border-left: 5px solid #E91E63;'>
            <h4>🔗 Strategic Product Bundle</h4>
            <h2 style='color: #C2185B;'>The Business Professional</h2>
            <p><strong>Lift:</strong> 2.89x higher likelihood</p>
            <p style='font-size: 14px;'>Association rule mining revealed a powerful combination: 
            <strong>Business Books + Flavored Coffee + International Cuisine</strong>. Perfect for a 
            "Business Lunch" special package.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic Recommendations
    st.markdown("### 🚀 Strategic Recommendations")
    
    tab1, tab2, tab3 = st.tabs(["💼 Marketing Strategy", "🎁 Product Bundles", "👥 Target Segments"])
    
    with tab1:
        st.markdown("""
        #### Recommended Marketing Approach
        
        1. **Premium Membership Program (Target: Cluster 1 & 3)**
           - Price point: 250-350 AED/month (based on WTP)
           - Include exclusive book access and event invitations.
        
        2. **Per-Visit Premium Offers (Target: Cluster 2)**
           - This cluster has high income but low membership interest (8 AED).
           - Do not market memberships; focus on high-profit *per-visit* items (premium food, special coffee, social event tickets).
        
        3. **Value-Oriented Daily Specials (Target: Cluster 0)**
           - Focus on affordable combinations to increase visit frequency.
           - Drive foot traffic during off-peak hours.
        """)
    
    with tab2:
        st.markdown("""
        #### High-Performance Product Bundles
        
        1. **The Business Professional** (Lift: 2.89x)
           - Business/Self-Help Book Selection
           - Premium Flavored Coffee
           - International Cuisine Options
        
        2. **The "Me Time" Bundle** (Lift: 2.03x)
           - Fiction/Literary Book
           - Light Snacks (cookies, biscuits)
        
        3. **The Morning Ritual** (Lift: 1.96x)
           - Breakfast Items
           - Specialty Coffee
        """)
    
    with tab3:
        st.markdown("""
        #### Customer Segment Prioritization
        
        **Priority 1: Cluster 3 - Premium Reading Enthusiast** ⭐
        - HIGHEST spend (169 AED) & HIGHEST membership willingness (368 AED).
        - **Action:** Target with "Elite" memberships and VIP events.
        
        **Priority 2: Cluster 1 - Middle-Income Bookworm** 📚
        - Good spend (74 AED) but HIGH membership willingness (247 AED).
        - **Action:** Target with standard book club memberships. This is your core, loyal reader base.
        
        **Priority 3: Cluster 2 - Affluent Social Visitor** 💰
        - HIGH spend (142 AED) but ZERO membership interest (8 AED).
        - **Action:** DO NOT sell memberships. Focus on premium per-visit items and social events.
        
        **Priority 4: Cluster 0 - Budget-Conscious Casual** 💼
        - Low spend (47 AED) and low membership willingness (67 AED).
        - **Action:** Entry-level offers to increase visit frequency.
        """)
    
    st.markdown("---")
    
    # Dataset Preview
    st.markdown("### 📋 Survey Dataset Preview")
    st.markdown("Complete cleaned dataset used for all analyses:")
    
    # Display with nice formatting
    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete Dataset (CSV)",
        data=csv,
        file_name="cafe_survey_data.csv",
        mime="text/csv",
    )

# ============================================================================
# 5. PAGE 2: MARKET INSIGHTS (EDA)
# ============================================================================

elif page == "📊 Market Insights (EDA)":
    st.title("📊 Market Insights & Exploratory Analysis")
    st.markdown("Comprehensive visualization of survey responses and market validation metrics.")
    st.markdown("---")
    
    # Key Insights Banner
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**✅ Survey Responses:** {len(df)}")
    with col2:
        likely_visitors = len(df[df['Visit_Likelihood'].isin(['Definitely will visit', 'Probably will visit'])])
        st.success(f"**👥 Likely Visitors:** {likely_visitors} ({likely_visitors/len(df)*100:.1f}%)")
    with col3:
        avg_spend = df['Total_Spend_AED'].mean()
        st.warning(f"**💰 Avg. Expected Spend:** {avg_spend:.2f} AED")
    
    st.markdown("---")
    
    # Visualization Section
    st.markdown("### 📈 Key Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Visit Likelihood", "Spending Patterns", "Demographics & Preferences"])
    
    with tab1:
        st.markdown("#### Visit Likelihood Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Visit Likelihood by Income
            fig1 = px.histogram(
                df,
                x='Income',
                color='Visit_Likelihood',
                barmode='group',
                title='Visit Likelihood Distribution by Income Level',
                color_discrete_map={
                    'Definitely will visit': PRIMARY_COLOR,
                    'Probably will visit': SECONDARY_COLOR,
                    'Might visit': '#DEB887',
                    'Probably will not visit': '#D3D3D3',
                    'Definitely will not visit': '#A9A9A9'
                }
            )
            fig1.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Visit Likelihood Pie Chart
            likelihood_counts = df['Visit_Likelihood'].value_counts()
            fig2 = px.pie(
                values=likelihood_counts.values,
                names=likelihood_counts.index,
                title='Overall Visit Likelihood Distribution',
                color_discrete_sequence=px.colors.sequential.YlOrBr_r
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Visit Likelihood by Education
        fig3 = px.histogram(
            df,
            x='Education',
            color='Visit_Likelihood',
            barmode='stack',
            title='Visit Likelihood by Education Level',
            color_discrete_sequence=px.colors.sequential.YlOrBr_r
        )
        fig3.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab2:
        st.markdown("#### Spending Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Total Spend Distribution
            fig4 = px.histogram(
                df,
                x='Total_Spend_AED',
                nbins=30,
                title='Distribution of Expected Total Spend per Visit',
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig4.update_layout(
                xaxis_title="Total Spend (AED)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Willingness to Pay for Membership
            fig5 = px.histogram(
                df,
                x='Willing_Pay_Membership',
                nbins=30,
                title='Distribution of "Willingness to Pay" for Membership',
                color_discrete_sequence=[ACCENT_COLOR]
            )
            fig5.update_layout(
                xaxis_title="Membership Willingness (AED)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        # Spending by Income Level
        spend_by_income = df.groupby('Income').agg({
            'Total_Spend_AED': 'mean',
            'Avg_Spend_AED': 'mean',
            'Willing_Pay_Membership': 'mean'
        }).reset_index()
        
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=spend_by_income['Income'],
            y=spend_by_income['Total_Spend_AED'],
            name='Total Spend',
            marker_color=PRIMARY_COLOR
        ))
        fig6.add_trace(go.Bar(
            x=spend_by_income['Income'],
            y=spend_by_income['Avg_Spend_AED'],
            name='Average Spend',
            marker_color=SECONDARY_COLOR
        ))
        fig6.add_trace(go.Bar(
            x=spend_by_income['Income'],
            y=spend_by_income['Willing_Pay_Membership'],
            name='Membership Willingness',
            marker_color=ACCENT_COLOR
        ))
        fig6.update_layout(
            title='Spending Metrics by Income Level',
            xaxis_tickangle=-45,
            barmode='group',
            height=400,
            xaxis_title="Income Level",
            yaxis_title="Amount (AED)"
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.markdown("#### Demographics & Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age Distribution
            age_counts = df['Age_Group'].value_counts()
            fig7 = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title='Age Group Distribution',
                labels={'x': 'Age Group', 'y': 'Count'},
                color=age_counts.values,
                color_continuous_scale='YlOrBr'
            )
            fig7.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            # Employment Status
            employment_counts = df['Employment'].value_counts()
            fig9 = px.bar(
                x=employment_counts.index,
                y=employment_counts.values,
                title='Employment Status Distribution',
                labels={'x': 'Employment Status', 'y': 'Count'},
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig9.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig9, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reading Frequency
            reading_counts = df['Reading_Frequency'].value_counts()
            fig11 = px.bar(
                x=reading_counts.index,
                y=reading_counts.values,
                title='Reading Frequency Distribution',
                labels={'x': 'Reading Frequency', 'y': 'Count'},
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig11.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig11, use_container_width=True)
        
        with col2:
            # Cafe Visit Frequency
            cafe_counts = df['Cafe_Frequency'].value_counts()
            fig12 = px.bar(
                x=cafe_counts.index,
                y=cafe_counts.values,
                title='Cafe Visit Frequency Distribution',
                labels={'x': 'Cafe Visit Frequency', 'y': 'Count'},
                color_discrete_sequence=[SECONDARY_COLOR]
            )
            fig12.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig12, use_container_width=True)
        

# ============================================================================
# 6. PAGE 3: CUSTOMER PERSONAS
# ============================================================================

elif page == "👥 Customer Personas":
    st.title("👥 Customer Personas (K-Means Clustering)")
    st.markdown("Detailed analysis of customer segments identified through unsupervised machine learning.")
    st.markdown("---")
    
    # Overview
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3>🎯 Clustering Methodology</h3>
        <p style='font-size: 16px;'>
        Using <strong>K-Means clustering</strong>, we identified <strong>4 distinct customer personas</strong> 
        based on spending behavior, demographics, and preferences. Each cluster represents a unique market 
        segment with specific characteristics and value propositions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Numerical Persona Profiles
    st.markdown("### 💰 Persona Financial Profiles")
    
    # Create styled dataframe
    formatter_dict = {
        'Avg_Spend_AED': '{:.2f}',
        'Total_Spend_AED': '{:.2f}',
        'Willing_Pay_Membership': '{:.2f}'
    }
    styled_personas = (
        df_task_b_personas.style.format(formatter_dict)
        .background_gradient(cmap='YlOrBr', subset=['Avg_Spend_AED'])
        .background_gradient(cmap='YlOrBr', subset=['Total_Spend_AED'])
        .background_gradient(cmap='YlGn', subset=['Willing_Pay_Membership'])
        .set_properties(**{'text-align': 'center'})
    )
    
    st.dataframe(styled_personas, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Persona Cards
    st.markdown("### 🎭 Detailed Persona Profiles")
    
    # Create tabs for each persona
    tab0, tab1, tab2, tab3 = st.tabs([
        "Cluster 0: 💼 Budget-Conscious Casual",
        "Cluster 1: 📚 Middle-Income Bookworm",
        "Cluster 2: 💰 Affluent Social Visitor",
        "Cluster 3: ⭐ Premium Reading Enthusiast"
    ])
    
    with tab0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #F5F5F0; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>💼</h1>
                <h3>Cluster 0</h3>
                <h4>Budget-Conscious Casual</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", f"{df_task_b_personas.loc[0, 'Avg_Spend_AED']:.2f} AED")
            st.metric("Total Spend", f"{df_task_b_personas.loc[0, 'Total_Spend_AED']:.2f} AED")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[0, 'Willing_Pay_Membership']:.2f} AED")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Income']}
            - **Reading Frequency:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Reading Frequency']}
            - **Cafe Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Cafe Visits']}
            - **Profile:** Entry-level customers with limited disposable income.
            
            #### Strategic Recommendations
            - ✅ Focus on **daily specials** and **value combos**.
            - ✅ Target with **entry-level promotions** (e.g., "coffee + pastry for 25 AED").
            - ✅ Avoid expensive membership pitches.
            - ✅ Use as **volume driver** during off-peak hours.
            """)
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #F5F5F0; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>📚</h1>
                <h3>Cluster 1</h3>
                <h4>Middle-Income Bookworm</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", f"{df_task_b_personas.loc[1, 'Avg_Spend_AED']:.2f} AED")
            st.metric("Total Spend", f"{df_task_b_personas.loc[1, 'Total_Spend_AED']:.2f} AED")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[1, 'Willing_Pay_Membership']:.2f} AED", delta="High WTP!")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Income']}
            - **Reading Frequency:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Reading Frequency']}
            - **Cafe Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Cafe Visits']}
            - **Profile:** Book enthusiasts with moderate spending power but **HIGH membership interest**.
            
            #### Strategic Recommendations
            - ✅ This is your **core membership base**.
            - ✅ Offer **book club programs** and reading rewards.
            - ✅ Host **author meetups and literary events**.
            - ✅ Market a mid-tier membership (e.g., 249 AED/month) focused on book access.
            """)

    with tab2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #F5F5F0; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>💰</h1>
                <h3>Cluster 2</h3>
                <h4>Affluent Social Visitor</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", f"{df_task_b_personas.loc[2, 'Avg_Spend_AED']:.2f} AED")
            st.metric("Total Spend", f"{df_task_b_personas.loc[2, 'Total_Spend_AED']:.2f} AED", delta="High Spend")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[2, 'Willing_Pay_Membership']:.2f} AED", delta="Very Low WTP!", delta_color="inverse")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Income']}
            - **Reading Frequency:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Reading Frequency']}
            - **Cafe Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Cafe Visits']}
            - **Profile:** High earners who value social ambiance over reading.
            
            #### Strategic Recommendations
            - ✅ **DO NOT** market memberships to this group.
            - ✅ Emphasize **premium food & beverage quality**.
            - ✅ Focus on **ambiance, aesthetics, and social events**.
            - ✅ Upsell high-profit *per-visit* items.
            """)

    with tab3:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #D2B48C;'>
                <h1 style='font-size: 60px;'>⭐</h1>
                <h3>Cluster 3</h3>
                <h4>Premium Reading Enthusiast</h4>
                <p style='color: #6F4E37; font-weight: bold;'>★ HIGHEST VALUE SEGMENT ★</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", f"{df_task_b_personas.loc[3, 'Avg_Spend_AED']:.2f} AED", delta="HIGHEST", delta_color="normal")
            st.metric("Total Spend", f"{df_task_b_personas.loc[3, 'Total_Spend_AED']:.2f} AED", delta="HIGHEST", delta_color="normal")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[3, 'Willing_Pay_Membership']:.2f} AED", delta="HIGHEST", delta_color="normal")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Income']}
            - **Reading Frequency:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Reading Frequency']}
            - **Cafe Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Cafe Visits']}
            - **Profile:** The "Golden Customer". Affluent, passionate readers with high spend and high membership interest.
            
            #### Strategic Recommendations
            - ✅ **PRIMARY TARGET** for all premium offerings.
            - ✅ Create an **"Elite" membership tier** (e.g., 350+ AED) with exclusive perks.
            - ✅ Market **VIP events, author sessions, and private reading rooms**.
            - ✅ This group validates the entire "Coffee & Books" premium concept.
            """)

# ============================================================================
# 7. PAGE 4: ML MODEL RESULTS
# ============================================================================

elif page == "📈 ML Model Results":
    st.title("📈 Machine Learning Model Results")
    st.markdown("Comprehensive results from all four machine learning tasks.")
    st.markdown("---")
    
    # Task A: Classification
    with st.expander("🎯 TASK A: Classification Model Results", expanded=True):
        st.markdown("### Model Performance Comparison")
        st.markdown("**Objective:** Predict whether a customer will visit the cafe (`Visit_Likelihood`)")
        
        # Display results table with styling
        styled_task_a = df_task_a.style\
            .format({'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'})\
            .highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#D2B48C')\
            .set_properties(**{'text-align': 'center'})
        
        st.dataframe(styled_task_a, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison chart
            fig_models = px.bar(
                df_task_a,
                x='Model',
                y=['Accuracy', 'F1-Score'],
                title='Model Performance (Accuracy & F1-Score)',
                barmode='group',
                color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR]
            )
            fig_models.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_models, use_container_width=True)
        
        with col2:
            # Champion model highlight
            fig_champion = go.Figure()
            champion_metrics = df_task_a[df_task_a['Model'] == 'K-Nearest Neighbors'].iloc[0]
            
            fig_champion.add_trace(go.Scatterpolar(
                r=[champion_metrics['Accuracy'], champion_metrics['Precision'], 
                   champion_metrics['Recall'], champion_metrics['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                name='K-Nearest Neighbors',
                marker=dict(color=PRIMARY_COLOR)
            ))
            
            fig_champion.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Champion Model: K-Nearest Neighbors',
                height=400
            )
            st.plotly_chart(fig_champion, use_container_width=True)
        
        st.success("""
        **🏆 CHAMPION MODEL: K-Nearest Neighbors**
        - **F1-Score: 86.96%** - Best overall balance between precision and recall.
        - **Recall: 98.9%** - This is the key metric! The model is exceptional at identifying *all* potential visitors.
        - **Business Impact:** This model minimizes "False Negatives," ensuring our marketing almost *never* misses a potential customer.
        """)
    
    # Task C: Regression
    with st.expander("💰 TASK C: Regression Analysis - Spending Drivers", expanded=False):
        st.markdown("### Key Drivers of Customer Spending (`Total_Spend_AED`)")
        st.markdown("**Objective:** Identify factors that most significantly impact total spending.")
        
        # Display coefficients
        styled_task_c = df_task_c.style\
            .format({'Coefficient (AED)': '{:.2f}'})\
            .background_gradient(cmap='BrBG', subset=['Coefficient (AED)'])\
            .set_properties(**{'text-align': 'left'})
        
        st.dataframe(styled_task_c, use_container_width=True)
        
        # Visualization
        fig_task_c = px.bar(
            df_task_c.sort_values('Coefficient (AED)'),
            x='Coefficient (AED)',
            y='Feature',
            orientation='h',
            title='Price Drivers (Lasso Coefficients)',
            color='Coefficient (AED)',
            color_continuous_scale='BrBG'
        )
        st.plotly_chart(fig_task_c, use_container_width=True)
        
        st.warning("""
        **💡 KEY FINDING: Income is the Dominant Driver**
        - Customers earning **75k+ AED** spend an additional **117.24 AED** per visit.
        - Customers earning **50-75k AED** spend an additional **89.74 AED** per visit.
        - Lower income brackets (<10k AED) show significant negative coefficients.
        
        **Business Impact:** All premium marketing and membership tiers *must* be targeted at the 50k+ income segments.
        """)
    
    # Task D: Association Rules
    with st.expander("🔗 TASK D: Association Rules - Product Bundles", expanded=False):
        st.markdown("### Strategic Product Bundle Opportunities")
        st.markdown("**Objective:** Discover product combinations frequently purchased together.")
        
        # Display rules
        styled_task_d = df_task_d.style\
            .format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.4f}'})\
            .background_gradient(cmap='YlOrBr', subset=['lift'])\
            .set_properties(**{'text-align': 'left'})
        
        st.dataframe(styled_task_d, use_container_width=True)
        
        # Visualize top rules
        fig_lift = px.bar(
            df_task_d.head(10),
            x='lift',
            y=df_task_d.head(10)['antecedents'],
            orientation='h',
            title='Top 10 Rules by Lift (Surprise Factor)',
            color='lift',
            color_continuous_scale='YlOrBr',
            labels={'y': 'Rule Antecedent (IF...)'}
        )
        st.plotly_chart(fig_lift, use_container_width=True)
        
        st.info("""
        **🎯 TOP STRATEGIC BUNDLE: "The Business Professional"**
        - **Combination:** Business Books + Flavored Coffee + International Cuisine
        - **Lift: 2.89x** - Customers buying these together are nearly 3x more likely.
        - **Confidence: 63.16%** - A strong predictive relationship.
        
        **Business Impact:** This is a perfect "Business Lunch" special.
        """)

# ============================================================================
# 8. PAGE 5: LIVE PROSPECT SIMULATOR
# ============================================================================

elif page == "🔮 Live Prospect Simulator":
    st.title("🔮 Live Prospect Simulator")
    st.markdown("This is an interactive tool using our **Champion Model (K-Nearest Neighbors)** to predict the visit likelihood of a new prospect.")
    st.markdown("---")
    
    # Build and cache the model
    @st.cache_resource
    def build_prediction_pipeline():
        """Build and train the KNN classification pipeline."""
        try:
            # Define features and target
            TARGET_VARIABLE = "Visit_Likelihood"
            numerical_features = ['Avg_Spend_AED', 'Total_Spend_AED', 'Willing_Pay_Membership']
            categorical_features = [
                'Age_Group', 'Gender', 'Employment', 'Income', 'Education',
                'Cafe_Frequency', 'Reading_Frequency', 'Visit_Reason'
            ]
            FEATURES = numerical_features + categorical_features
            
            # Prepare data
            X = df[FEATURES]
            positive_maps = ['Definitely will visit', 'Probably will visit']
            y = df[TARGET_VARIABLE].map(lambda x: 1 if x in positive_maps else 0)
            
            # Build preprocessing pipeline
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Build and train model
            champion_model = KNeighborsClassifier()
            clf_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', champion_model)
            ])
            
            # Train on full dataset (for demo purposes)
            clf_pipeline.fit(X, y)
            
            return clf_pipeline, df, True
        except Exception as e:
            st.error(f"Error building model: {e}")
            return None, None, False
    
    # Load model with progress
    with st.spinner('Loading Champion Model...'):
        pipeline, df_reference, model_ready = build_prediction_pipeline()
    
    if not model_ready or pipeline is None:
        st.error("❌ Model could not be loaded. Please check the data and try again.")
        st.stop()
    
    st.success("✅ Champion Model (K-Nearest Neighbors) is trained and ready!")
    
    st.markdown("---")
    
    # Input Form
    st.markdown("### 📝 Enter Prospect Information")
    
    with st.form("prospect_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            age = st.selectbox("Age Group", options=sorted(df_reference['Age_Group'].unique()))
            gender = st.selectbox("Gender", options=sorted(df_reference['Gender'].unique()))
            employment = st.selectbox("Employment Status", options=sorted(df_reference['Employment'].unique()))
            education = st.selectbox("Education Level", options=sorted(df_reference['Education'].unique()))

        with col2:
            st.markdown("#### 📚 Behavior")
            income = st.selectbox("Income Level (AED)", options=sorted(df_reference['Income'].unique()))
            cafe_freq = st.selectbox("Cafe Visit Frequency", options=sorted(df_reference['Cafe_Frequency'].unique()))
            read_freq = st.selectbox("Reading Frequency", options=sorted(df_reference['Reading_Frequency'].unique()))
            visit_reason = st.selectbox("Primary Visit Reason", options=sorted(df_reference['Visit_Reason'].unique()))

        with col3:
            st.markdown("#### 💳 Spending Profile")
            avg_spend = st.slider("Average Spend (AED)", 0, 150, 50)
            total_spend = st.slider("Total Spend (AED)", 0, 300, 100)
            pay_membership = st.slider("Willing to Pay Membership (AED)", 0, 500, 50)
            
            # Submit button
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🔮 Predict Visit Likelihood",
                type="primary",
                use_container_width=True
            )
    
    # Process prediction
    if submitted:
        with st.spinner('Analyzing prospect profile...'):
            time.sleep(1) # Simulate processing
            
            # Create input dataframe
            input_data = pd.DataFrame({
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
            
            try:
                # Get prediction probability
                probability = pipeline.predict_proba(input_data)[0][1]
                
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                # Display probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Visit Likelihood Score", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': PRIMARY_COLOR},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': '#FFE5E5'},
                            {'range': [40, 70], 'color': '#FFF5E5'},
                            {'range': [70, 100], 'color': '#E5FFE5'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Detailed recommendation
                if probability > 0.7:
                    st.success("""
                    ### ✅ HIGH-VALUE PROSPECT
                    **Recommendation:** This is a high-priority prospect (likely Cluster 1 or 3).
                    **Suggested Actions:**
                    - ✅ Immediate follow-up with personalized offer.
                    - ✅ Offer premium membership package.
                    - ✅ Invite to exclusive events or book launch.
                    """)
                    st.balloons()
                
                elif probability > 0.4:
                    st.info("""
                    ### ⚠️ MEDIUM-POTENTIAL PROSPECT
                    **Recommendation:** This prospect needs nurturing (likely Cluster 0 or 2).
                    **Suggested Actions:**
                    - 📧 Add to email nurture campaign.
                    - 🎁 Offer "first visit discount" (15-20% off).
                    - 📚 Highlight specific features aligned with their interests.
                    """)
                
                else:
                    st.warning("""
                    ### ❌ LOW-PRIORITY PROSPECT
                    **Recommendation:** This prospect is unlikely to convert.
                    **Suggested Actions:**
                    - 📮 Add to general newsletter (low priority).
                    - ⏸️ Do not allocate marketing budget.
                    - 🔄 Re-evaluate if profile changes.
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.info("Please check your inputs and try again.")

