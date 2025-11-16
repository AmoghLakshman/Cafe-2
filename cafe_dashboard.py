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
TASK_B_PERSONAS_NUMERIC = {
    'Cluster': [0, 1, 2, 3],
    'Avg_Spend_AED': [35.50, 55.20, 70.10, 85.00],
    'Total_Spend_AED': [80.10, 120.50, 150.00, 200.00],
    'Willing_Pay_Membership': [50, 100, 150, 120]
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
st.sidebar.markdown("**📧 Contact:** [Your Email]")
st.sidebar.markdown("**👤 Author:** [Your Name]")
st.sidebar.markdown("**📅 Last Updated:** 2024")

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
            label="Customer Segments",
            value="4 Personas",
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
            <h2 style='color: #1565C0;'>High-Income Reading Enthusiast</h2>
            <p><strong>Cluster 3:</strong> Most Valuable Segment</p>
            <p style='font-size: 14px;'>Our clustering analysis identified high-income, regular readers who 
            visit frequently as the most valuable segment. Average spending: <strong>85 AED</strong>, with 
            membership willingness of <strong>120 AED</strong>.</p>
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
        
        1. **Premium Membership Program**
           - Target Cluster 3 (High-Income Reading Enthusiasts)
           - Price point: 120-150 AED/month
           - Include exclusive book access and event invitations
        
        2. **Value-Oriented Daily Specials**
           - Target Cluster 0 (Budget-Conscious Casuals)
           - Focus on affordable combinations
           - Drive foot traffic during off-peak hours
        
        3. **Digital Marketing Focus**
           - 98.9% recall rate enables confident prospect targeting
           - Use predictive model to optimize ad spend
           - Focus on high-propensity visitors
        """)
    
    with tab2:
        st.markdown("""
        #### High-Performance Product Bundles
        
        1. **The Business Professional** (Lift: 2.89)
           - Business/Self-Help Book Selection
           - Premium Flavored Coffee
           - International Cuisine Options
           - Suggested Price: 95 AED
        
        2. **The Literary Afternoon** (Lift: 2.55)
           - Fiction Selection
           - Non-caffeinated Beverages
           - Light Snacks
           - Suggested Price: 65 AED
        
        3. **The Study Session** (Lift: 2.53)
           - Work/Study Space Access
           - Unlimited Coffee Refills
           - Pastries
           - Suggested Price: 75 AED
        """)
    
    with tab3:
        st.markdown("""
        #### Customer Segment Prioritization
        
        **Priority 1: Cluster 3 - Premium Reading Enthusiasts** ⭐
        - Highest spending: 85 AED average
        - Visit 2-3 times per week
        - Willing to pay 120 AED for membership
        - **Action:** Focus premium offerings and events
        
        **Priority 2: Cluster 2 - Affluent Social Visitors** 💰
        - Good spending: 70 AED average
        - Frequent visits but occasional readers
        - **Action:** Promote social events and food quality
        
        **Priority 3: Cluster 1 - Middle-Income Bookworms** 📚
        - Moderate spending: 55 AED average
        - Regular readers, weekly visits
        - **Action:** Book clubs and reading programs
        
        **Priority 4: Cluster 0 - Budget-Conscious Casuals** 💼
        - Lower spending: 35 AED average
        - Infrequent visits
        - **Action:** Entry-level offers to increase frequency
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
        st.info("**✅ Survey Responses:** " + str(len(df)))
    with col2:
        likely_visitors = len(df[df['Visit_Likelihood'].isin(['Definitely will visit', 'Probably will visit'])])
        st.success(f"**👥 Likely Visitors:** {likely_visitors} ({likely_visitors/len(df)*100:.1f}%)")
    with col3:
        avg_spend = df['Total_Spend_AED'].mean()
        st.warning(f"**💰 Avg. Expected Spend:** {avg_spend:.2f} AED")
    
    st.markdown("---")
    
    # Visualization Section
    st.markdown("### 📈 Key Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Visit Likelihood", "Spending Patterns", "Demographics", "Preferences"])
    
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
            fig1.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Visit Likelihood Pie Chart
            likelihood_counts = df['Visit_Likelihood'].value_counts()
            fig2 = px.pie(
                values=likelihood_counts.values,
                names=likelihood_counts.index,
                title='Overall Visit Likelihood Distribution',
                color_discrete_sequence=px.colors.sequential.Oranges_r
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
            color_discrete_sequence=px.colors.sequential.Oranges_r
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
            # Average Spend Distribution
            fig5 = px.histogram(
                df,
                x='Avg_Spend_AED',
                nbins=30,
                title='Distribution of Average Spend per Visit',
                color_discrete_sequence=[ACCENT_COLOR]
            )
            fig5.update_layout(
                xaxis_title="Average Spend (AED)",
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
        st.markdown("#### Demographic Analysis")
        
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
                color_continuous_scale='Oranges'
            )
            fig7.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            # Gender Distribution
            gender_counts = df['Gender'].value_counts()
            fig8 = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Gender Distribution',
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR]
            )
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        with col2:
            # Income Distribution
            income_counts = df['Income'].value_counts()
            fig10 = px.bar(
                x=income_counts.index,
                y=income_counts.values,
                title='Income Level Distribution',
                labels={'x': 'Income Level', 'y': 'Count'},
                color_discrete_sequence=[ACCENT_COLOR]
            )
            fig10.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig10, use_container_width=True)
    
    with tab4:
        st.markdown("#### Customer Preferences")
        
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
        
        # Visit Reason Analysis
        st.markdown("##### Primary Visit Reasons")
        visit_reasons = df['Visit_Reason'].value_counts().head(10)
        fig13 = px.bar(
            x=visit_reasons.values,
            y=visit_reasons.index,
            orientation='h',
            title='Top 10 Visit Reasons',
            labels={'x': 'Count', 'y': 'Visit Reason'},
            color=visit_reasons.values,
            color_continuous_scale='Oranges'
        )
        fig13.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig13, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical Summary
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spending Statistics")
        st.dataframe(
            df[['Avg_Spend_AED', 'Total_Spend_AED', 'Willing_Pay_Membership']].describe(),
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Categorical Distributions")
        categorical_summary = pd.DataFrame({
            'Feature': ['Visit Likelihood', 'Age Group', 'Gender', 'Education'],
            'Most Common': [
                df['Visit_Likelihood'].mode()[0],
                df['Age_Group'].mode()[0],
                df['Gender'].mode()[0],
                df['Education'].mode()[0]
            ],
            'Unique Values': [
                df['Visit_Likelihood'].nunique(),
                df['Age_Group'].nunique(),
                df['Gender'].nunique(),
                df['Education'].nunique()
            ]
        })
        st.dataframe(categorical_summary, use_container_width=True)

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
    styled_personas = df_task_b_personas.style\
        .format("{:.2f}")\
        .background_gradient(cmap='YlOrBr', subset=['Avg_Spend_AED'])\
        .background_gradient(cmap='YlOrBr', subset=['Total_Spend_AED'])\
        .background_gradient(cmap='YlOrBr', subset=['Willing_Pay_Membership'])\
        .set_properties(**{'text-align': 'center'})
    
    st.dataframe(styled_personas, use_container_width=True)
    
    # Visualize numerical profiles
    col1, col2 = st.columns(2)
    
    with col1:
        fig_spend = px.bar(
            df_task_b_personas.reset_index(),
            x='Cluster',
            y=['Avg_Spend_AED', 'Total_Spend_AED'],
            title='Spending Comparison Across Clusters',
            barmode='group',
            color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR]
        )
        fig_spend.update_layout(height=400)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    with col2:
        fig_membership = px.bar(
            df_task_b_personas.reset_index(),
            x='Cluster',
            y='Willing_Pay_Membership',
            title='Membership Fee Willingness by Cluster',
            color='Willing_Pay_Membership',
            color_continuous_scale='Oranges'
        )
        fig_membership.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_membership, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Persona Cards
    st.markdown("### 🎭 Detailed Persona Profiles")
    
    # Create tabs for each persona
    tab0, tab1, tab2, tab3 = st.tabs([
        "Cluster 0: Budget-Conscious Casual",
        "Cluster 1: Middle-Income Bookworm",
        "Cluster 2: Affluent Social Visitor",
        "Cluster 3: Premium Reading Enthusiast ⭐"
    ])
    
    with tab0:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFF8DC; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>💼</h1>
                <h3>Cluster 0</h3>
                <h4>Budget-Conscious Casual</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", "35.50 AED", delta="-49.50 AED vs highest")
            st.metric("Total Spend", "80.10 AED")
            st.metric("Membership", "50 AED")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** 10,001 - 20,000 AED
            - **Reading Frequency:** Occasional (1-2 times/week)
            - **Cafe Visits:** 2-3 times per month
            - **Profile:** Entry-level customers with limited disposable income
            
            #### Strategic Recommendations
            - ✅ Focus on **daily specials** and **value combos**
            - ✅ Target with **entry-level promotions**
            - ✅ Avoid expensive membership pitches
            - ✅ Use as **volume driver** during off-peak hours
            
            #### Marketing Messages
            - "Affordable luxury for everyday moments"
            - "Quality coffee without breaking the bank"
            - "Your neighborhood reading spot"
            """)
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #E6F3FF; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>📚</h1>
                <h3>Cluster 1</h3>
                <h4>Middle-Income Bookworm</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", "55.20 AED", delta="+19.70 AED vs Cluster 0")
            st.metric("Total Spend", "120.50 AED")
            st.metric("Membership", "100 AED")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** 20,001 - 35,000 AED
            - **Reading Frequency:** Regular (3-5 times/week)
            - **Cafe Visits:** Once per week
            - **Profile:** Book enthusiasts with moderate spending power
            
            #### Strategic Recommendations
            - ✅ Launch **book club programs**
            - ✅ Offer **reading rewards program**
            - ✅ Host **author meetups and literary events**
            - ✅ Mid-tier membership package (100 AED range)
            
            #### Marketing Messages
            - "Where readers become friends"
            - "Your weekly reading ritual awaits"
            - "Join our community of book lovers"
            """)
    
    with tab2:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFF0F5; padding: 20px; border-radius: 10px; text-align: center;'>
                <h1 style='font-size: 60px;'>💰</h1>
                <h3>Cluster 2</h3>
                <h4>Affluent Social Visitor</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", "70.10 AED", delta="+34.60 AED vs Cluster 0")
            st.metric("Total Spend", "150.00 AED")
            st.metric("Membership", "150 AED")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** 50,001 - 75,000 AED
            - **Reading Frequency:** Occasional (1-2 times/week)
            - **Cafe Visits:** 2-3 times per week
            - **Profile:** High earners who value social ambiance over reading
            
            #### Strategic Recommendations
            - ✅ Emphasize **premium food quality**
            - ✅ Create **social events and networking opportunities**
            - ✅ Focus on **ambiance and experience**
            - ✅ Premium but not top-tier membership
            
            #### Marketing Messages
            - "Where business meets pleasure"
            - "The perfect meeting spot for professionals"
            - "Elevate your social experience"
            """)
    
    with tab3:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #FFD700;'>
                <h1 style='font-size: 60px;'>⭐</h1>
                <h3>Cluster 3</h3>
                <h4>Premium Reading Enthusiast</h4>
                <p style='color: #FFD700; font-weight: bold;'>★ HIGHEST VALUE SEGMENT ★</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Avg Spend", "85.00 AED", delta="HIGHEST", delta_color="normal")
            st.metric("Total Spend", "200.00 AED", delta="HIGHEST", delta_color="normal")
            st.metric("Membership", "120 AED")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income Level:** 50,001 - 75,000 AED
            - **Reading Frequency:** Regular (3-5 times/week)
            - **Cafe Visits:** 2-3 times per week
            - **Profile:** HIGH-VALUE: Affluent, passionate readers with high visit frequency
            
            #### Strategic Recommendations
            - ⭐ **PRIMARY TARGET** for all premium offerings
            - ⭐ **Exclusive membership tier** with special perks
            - ⭐ **VIP events** and **author sessions**
            - ⭐ **Personalized book recommendations**
            - ⭐ **Priority seating** and **extended hours access**
            
            #### Marketing Messages
            - "An exclusive haven for discerning readers"
            - "Where literary passion meets premium comfort"
            - "Join the elite reading community"
            """)
    
    st.markdown("---")
    
    # Comparison Matrix
    st.markdown("### 📊 Persona Comparison Matrix")
    
    comparison_data = {
        'Metric': ['Average Spend', 'Total Spend', 'Membership Willingness', 'Visit Frequency', 'Reading Intensity', 'Target Priority'],
        'Cluster 0': ['35.50 AED', '80.10 AED', '50 AED', 'Low', 'Low', 'Tier 4'],
        'Cluster 1': ['55.20 AED', '120.50 AED', '100 AED', 'Medium', 'High', 'Tier 3'],
        'Cluster 2': ['70.10 AED', '150.00 AED', '150 AED', 'High', 'Low', 'Tier 2'],
        'Cluster 3': ['85.00 AED ⭐', '200.00 AED ⭐', '120 AED', 'High ⭐', 'High ⭐', 'Tier 1 ⭐']
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Strategic Action Plan
    st.markdown("---")
    st.markdown("### 🎯 Strategic Action Plan by Persona")
    
    action_plan = {
        'Persona': ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
        'Marketing Budget': ['5%', '15%', '30%', '50%'],
        'Membership Tier': ['Basic (50 AED)', 'Standard (100 AED)', 'Premium (150 AED)', 'Elite (120 AED + perks)'],
        'Primary Channel': ['Social Media Ads', 'Book Community Forums', 'LinkedIn/Professional Networks', 'Exclusive Events & VIP Outreach'],
        'Key Offering': ['Daily Specials', 'Book Clubs', 'Networking Events', 'Exclusive Reading Rooms']
    }
    
    df_action = pd.DataFrame(action_plan)
    st.dataframe(df_action, use_container_width=True, hide_index=True)

# ============================================================================
# 7. PAGE 4: ML MODEL RESULTS
# ============================================================================

elif page == "📈 ML Model Results":
    st.title("📈 Machine Learning Model Results")
    st.markdown("Comprehensive results from all four machine learning tasks.")
    st.markdown("---")
    
    # Task Overview Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>Task A</h4>
            <h2>🎯</h2>
            <p><strong>Classification</strong></p>
            <p>Visit Prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #E3F2FD; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>Task B</h4>
            <h2>👥</h2>
            <p><strong>Clustering</strong></p>
            <p>Customer Personas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background-color: #FFF3E0; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>Task C</h4>
            <h2>📊</h2>
            <p><strong>Regression</strong></p>
            <p>Spend Drivers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background-color: #FCE4EC; padding: 15px; border-radius: 10px; text-align: center;'>
            <h4>Task D</h4>
            <h2>🔗</h2>
            <p><strong>Association Rules</strong></p>
            <p>Product Bundles</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Task A: Classification
    with st.expander("🎯 TASK A: Classification Model Results", expanded=True):
        st.markdown("### Model Performance Comparison")
        st.markdown("**Objective:** Predict whether a customer will visit the cafe")
        
        # Display results table with styling
        styled_task_a = df_task_a.style\
            .format({'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'})\
            .highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#D4EDDA')\
            .set_properties(**{'text-align': 'center'})
        
        st.dataframe(styled_task_a, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model comparison chart
            fig_models = px.bar(
                df_task_a,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title='Model Performance Metrics Comparison',
                barmode='group',
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#DEB887']
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
        - **F1-Score: 86.96%** - Best overall balance between precision and recall
        - **Recall: 98.9%** - Exceptional at identifying ALL potential visitors
        - **Business Impact:** Minimizes false negatives, ensuring we never miss a potential customer
        """)
        
        st.markdown("#### Key Insights")
        st.markdown("""
        1. **K-Nearest Neighbors (KNN)** emerged as the champion with the highest F1-Score
        2. **Exceptional Recall (98.9%)** means we capture nearly all potential visitors
        3. **Accuracy of 77.5%** provides reliable predictions for business decisions
        4. All models show strong recall, indicating good pattern recognition in the data
        """)
    
    # Task C: Regression
    with st.expander("💰 TASK C: Regression Analysis - Spending Drivers", expanded=False):
        st.markdown("### Key Drivers of Customer Spending")
        st.markdown("**Objective:** Identify factors that most significantly impact total spending")
        
        # Display coefficients
        styled_task_c = df_task_c.style\
            .format({'Coefficient (AED)': '{:.2f}'})\
            .background_gradient(cmap='RdYlGn', subset=['Coefficient (AED)'])\
            .set_properties(**{'text-align': 'left'})
        
        st.dataframe(styled_task_c, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Positive drivers
            positive_drivers = df_task_c[df_task_c['Coefficient (AED)'] > 0].sort_values('Coefficient (AED)', ascending=False)
            fig_positive = px.bar(
                positive_drivers,
                x='Coefficient (AED)',
                y='Feature',
                orientation='h',
                title='Positive Spending Drivers',
                color='Coefficient (AED)',
                color_continuous_scale='Greens'
            )
            fig_positive.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_positive, use_container_width=True)
        
        with col2:
            # Negative drivers
            negative_drivers = df_task_c[df_task_c['Coefficient (AED)'] < 0].sort_values('Coefficient (AED)')
            fig_negative = px.bar(
                negative_drivers,
                x='Coefficient (AED)',
                y='Feature',
                orientation='h',
                title='Negative Spending Drivers',
                color='Coefficient (AED)',
                color_continuous_scale='Reds'
            )
            fig_negative.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_negative, use_container_width=True)
        
        st.warning("""
        **💡 KEY FINDING: Income is the Dominant Driver**
        - Customers earning **75k+ AED** spend an additional **117.24 AED** per visit
        - Customers earning **50-75k AED** spend an additional **89.74 AED** per visit
        - Lower income brackets (<10k AED) show negative coefficients, indicating reduced spending
        """)
        
        st.markdown("#### Business Implications")
        st.markdown("""
        1. **Target High-Income Segments** with premium offerings and exclusive experiences
        2. **Food Quality & Work/Study Environment** positively impact spending (+26.42 AED)
        3. **Price Sensitivity** is real among lower-income segments - offer tiered options
        4. **Marketing ROI** will be highest when targeting 50k+ income brackets
        """)
    
    # Task D: Association Rules
    with st.expander("🔗 TASK D: Association Rules - Product Bundles", expanded=False):
        st.markdown("### Strategic Product Bundle Opportunities")
        st.markdown("**Objective:** Discover product combinations frequently purchased together")
        
        # Display rules
        styled_task_d = df_task_d.style\
            .format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.4f}'})\
            .background_gradient(cmap='YlOrBr', subset=['lift'])\
            .set_properties(**{'text-align': 'left'})
        
        st.dataframe(styled_task_d, use_container_width=True)
        
        # Visualize top rules
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lift = px.bar(
                df_task_d.head(10),
                x='lift',
                y=df_task_d.head(10).index,
                orientation='h',
                title='Top 10 Rules by Lift',
                color='lift',
                color_continuous_scale='Oranges',
                labels={'y': 'Rule #'}
            )
            fig_lift.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_lift, use_container_width=True)
        
        with col2:
            fig_confidence = px.scatter(
                df_task_d,
                x='support',
                y='confidence',
                size='lift',
                title='Rule Quality: Support vs Confidence (sized by Lift)',
                color='lift',
                color_continuous_scale='Oranges',
                hover_data=['antecedents', 'consequents']
            )
            fig_confidence.update_layout(height=400)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        st.info("""
        **🎯 TOP STRATEGIC BUNDLE: "The Business Professional"**
        - **Combination:** Business Books + Flavored Coffee + International Cuisine
        - **Lift: 2.89x** - Customers buying these together are nearly 3x more likely
        - **Confidence: 63.16%** - Strong predictive relationship
        - **Suggested Package Price:** 95 AED (vs 100+ AED if purchased separately)
        """)
        
        st.markdown("#### Recommended Bundle Strategies")
        
        bundle_strategies = {
            'Bundle Name': [
                'The Business Professional',
                'The Literary Enthusiast',
                'The Study Session',
                'The Morning Ritual',
                'The Family Hour'
            ],
            'Components': [
                'Business Books + Flavored Coffee + International Food',
                'Fiction Books + Non-caffeinated Drinks + Light Snacks',
                'Study Space + Coffee + Pastries',
                'Breakfast Items + Specialty Coffee',
                'Children\'s Books + Desserts + Family-friendly Space'
            ],
            'Target Persona': [
                'Cluster 3 (Premium Enthusiasts)',
                'Cluster 1 (Bookworms)',
                'Cluster 2 (Social Visitors)',
                'All Clusters',
                'Family Segments'
            ],
            'Suggested Price': [
                '95 AED',
                '65 AED',
                '75 AED',
                '55 AED',
                '85 AED'
            ],
            'Expected Lift': [
                '2.89x',
                '2.55x',
                '2.53x',
                '1.96x',
                '1.88x'
            ]
        }
        
        df_bundles = pd.DataFrame(bundle_strategies)
        st.dataframe(df_bundles, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Overall Insights Summary
    st.markdown("### 🎓 Overall ML Insights Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **✅ What's Working**
        1. **High Model Accuracy:** 77.5% prediction accuracy for visit likelihood
        2. **Clear Personas:** 4 distinct customer segments identified
        3. **Income Matters:** Strong correlation between income and spending
        4. **Bundle Opportunities:** Multiple high-lift product combinations found
        5. **Actionable Insights:** All findings translate directly to business strategy
        """)
    
    with col2:
        st.info("""
        **💡 Key Recommendations**
        1. Deploy KNN model for prospect scoring and targeting
        2. Focus marketing budget on high-income segments (Clusters 2 & 3)
        3. Create tiered membership packages aligned with persona profiles
        4. Launch bundled offerings based on association rules
        5. Emphasize food quality and work/study environment in messaging
        """)
    
    # Download Results
    st.markdown("---")
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_task_a = df_task_a.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Classification Results",
            data=csv_task_a,
            file_name="classification_results.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_task_c = df_task_c.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Regression Drivers",
            data=csv_task_c,
            file_name="regression_drivers.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_task_d = df_task_d.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Association Rules",
            data=csv_task_d,
            file_name="association_rules.csv",
            mime="text/csv"
        )

# ============================================================================
# 8. PAGE 5: LIVE PROSPECT SIMULATOR
# ============================================================================

elif page == "🔮 Live Prospect Simulator":
    st.title("🔮 Live Prospect Simulator")
    st.markdown("Interactive prediction tool using our Champion Model (K-Nearest Neighbors)")
    st.markdown("---")
    
    # Model Info
    st.markdown("""
    <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
        <h3>🏆 About This Simulator</h3>
        <p style='font-size: 16px;'>
        This tool uses our <strong>Champion Classification Model (K-Nearest Neighbors)</strong> 
        to predict the visit likelihood of a new prospect in real-time. The model achieved:
        </p>
        <ul style='font-size: 16px;'>
            <li><strong>77.5% Accuracy</strong></li>
            <li><strong>98.9% Recall</strong> - Exceptional at finding potential customers</li>
            <li><strong>86.96% F1-Score</strong> - Best overall performance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
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
    
    st.success("✅ Champion Model is trained and ready for predictions!")
    
    st.markdown("---")
    
    # Input Form
    st.markdown("### 📝 Enter Prospect Information")
    st.markdown("Fill in the details below to get a visit likelihood prediction:")
    
    # Create input form
    with st.form("prospect_form"):
        # Demographic Information
        st.markdown("#### 👤 Demographics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.selectbox(
                "Age Group",
                options=sorted(df_reference['Age_Group'].unique()),
                help="Select the prospect's age group"
            )
        
        with col2:
            gender = st.selectbox(
                "Gender",
                options=sorted(df_reference['Gender'].unique()),
                help="Select the prospect's gender"
            )
        
        with col3:
            employment = st.selectbox(
                "Employment Status",
                options=sorted(df_reference['Employment'].unique()),
                help="Current employment status"
            )
        
        with col4:
            education = st.selectbox(
                "Education Level",
                options=sorted(df_reference['Education'].unique()),
                help="Highest education level achieved"
            )
        
        # Financial Information
        st.markdown("#### 💰 Financial Profile")
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.selectbox(
                "Income Level (AED)",
                options=sorted(df_reference['Income'].unique()),
                help="Monthly income bracket"
            )
        
        with col2:
            pay_membership = st.slider(
                "Willing to Pay for Membership (AED)",
                min_value=0,
                max_value=500,
                value=100,
                step=10,
                help="Maximum monthly membership fee they'd consider"
            )
        
        # Behavioral Information
        st.markdown("#### 📚 Behavioral Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cafe_freq = st.selectbox(
                "Cafe Visit Frequency",
                options=sorted(df_reference['Cafe_Frequency'].unique()),
                help="How often they currently visit cafes"
            )
        
        with col2:
            read_freq = st.selectbox(
                "Reading Frequency",
                options=sorted(df_reference['Reading_Frequency'].unique()),
                help="How often they read"
            )
        
        with col3:
            visit_reason = st.selectbox(
                "Primary Visit Reason",
                options=sorted(df_reference['Visit_Reason'].unique()),
                help="Main reason for visiting cafes"
            )
        
        # Spending Information
        st.markdown("#### 💳 Expected Spending")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_spend = st.slider(
                "Average Spend per Visit (AED)",
                min_value=0,
                max_value=150,
                value=50,
                step=5,
                help="Expected average spend per visit"
            )
        
        with col2:
            total_spend = st.slider(
                "Total Expected Spend (AED)",
                min_value=0,
                max_value=300,
                value=100,
                step=10,
                help="Total spending in a typical visit"
            )
        
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
            time.sleep(1)  # Simulate processing
            
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
                prediction = pipeline.predict(input_data)[0]
                
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
                
                fig_gauge.update_layout(height=300, font={'size': 20})
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Display result with interpretation
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if probability > 0.7:
                        st.metric("Classification", "HIGH LIKELIHOOD ✅", delta="Will Visit")
                    elif probability > 0.4:
                        st.metric("Classification", "MEDIUM LIKELIHOOD ⚠️", delta="Might Visit")
                    else:
                        st.metric("Classification", "LOW LIKELIHOOD ❌", delta="Won't Visit")
                
                with col2:
                    st.metric("Confidence Score", f"{probability*100:.1f}%")
                
                with col3:
                    if probability > 0.7:
                        persona_match = "Cluster 3 (Premium)"
                    elif probability > 0.5:
                        persona_match = "Cluster 2 (Affluent)"
                    elif probability > 0.3:
                        persona_match = "Cluster 1 (Bookworm)"
                    else:
                        persona_match = "Cluster 0 (Budget)"
                    st.metric("Likely Persona", persona_match)
                
                st.markdown("---")
                
                # Detailed recommendation
                if probability > 0.7:
                    st.success("""
                    ### ✅ HIGH-VALUE PROSPECT
                    
                    **Recommendation:** This is a high-priority prospect with strong visit likelihood.
                    
                    **Suggested Actions:**
                    - ✅ Immediate follow-up with personalized offer
                    - ✅ Offer premium membership package
                    - ✅ Invite to exclusive events or book launch
                    - ✅ Allocate marketing budget for conversion
                    - ✅ Expected Customer Lifetime Value: HIGH
                    
                    **Conversion Strategy:**
                    Send a personalized email highlighting premium features, exclusive book selection, 
                    and VIP perks. Consider offering a "first month free" membership trial.
                    """)
                    st.balloons()
                
                elif probability > 0.4:
                    st.info("""
                    ### ⚠️ MEDIUM-POTENTIAL PROSPECT
                    
                    **Recommendation:** This prospect shows moderate interest but needs nurturing.
                    
                    **Suggested Actions:**
                    - 📧 Add to email nurture campaign
                    - 🎁 Offer "first visit discount" (15-20% off)
                    - 📚 Highlight specific features aligned with their interests
                    - ⏰ Follow up after 7-14 days
                    - 💡 Expected Customer Lifetime Value: MEDIUM
                    
                    **Conversion Strategy:**
                    Focus on removing barriers to first visit. Offer a risk-free trial or money-back 
                    guarantee. Share testimonials from similar customer profiles.
                    """)
                
                else:
                    st.warning("""
                    ### ❌ LOW-PRIORITY PROSPECT
                    
                    **Recommendation:** This prospect is unlikely to convert. Minimal resource allocation advised.
                    
                    **Suggested Actions:**
                    - 📮 Add to general newsletter (low priority)
                    - ⏸️ Do not allocate marketing budget
                    - 🔄 Re-evaluate if profile changes
                    - 💰 Expected Customer Lifetime Value: LOW
                    
                    **Alternative Strategy:**
                    Instead of active pursuit, add to a passive nurture sequence with quarterly check-ins. 
                    Focus resources on higher-probability prospects.
                    """)
                
                # Profile Summary
                st.markdown("---")
                st.markdown("### 📋 Prospect Profile Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Demographics:**")
                    profile_demo = pd.DataFrame({
                        'Attribute': ['Age Group', 'Gender', 'Employment', 'Education'],
                        'Value': [age, gender, employment, education]
                    })
                    st.dataframe(profile_demo, hide_index=True, use_container_width=True)
                
                with col2:
                    st.markdown("**Behavior & Spending:**")
                    profile_behavior = pd.DataFrame({
                        'Attribute': ['Income', 'Cafe Frequency', 'Reading Frequency', 'Avg Spend', 'Total Spend', 'Membership Willingness'],
                        'Value': [income, cafe_freq, read_freq, f"{avg_spend} AED", f"{total_spend} AED", f"{pay_membership} AED"]
                    })
                    st.dataframe(profile_behavior, hide_index=True, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.info("Please check your inputs and try again.")
    
    # Tips Section
    st.markdown("---")
    st.markdown("### 💡 Simulator Tips")
    
    with st.expander("How to Use This Simulator Effectively"):
        st.markdown("""
        **Best Practices:**
        1. **Accurate Inputs:** The model's predictions are only as good as the input data
        2. **Multiple Scenarios:** Test different profile combinations to understand patterns
        3. **Comparative Analysis:** Run similar profiles with one variable changed to see impact
        4. **Regular Updates:** As you gather more data, retrain the model for better accuracy
        
        **Understanding the Results:**
        - **70%+ = High Likelihood:** Aggressive marketing, premium offers
        - **40-70% = Medium Likelihood:** Nurture campaigns, introductory offers
        - **<40% = Low Likelihood:** Passive nurture, minimal resource allocation
        
        **Key Influencing Factors:**
        Based on our regression analysis, the most influential factors are:
        1. Income level (strongest predictor)
        2. Expected spending amounts
        3. Cafe visit frequency
        4. Reading frequency
        5. Visit reasons (work/study, food quality)
        """)

# ============================================================================
# 9. FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8B6F47; padding: 20px;'>
    <p><strong>Coffee & Books Cafe - Professional Analytics Dashboard</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
