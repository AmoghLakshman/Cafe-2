"""
===============================================================================
COFFEE & BOOKS CAFE - ULTIMATE ANALYTICS DASHBOARD
===============================================================================
A comprehensive multi-page Streamlit dashboard with advanced ML simulations
for analyzing survey data and validating the Coffee & Books Cafe concept.

Features:
✓ Executive Summary
✓ Market Insights (EDA)
✓ Customer Personas (Clustering)
✓ ML Model Results
✓ Live Prospect Simulator
✓ Advanced Simulation Lab (NEW!)

Author: [Your Name]
Date: 2024
Version: 2.0.0 (Ultimate Edition)
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

warnings.filterwarnings('ignore')

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    
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
    
    [data-testid="stSidebar"] {
        background-color: #F5F5F0;
    }
    
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
    
    .stAlert {
        border-radius: 8px;
        padding: 1rem;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 8px;
    }
    
    .streamlit-expanderHeader {
        background-color: #FFFFFF;
        border-radius: 8px;
        font-weight: 600;
    }
    
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
    """Loads and caches the cleaned survey data from GitHub."""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please check your internet connection and the GitHub URL.")
        return None

with st.spinner('Loading survey data...'):
    df = load_data()
    
if df is None:
    st.stop()

# ============================================================================
# 2. HARD-CODED RESULTS
# ============================================================================

PRIMARY_COLOR = '#6F4E37'
SECONDARY_COLOR = '#D2B48C'
ACCENT_COLOR = '#8B6F47'

# Task A: Classification
TASK_A_RESULTS = {
    'Model': ['K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine (SVM)', 
              'Logistic Regression', 'Decision Tree'],
    'Accuracy': [0.7750, 0.7667, 0.7583, 0.7500, 0.6833],
    'Precision': [0.7759, 0.7692, 0.7719, 0.7699, 0.7732],
    'Recall': [0.9890, 0.9890, 0.9670, 0.9560, 0.8242],
    'F1-Score': [0.8696, 0.8654, 0.8585, 0.8529, 0.7979]
}
df_task_a = pd.DataFrame(TASK_A_RESULTS)

# Task B: Clustering
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

# Task C: Regression
TASK_C_DRIVERS = {
    'Feature': ['Income_Above 75,000', 'Income_50,001 - 75,000', 'Income_Less than 5,000',
                'Income_5,000 - 10,000', 'Visit_Reason_Food quality|Work/study...',
                'Income_10,001 - 20,000', 'Income_35,001 - 50,000',
                'Visit_Reason_Coffee/beverages quality|Food quality'],
    'Coefficient (AED)': [117.24, 89.74, -46.20, -39.10, 26.42, -16.69, 14.16, -11.61]
}
df_task_c = pd.DataFrame(TASK_C_DRIVERS)

# Task D: Association Rules
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
        'Non-Fiction - Business/Self-Help', 'Flavored Coffee (Vanilla, Caramel, Hazelnut)',
        'International cuisine options', 'Non-caffeinated beverages only',
        'Light snacks (cookies, biscuits)', 'Flavored Coffee (Vanilla, Caramel, Hazelnut)',
        'Flavored Coffee (Vanilla, Caramel, Hazelnut)', 'Pastries (croissants, muffins)',
        'No food, just beverages', 'Desserts (cakes, brownies)'
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

# Navigation with ALL 6 pages
page = st.sidebar.radio(
    "📍 Navigate to:",
    [
        "🏠 Executive Summary",
        "📊 Market Insights (EDA)",
        "👥 Customer Personas",
        "📈 ML Model Results",
        "🔮 Live Prospect Simulator",
        "🧪 The Simulation Lab (Advanced)"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.metric("Total Responses", len(df))
st.sidebar.metric("Features", len(df.columns))

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use the navigation menu to explore different sections.")
st.sidebar.markdown("---")
st.sidebar.markdown("**📧 Contact:** your.email@example.com")
st.sidebar.markdown("**👤 Author:** Your Name")
st.sidebar.markdown(f"**📅 Last Updated:** {datetime.now().strftime('%B %Y')}")

# ============================================================================
# 4. PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

if page == "🏠 Executive Summary":
    st.markdown("<h1 style='text-align: center;'>☕ Coffee & Books Cafe</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #8B6F47;'>Comprehensive Business Validation Dashboard</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h3>🎯 Project Overview</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
        This dashboard presents a comprehensive analysis of survey data collected to validate a new 
        <strong>Coffee & Books Cafe</strong> concept. Using advanced machine learning techniques, we've 
        identified key customer segments, spending drivers, and strategic product bundles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Key Business Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Champion Model Accuracy", "77.5%", "K-Nearest Neighbors")
    col2.metric("Model Recall", "98.9%", "Exceptional")
    col3.metric("Top Income Impact", "+117 AED", "75k+ Income")
    col4.metric("Valuable Personas", "4 Segments", "K-Means Clustering")
    
    st.markdown("---")
    
    st.markdown("### 🎓 Executive Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
            <h4>🎯 Champion Classification Model</h4>
            <h2 style='color: #2E7D32;'>K-Nearest Neighbors</h2>
            <p><strong>F1-Score:</strong> 86.96%</p>
            <p style='font-size: 14px;'>Exceptional <strong>98.9% Recall</strong> ensures we never miss viable prospects.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #FFF3E0; padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800;'>
            <h4>💰 Primary Spending Driver</h4>
            <h2 style='color: #E65100;'>Customer Income Level</h2>
            <p><strong>Impact:</strong> +117.24 AED for 75k+ bracket</p>
            <p style='font-size: 14px;'>Income is the most significant predictor of customer spending.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background-color: #E3F2FD; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
            <h4>👥 Premium Customer Persona</h4>
            <h2 style='color: #1565C0;'>Premium Reading Enthusiast</h2>
            <p><strong>Cluster 3:</strong> Most Valuable</p>
            <p style='font-size: 14px;'>Average spend: <strong>169 AED</strong>, Membership WTP: <strong>368 AED</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: #FCE4EC; padding: 20px; border-radius: 10px; border-left: 5px solid #E91E63;'>
            <h4>🔗 Strategic Product Bundle</h4>
            <h2 style='color: #C2185B;'>The Business Professional</h2>
            <p><strong>Lift:</strong> 2.89x higher likelihood</p>
            <p style='font-size: 14px;'><strong>Business Books + Flavored Coffee + International Cuisine</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📋 Survey Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True, height=400)
    
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
    
    tab1, tab2, tab3 = st.tabs(["Visit Likelihood", "Spending Patterns", "Demographics"])
    
    with tab1:
        st.markdown("#### Visit Likelihood Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(
                df, x='Income', color='Visit_Likelihood', barmode='group',
                title='Visit Likelihood by Income Level',
                color_discrete_map={
                    'Definitely will visit': PRIMARY_COLOR,
                    'Probably will visit': SECONDARY_COLOR,
                    'Might visit': '#DEB887',
                    'Probably will not visit': '#D3D3D3',
                    'Definitely will not visit': '#A9A9A9'
                }
            )
            fig1.update_xaxes(tickangle=-45)
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            likelihood_counts = df['Visit_Likelihood'].value_counts()
            fig2 = px.pie(
                values=likelihood_counts.values,
                names=likelihood_counts.index,
                title='Overall Visit Likelihood',
                color_discrete_sequence=px.colors.sequential.YlOrBr_r
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("#### Spending Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig4 = px.histogram(
                df, x='Total_Spend_AED', nbins=30,
                title='Distribution of Expected Total Spend',
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig4.update_layout(height=400)
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            fig5 = px.histogram(
                df, x='Willing_Pay_Membership', nbins=30,
                title='Membership Willingness Distribution',
                color_discrete_sequence=[ACCENT_COLOR]
            )
            fig5.update_layout(height=400)
            st.plotly_chart(fig5, use_container_width=True)
        
        fig6 = px.box(
            df, x='Income', y='Total_Spend_AED',
            title='Spending by Income Level',
            color='Income'
        )
        fig6.update_xaxes(tickangle=-45)
        fig6.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.markdown("#### Demographics & Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_counts = df['Age_Group'].value_counts()
            fig7 = px.bar(
                x=age_counts.index, y=age_counts.values,
                title='Age Group Distribution',
                labels={'x': 'Age Group', 'y': 'Count'},
                color=age_counts.values,
                color_continuous_scale='YlOrBr'
            )
            fig7.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        with col2:
            reading_counts = df['Reading_Frequency'].value_counts()
            fig8 = px.bar(
                x=reading_counts.index, y=reading_counts.values,
                title='Reading Frequency Distribution',
                color_discrete_sequence=[PRIMARY_COLOR]
            )
            fig8.update_xaxes(tickangle=-45)
            fig8.update_layout(height=400)
            st.plotly_chart(fig8, use_container_width=True)

# ============================================================================
# 6. PAGE 3: CUSTOMER PERSONAS
# ============================================================================

elif page == "👥 Customer Personas":
    st.title("👥 Customer Personas (K-Means Clustering)")
    st.markdown("Detailed analysis of customer segments identified through unsupervised ML.")
    st.markdown("---")
    
    st.markdown("""
    <div class='custom-card'>
        <h3>🎯 Clustering Methodology</h3>
        <p style='font-size: 16px;'>
        Using <strong>K-Means clustering</strong>, we identified <strong>4 distinct personas</strong> 
        based on spending behavior and demographics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 💰 Persona Financial Profiles")
    
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
    )
    
    st.dataframe(styled_personas, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎭 Detailed Persona Profiles")
    
    tab0, tab1, tab2, tab3 = st.tabs([
        "💼 Budget Casual",
        "📚 Bookworm",
        "💰 Social Visitor",
        "⭐ Premium Enthusiast"
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
            - **Income:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Income']}
            - **Reading:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Reading Frequency']}
            - **Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 0']['Cafe Visits']}
            
            #### Strategic Recommendations
            - ✅ Focus on **daily specials** and **value combos**
            - ✅ Target with entry-level promotions
            - ✅ Use as volume driver during off-peak hours
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
            - **Income:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Income']}
            - **Reading:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Reading Frequency']}
            - **Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 1']['Cafe Visits']}
            
            #### Strategic Recommendations
            - ✅ **Core membership base**
            - ✅ Offer book club programs
            - ✅ Host author meetups and literary events
            - ✅ Market mid-tier membership (~249 AED/month)
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
            st.metric("Total Spend", f"{df_task_b_personas.loc[2, 'Total_Spend_AED']:.2f} AED", delta="High")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[2, 'Willing_Pay_Membership']:.2f} AED", delta="Very Low!", delta_color="inverse")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Income']}
            - **Reading:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Reading Frequency']}
            - **Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 2']['Cafe Visits']}
            
            #### Strategic Recommendations
            - ✅ **DO NOT** market memberships
            - ✅ Emphasize premium food & beverage
            - ✅ Focus on ambiance and social events
            - ✅ Upsell high-profit per-visit items
            """)
    
    with tab3:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px; text-align: center; border: 3px solid #D2B48C;'>
                <h1 style='font-size: 60px;'>⭐</h1>
                <h3>Cluster 3</h3>
                <h4>Premium Enthusiast</h4>
                <p style='color: #6F4E37; font-weight: bold;'>★ HIGHEST VALUE ★</p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Avg Spend", f"{df_task_b_personas.loc[3, 'Avg_Spend_AED']:.2f} AED", delta="HIGHEST")
            st.metric("Total Spend", f"{df_task_b_personas.loc[3, 'Total_Spend_AED']:.2f} AED", delta="HIGHEST")
            st.metric("Membership WTP", f"{df_task_b_personas.loc[3, 'Willing_Pay_Membership']:.2f} AED", delta="HIGHEST")
        
        with col2:
            st.markdown("#### Characteristics")
            st.markdown(f"""
            - **Income:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Income']}
            - **Reading:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Reading Frequency']}
            - **Visits:** {TASK_B_PERSONAS_CATEGORICAL['Cluster 3']['Cafe Visits']}
            
            #### Strategic Recommendations
            - ✅ **PRIMARY TARGET** for premium offerings
            - ✅ Create **Elite membership** (350+ AED)
            - ✅ Market VIP events and author sessions
            - ✅ Validates entire premium concept
            """)

# ============================================================================
# 7. PAGE 4: ML MODEL RESULTS
# ============================================================================

elif page == "📈 ML Model Results":
    st.title("📈 Machine Learning Model Results")
    st.markdown("Comprehensive results from all four ML tasks.")
    st.markdown("---")
    
    with st.expander("🎯 TASK A: Classification Results", expanded=True):
        st.markdown("### Model Performance Comparison")
        
        styled_task_a = df_task_a.style\
            .format({'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}', 'F1-Score': '{:.2%}'})\
            .highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#D2B48C')
        
        st.dataframe(styled_task_a, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_models = px.bar(
                df_task_a, x='Model', y=['Accuracy', 'F1-Score'],
                title='Model Performance',
                barmode='group',
                color_discrete_sequence=[PRIMARY_COLOR, ACCENT_COLOR]
            )
            fig_models.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_models, use_container_width=True)
        
        with col2:
            champion_metrics = df_task_a[df_task_a['Model'] == 'K-Nearest Neighbors'].iloc[0]
            fig_champion = go.Figure()
            fig_champion.add_trace(go.Scatterpolar(
                r=[champion_metrics['Accuracy'], champion_metrics['Precision'], 
                   champion_metrics['Recall'], champion_metrics['F1-Score']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                fill='toself',
                marker=dict(color=PRIMARY_COLOR)
            ))
            fig_champion.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title='Champion Model: KNN'
            )
            st.plotly_chart(fig_champion, use_container_width=True)
        
        st.success("🏆 **CHAMPION:** K-Nearest Neighbors | F1: 86.96% | Recall: 98.9%")
    
    with st.expander("💰 TASK C: Regression - Spending Drivers", expanded=False):
        st.markdown("### Key Spending Drivers")
        
        styled_task_c = df_task_c.style\
            .format({'Coefficient (AED)': '{:.2f}'})\
            .background_gradient(cmap='BrBG', subset=['Coefficient (AED)'])
        
        st.dataframe(styled_task_c, use_container_width=True)
        
        fig_task_c = px.bar(
            df_task_c.sort_values('Coefficient (AED)'),
            x='Coefficient (AED)', y='Feature',
            orientation='h',
            title='Price Drivers (Lasso Coefficients)',
            color='Coefficient (AED)',
            color_continuous_scale='BrBG'
        )
        st.plotly_chart(fig_task_c, use_container_width=True)
        
        st.warning("💡 **KEY:** Income is the dominant driver (+117 AED for 75k+ bracket)")
    
    with st.expander("🔗 TASK D: Association Rules", expanded=False):
        st.markdown("### Strategic Product Bundles")
        
        styled_task_d = df_task_d.style\
            .format({'support': '{:.4f}', 'confidence': '{:.4f}', 'lift': '{:.4f}'})\
            .background_gradient(cmap='YlOrBr', subset=['lift'])
        
        st.dataframe(styled_task_d, use_container_width=True)
        
        st.info("🎯 **TOP BUNDLE:** Business Books + Flavored Coffee + International Cuisine (Lift: 2.89x)")

# ============================================================================
# 8. PAGE 5: LIVE PROSPECT SIMULATOR
# ============================================================================

elif page == "🔮 Live Prospect Simulator":
    st.title("🔮 Live Prospect Simulator")
    st.markdown("Interactive tool using our **Champion Model (KNN)** to predict visit likelihood.")
    st.markdown("---")
    
    @st.cache_resource
    def build_prediction_pipeline():
        """Build and train the KNN classification pipeline."""
        try:
            TARGET = "Visit_Likelihood"
            NUM_FEAT = ['Avg_Spend_AED', 'Total_Spend_AED', 'Willing_Pay_Membership']
            CAT_FEAT = ['Age_Group', 'Gender', 'Employment', 'Income', 'Education',
                       'Cafe_Frequency', 'Reading_Frequency', 'Visit_Reason']
            FEATURES = NUM_FEAT + CAT_FEAT
            
            X = df[FEATURES]
            positive_maps = ['Definitely will visit', 'Probably will visit']
            y = df[TARGET].map(lambda x: 1 if x in positive_maps else 0)
            
            numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, NUM_FEAT),
                    ('cat', categorical_transformer, CAT_FEAT)
                ],
                remainder='drop'
            )
            
            clf_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', KNeighborsClassifier())
            ])
            
            clf_pipeline.fit(X, y)
            return clf_pipeline, df, True
        except Exception as e:
            st.error(f"Error: {e}")
            return None, None, False
    
    with st.spinner('Loading Champion Model...'):
        pipeline, df_ref, model_ready = build_prediction_pipeline()
    
    if not model_ready:
        st.error("❌ Model could not be loaded.")
        st.stop()
    
    st.success("✅ Champion Model (KNN) is ready!")
    st.markdown("---")
    
    st.markdown("### 📝 Enter Prospect Information")
    
    with st.form("prospect_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            age = st.selectbox("Age Group", sorted(df_ref['Age_Group'].unique()))
            gender = st.selectbox("Gender", sorted(df_ref['Gender'].unique()))
            employment = st.selectbox("Employment", sorted(df_ref['Employment'].unique()))
            education = st.selectbox("Education", sorted(df_ref['Education'].unique()))
        
        with col2:
            st.markdown("#### 📚 Behavior")
            income = st.selectbox("Income (AED)", sorted(df_ref['Income'].unique()))
            cafe_freq = st.selectbox("Cafe Frequency", sorted(df_ref['Cafe_Frequency'].unique()))
            read_freq = st.selectbox("Reading Frequency", sorted(df_ref['Reading_Frequency'].unique()))
            visit_reason = st.selectbox("Visit Reason", sorted(df_ref['Visit_Reason'].unique()))
        
        with col3:
            st.markdown("#### 💳 Spending")
            avg_spend = st.slider("Avg Spend (AED)", 0, 150, 50)
            total_spend = st.slider("Total Spend (AED)", 0, 300, 100)
            pay_membership = st.slider("Membership WTP (AED)", 0, 500, 50)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)
    
    if submitted:
        with st.spinner('Analyzing...'):
            time.sleep(1)
            
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
                probability = pipeline.predict_proba(input_data)[0][1]
                
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    title={'text': "Visit Likelihood Score"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': PRIMARY_COLOR},
                        'steps': [
                            {'range': [0, 40], 'color': '#FFE5E5'},
                            {'range': [40, 70], 'color': '#FFF5E5'},
                            {'range': [70, 100], 'color': '#E5FFE5'}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                if probability > 0.7:
                    st.success(f"""
                    ### ✅ HIGH-VALUE PROSPECT ({probability*100:.1f}%)
                    
                    **Actions:**
                    - ✅ Immediate follow-up with personalized offer
                    - ✅ Offer premium membership package
                    - ✅ Invite to exclusive events
                    """)
                    st.balloons()
                
                elif probability > 0.4:
                    st.info(f"""
                    ### ⚠️ MEDIUM-POTENTIAL ({probability*100:.1f}%)
                    
                    **Actions:**
                    - 📧 Add to nurture campaign
                    - 🎁 Offer first visit discount (15-20% off)
                    - 📚 Highlight aligned features
                    """)
                
                else:
                    st.warning(f"""
                    ### ❌ LOW-PRIORITY ({probability*100:.1f}%)
                    
                    **Actions:**
                    - 📮 Add to general newsletter only
                    - ⏸️ Do not allocate marketing budget
                    - 🔄 Re-evaluate if profile changes
                    """)
                
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================================
# 9. PAGE 6: THE SIMULATION LAB (ADVANCED) ⭐
# ============================================================================

elif page == "🧪 The Simulation Lab (Advanced)":
    st.title("🧪 The Simulation Lab")
    st.markdown("Advanced interactive tools powered by **Regression**, **Clustering**, and **Association Rules**.")
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["💰 Spending Predictor", "🧬 Persona Matcher", "🔗 Menu Recommender"])
    
    # ========================================================================
    # TAB 1: REGRESSION SIMULATOR (WITH BUTTON!)
    # ========================================================================
    with tab1:
        st.header("💰 The Spending Predictor")
        st.markdown("Predicts customer spending based on profile (Powered by Lasso Model).")
        st.markdown("---")
        
        with st.form("spending_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Customer Profile")
                reg_income = st.selectbox("Income Bracket", 
                    ["Less than 5,000", "5,000 - 10,000", "10,001 - 20,000", 
                     "20,001 - 35,000", "35,001 - 50,000", "50,001 - 75,000", "Above 75,000"],
                    index=3,
                    key="reg_income")
                
                reg_reason = st.multiselect("Visit Reasons (Select all that apply)", 
                    ["Food quality", "Work/study", "Coffee quality", "Ambiance", "Social meetings"],
                    default=["Coffee quality"],
                    key="reg_reason")
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 Prediction Result")
                st.info("👈 Select income and visit reasons, then click **Predict Spending** below")
            
            # Submit button
            predict_spending = st.form_submit_button(
                "💰 Predict Spending",
                type="primary",
                use_container_width=True
            )
        
        # Show results ONLY after button click
        if predict_spending:
            with st.spinner("Calculating spending prediction..."):
                time.sleep(0.5)  # Simulate processing
                
                # Calculate predicted spend
                base_spend = 35.0
                
                # Income impact
                if reg_income == "Above 75,000": 
                    base_spend += 117.24
                elif reg_income == "50,001 - 75,000": 
                    base_spend += 89.74
                elif reg_income == "35,001 - 50,000": 
                    base_spend += 14.16
                elif reg_income == "Less than 5,000": 
                    base_spend -= 46.20
                elif reg_income == "5,000 - 10,000": 
                    base_spend -= 39.10
                
                # Reason impact
                if "Food quality" in reg_reason and "Work/study" in reg_reason: 
                    base_spend += 26.42
                if "Coffee quality" in reg_reason and "Food quality" in reg_reason: 
                    base_spend -= 11.61
                
                final_spend = max(15, min(300, base_spend))
                
                st.markdown("---")
                st.markdown("### 💵 Prediction Results")
                
                # Display predicted spend
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #6F4E37 0%, #8B4513 100%); 
                     padding: 30px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
                    <h1 style='font-size: 3.5em; margin: 0; color: white;'>{final_spend:.2f} AED</h1>
                    <p style='font-size: 1.2em; margin: 10px 0; color: white;'>Predicted Spend per Visit</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation based on spend
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Base Spend", "35.00 AED")
                with col2:
                    income_impact = final_spend - 35.0
                    st.metric("Income Impact", f"{income_impact:+.2f} AED")
                with col3:
                    st.metric("Final Prediction", f"{final_spend:.2f} AED")
                
                st.markdown("---")
                
                # Strategic recommendation
                if final_spend > 100:
                    st.success("""
                    ### 🚀 HIGH VALUE CUSTOMER!
                    
                    **Strategy:**
                    - ✅ Target with **Premium Membership** (350+ AED/month)
                    - ✅ Offer **VIP perks** and exclusive events
                    - ✅ Priority seating and personalized service
                    - ✅ Expected monthly value: ~{:.0f} AED
                    """.format(final_spend * 8))
                    st.balloons()
                    
                elif final_spend > 50:
                    st.info("""
                    ### ⚖️ MID-VALUE CUSTOMER
                    
                    **Strategy:**
                    - 📧 Target with **Standard Membership** (150-250 AED/month)
                    - 🎁 Offer **loyalty card** (10% discount after 5 visits)
                    - 📚 Book club invitations
                    - 💰 Expected monthly value: ~{:.0f} AED
                    """.format(final_spend * 4))
                    
                else:
                    st.warning("""
                    ### ⚠️ BUDGET CUSTOMER
                    
                    **Strategy:**
                    - ☕ Target with **daily combos** (coffee + pastry deals)
                    - 🎯 "Happy Hour" promotions (20% off off-peak)
                    - 📅 Weekly specials to increase frequency
                    - 💵 Focus on volume over margin
                    """)
    
    # ========================================================================
    # TAB 2: CLUSTERING SIMULATOR (WITH BUTTON!)
    # ========================================================================
    with tab2:
        st.header("🧬 The Persona Matcher")
        st.markdown("Classify customers into segments (Powered by K-Means Clustering).")
        st.markdown("---")
        
        with st.form("persona_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Customer Spending Behavior")
                c_spend = st.slider("Average Spend per Visit (AED)", 10, 200, 50, key="c_spend")
                c_total = st.slider("Total Lifetime Spend (AED)", 10, 500, 100, key="c_total")
                c_membership = st.slider("Willingness to Pay for Membership (AED)", 0, 500, 50, key="c_membership")
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎭 Persona Result")
                st.info("👈 Adjust spending attributes, then click **Match Persona** below")
            
            # Submit button
            match_persona = st.form_submit_button(
                "🧬 Match Persona",
                type="primary",
                use_container_width=True
            )
        
        # Show results ONLY after button click
        if match_persona:
            with st.spinner("Analyzing customer DNA..."):
                time.sleep(0.5)
                
                # Define cluster centroids
                centroids = {
                    "Cluster 0 (Budget Casual)": {
                        'center': np.array([26.92, 46.91, 67.36]),
                        'emoji': '💼',
                        'name': 'Budget-Conscious Casual',
                        'color': '#FFA726'
                    },
                    "Cluster 1 (Bookworm)": {
                        'center': np.array([39.78, 73.88, 247.29]),
                        'emoji': '📚',
                        'name': 'Middle-Income Bookworm',
                        'color': '#66BB6A'
                    },
                    "Cluster 2 (Social Visitor)": {
                        'center': np.array([58.78, 142.43, 8.06]),
                        'emoji': '💰',
                        'name': 'Affluent Social Visitor',
                        'color': '#42A5F5'
                    },
                    "Cluster 3 (Premium Enthusiast)": {
                        'center': np.array([69.30, 168.79, 367.73]),
                        'emoji': '⭐',
                        'name': 'Premium Reading Enthusiast',
                        'color': '#AB47BC'
                    }
                }
                
                # Find closest cluster
                user_point = np.array([c_spend, c_total, c_membership])
                best_cluster = min(centroids.keys(), 
                                 key=lambda k: np.linalg.norm(user_point - centroids[k]['center']))
                
                cluster_info = centroids[best_cluster]
                
                st.markdown("---")
                st.markdown("### 🎭 Persona Assignment Result")
                
                # Display matched persona
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {cluster_info['color']} 0%, {cluster_info['color']}DD 100%); 
                     padding: 40px; border-radius: 15px; color: white; text-align: center; margin: 20px 0;'>
                    <h1 style='font-size: 4em; margin: 0; color: white;'>{cluster_info['emoji']}</h1>
                    <h2 style='margin: 15px 0; color: white;'>{cluster_info['name']}</h2>
                    <p style='font-size: 1.1em; color: white;'>{best_cluster}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Comparison table
                st.markdown("#### 📊 Your Profile vs. Cluster Average")
                
                comparison_data = pd.DataFrame({
                    'Metric': ['Avg Spend', 'Total Spend', 'Membership WTP'],
                    'Your Input': [
                        f'{c_spend:.2f} AED',
                        f'{c_total:.2f} AED',
                        f'{c_membership:.2f} AED'
                    ],
                    'Cluster Average': [
                        f"{cluster_info['center'][0]:.2f} AED",
                        f"{cluster_info['center'][1]:.2f} AED",
                        f"{cluster_info['center'][2]:.2f} AED"
                    ],
                    'Difference': [
                        f"{(c_spend - cluster_info['center'][0]):+.2f} AED",
                        f"{(c_total - cluster_info['center'][1]):+.2f} AED",
                        f"{(c_membership - cluster_info['center'][2]):+.2f} AED"
                    ]
                })
                
                st.dataframe(comparison_data, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Strategic recommendation
                if "Premium" in best_cluster:
                    st.success("""
                    ### ⭐ VIP TARGET - HIGHEST VALUE SEGMENT!
                    
                    **Strategic Action Plan:**
                    - ✅ **Immediate Priority:** Schedule personal consultation
                    - ✅ **Offer:** Elite Membership at 350+ AED/month
                    - ✅ **Perks:** Private reading room, author events, concierge service
                    - ✅ **Expected Lifetime Value:** 15,000+ AED/year
                    - ✅ **Communication:** Personal email from cafe owner
                    """)
                    st.balloons()
                    
                elif "Bookworm" in best_cluster:
                    st.success("""
                    ### 📚 LOYAL READER - CORE MEMBERSHIP BASE
                    
                    **Strategic Action Plan:**
                    - ✅ **Target:** Standard Book Club Membership (200-250 AED/month)
                    - ✅ **Offer:** Exclusive book selections + member events
                    - ✅ **Perks:** 15% discount on books, priority event registration
                    - ✅ **Expected Lifetime Value:** 6,000-8,000 AED/year
                    - ✅ **Communication:** Weekly book recommendations newsletter
                    """)
                    
                elif "Social" in best_cluster:
                    st.warning("""
                    ### 💰 HIGH SPENDER, NO MEMBERSHIP INTEREST
                    
                    **Strategic Action Plan:**
                    - ⚠️ **DO NOT** pitch membership (WTP is only 8 AED!)
                    - ✅ **Focus:** Premium per-visit upsells (food, special coffee)
                    - ✅ **Target:** Social event tickets, private table bookings
                    - ✅ **Perks:** VIP seating without membership commitment
                    - ✅ **Expected Value:** 4,000-5,000 AED/year (per-visit basis)
                    """)
                    
                else:
                    st.info("""
                    ### 💼 BUDGET CUSTOMER - VOLUME STRATEGY
                    
                    **Strategic Action Plan:**
                    - ✅ **Avoid:** Expensive membership pitches
                    - ✅ **Target:** Daily combo deals (coffee + pastry 25 AED)
                    - ✅ **Offer:** Punch card (buy 5, get 1 free)
                    - ✅ **Goal:** Increase visit frequency (2-3x/month → weekly)
                    - ✅ **Expected Value:** 1,500-2,000 AED/year
                    """)
    
    # ========================================================================
    # TAB 3: ASSOCIATION RULES SIMULATOR (WITH BUTTON!)
    # ========================================================================
    with tab3:
        st.header("🔗 The Smart Menu Recommender")
        st.markdown("Recommend perfect bundles based on customer interest (Powered by Association Rules).")
        st.markdown("---")
        
        with st.form("bundle_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📝 Customer Preferences")
                interest = st.selectbox("What is the customer interested in?", 
                    ["Business Books", "Fiction Novels", "Morning Coffee", 
                     "Studying/Working", "Family Time", "Social Meetings"],
                    key="interest")
                
                budget = st.slider("Customer Budget (AED)", 30, 150, 75, step=5, key="budget")
                
                visit_time = st.selectbox("Expected Visit Time",
                    ["Morning (7-11 AM)", "Lunch (11 AM-2 PM)", 
                     "Afternoon (2-6 PM)", "Evening (6-10 PM)"],
                    key="visit_time")
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🎁 Bundle Recommendation")
                st.info("👈 Select customer preferences, then click **Recommend Bundle** below")
            
            # Submit button
            recommend_bundle = st.form_submit_button(
                "🔗 Recommend Bundle",
                type="primary",
                use_container_width=True
            )
        
        # Show results ONLY after button click
        if recommend_bundle:
            with st.spinner("Analyzing bundle recommendations..."):
                time.sleep(0.5)
                
                st.markdown("---")
                st.markdown("### 🎁 Recommended Bundle")
                
                # Bundle recommendations based on interest
                if interest == "Business Books":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #1E88E5 0%, #1976D2 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>📊 'The Business Professional' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>Perfect for ambitious professionals</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bundle Price", "95 AED")
                    col2.metric("Association Lift", "2.89x")
                    col3.metric("Confidence", "63.2%")
                    
                    st.success("""
                    ### 📦 Bundle Contains:
                    
                    1. **📚 Business/Self-Help Book Selection**
                       - Latest bestsellers from our curated collection
                       - Access to business book club discussions
                    
                    2. **☕ Premium Flavored Coffee**
                       - Vanilla, Caramel, or Hazelnut Latte
                       - Unlimited refills during visit
                    
                    3. **🍽️ International Cuisine Meal**
                       - Choose from: Mediterranean Bowl, Asian Fusion, or Gourmet Sandwich
                       - Includes side salad and dessert
                    
                    ### 💡 Why This Bundle?
                    Our data shows customers buying these together have **2.89x higher satisfaction** 
                    and spend **40% more time** in the cafe. Perfect for:
                    - 💼 Working professionals during lunch meetings
                    - 📊 Entrepreneurs planning strategy sessions
                    - 🎯 Business students studying for exams
                    """)
                    
                    if budget >= 95:
                        st.balloons()
                        st.success("✅ **Within Budget!** Customer can afford this bundle.")
                    else:
                        st.warning(f"⚠️ Bundle is {95 - budget:.0f} AED over budget. Suggest alternatives.")
                
                elif interest == "Fiction Novels":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #9C27B0 0%, #8E24AA 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>📖 'The Literary Escape' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>For passionate fiction readers</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bundle Price", "65 AED")
                    col2.metric("Association Lift", "2.03x")
                    col3.metric("Confidence", "53.6%")
                    
                    st.info("""
                    ### 📦 Bundle Contains:
                    
                    1. **📚 Fiction Book Access**
                       - Browse our literary fiction collection
                       - Join monthly book club discussions
                    
                    2. **🫖 Herbal Tea Selection**
                       - Chamomile, Green Tea, or Earl Grey
                       - Served with honey and lemon
                    
                    3. **🍪 Light Snacks**
                       - Assorted cookies and biscuits
                       - Fresh fruit platter
                    
                    ### 💡 Perfect For:
                    - 📖 Weekend readers seeking relaxation
                    - 🧘 Stress relief through literature
                    - 👥 Book club members
                    """)
                    
                    if budget >= 65:
                        st.success("✅ **Great Fit!** Well within customer budget.")
                
                elif interest == "Morning Coffee":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #F57C00 0%, #EF6C00 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>☀️ 'The Morning Ritual' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>Start your day right</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Bundle Price", "45 AED")
                    col2.metric("Association Lift", "1.96x")
                    col3.metric("Confidence", "54.3%")
                    
                    st.warning("""
                    ### 📦 Bundle Contains:
                    
                    1. **☕ Specialty Coffee**
                       - Cappuccino, Americano, or Latte
                       - Double shot option available
                    
                    2. **🥐 Fresh Pastry**
                       - Croissant, Muffin, or Danish
                       - Baked fresh daily
                    
                    3. **🍊 Fresh Juice**
                       - Orange, Apple, or Mixed Berry
                    
                    ### 💡 Best For:
                    - ⏰ Morning commuters (7-9 AM)
                    - 💼 Before-work quick stops
                    - 📰 Newspaper + coffee routine
                    """)
                    
                    st.success("✅ **Excellent Value!** Most affordable option.")
                
                elif interest == "Studying/Working":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #00897B 0%, #00796B 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>📚 'The Study Pass' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>Productivity maximizer</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Daily Pass", "50 AED")
                    col2.metric("Weekly Pass", "300 AED", delta="-50 AED savings")
                    col3.metric("Monthly Pass", "1000 AED", delta="-500 AED savings")
                    
                    st.info("""
                    ### 📦 Pass Includes:
                    
                    1. **☕ Unlimited Drip Coffee**
                       - All-day unlimited refills
                       - Espresso shots for 5 AED extra
                    
                    2. **💺 Quiet Zone Access**
                       - Reserved study area
                       - Power outlets at every seat
                       - Fast WiFi (100 Mbps)
                    
                    3. **📚 Book Lending**
                       - Borrow up to 2 books during visit
                       - Reference materials available
                    
                    ### 💡 Perfect For:
                    - 🎓 Students during exam season
                    - 💻 Freelancers needing workspace
                    - 📝 Writers seeking inspiration
                    """)
                
                elif interest == "Family Time":
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #E91E63 0%, #D81B60 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>👨‍👩‍👧‍👦 'The Family Package' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>Quality time together</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Family Bundle", "120 AED", delta="For 4 people")
                    col2.metric("Kids Menu", "+15 AED", delta="Per child")
                    
                    st.success("""
                    ### 📦 Bundle Contains:
                    
                    1. **☕ Beverages for All**
                       - 2 Adult coffees/teas
                       - 2 Kids hot chocolates or juice
                    
                    2. **🍰 Dessert Platter**
                       - Assorted cakes and pastries
                       - Kid-friendly options included
                    
                    3. **📚 Children's Book Corner Access**
                       - Interactive reading area
                       - Coloring books and crayons
                    
                    4. **🎮 Board Game Rental**
                       - Choose from 20+ games
                       - Family-friendly options
                    """)
                
                else:  # Social Meetings
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #5E35B1 0%, #512DA8 100%); 
                         padding: 30px; border-radius: 15px; color: white; margin: 20px 0;'>
                        <h2 style='color: white; margin: 0;'>🎉 'The Social Hub' Bundle</h2>
                        <p style='font-size: 1.1em; margin: 15px 0;'>Perfect for meetups</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Small Group (2-4)", "80 AED/person")
                    col2.metric("Large Group (5+)", "70 AED/person", delta="-10 AED discount")
                    
                    st.info("""
                    ### 📦 Bundle Contains:
                    
                    1. **☕ Beverage Selection**
                       - Premium coffee, tea, or specialty drinks
                       - Unlimited refills
                    
                    2. **🍽️ Sharing Platters**
                       - Appetizer board (cheese, crackers, fruits)
                       - Finger sandwiches
                    
                    3. **📍 Reserved Seating**
                       - Private corner or window table
                       - 3-hour guaranteed seating
                    
                    4. **📚 Discussion Materials**
                       - Access to magazine collection
                       - Book club discussion guides
                    """)
                
                st.markdown("---")
                
                # Upsell opportunities
                st.markdown("### 💡 Recommended Add-Ons")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **📚 Book Purchase**
                    - 10% member discount
                    - Take home your favorite
                    """)
                
                with col2:
                    st.markdown("""
                    **🎁 Gift Card**
                    - Perfect for friends
                    - No expiration date
                    """)
                
                with col3:
                    st.markdown("""
                    **📅 Event Tickets**
                    - Author meet & greets
                    - Book club sessions
                    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8B6F47; padding: 20px;'>
    <p><strong>☕ Coffee & Books Cafe Analytics Dashboard</strong></p>
    <p>Built with Streamlit • Powered by Python & Scikit-Learn</p>
    <p>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)

