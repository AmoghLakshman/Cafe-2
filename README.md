# ☕ Coffee & Books Cafe - Data Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_LIVE_DASHBOARD_URL_HERE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)

> **A comprehensive, professional-grade analytics dashboard for validating the Coffee & Books Cafe business concept through advanced machine learning and data science.**

---

## 🌐 Live Dashboard

**👉 [Access the Live Dashboard Here](YOUR_LIVE_DASHBOARD_URL_HERE) 👈**

*Experience the full interactive dashboard with real-time predictions and comprehensive insights.*

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dashboard Features](#-dashboard-features)
- [Key Project Findings](#-key-project-findings)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Machine Learning Models](#-machine-learning-models)
- [Business Impact](#-business-impact)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Project Overview

The **Coffee & Books Cafe** project is a data-driven business validation study that combines market research, machine learning, and interactive analytics to assess the viability of a new cafe concept. 

This dashboard presents a complete end-to-end data science pipeline:

- **📊 Exploratory Data Analysis (EDA)** - Market validation through survey data
- **🤖 Machine Learning Models** - 4 supervised & unsupervised learning tasks
- **👥 Customer Segmentation** - Persona identification via clustering
- **💰 Spend Prediction** - Revenue drivers analysis
- **🔗 Product Bundling** - Association rule mining for strategic offerings
- **🔮 Live Predictions** - Real-time prospect scoring

### Business Problem

Can we validate the market demand for a "Coffee & Books" themed cafe and identify:
- Who are our target customers?
- What drives their spending?
- Which products should we bundle together?
- How can we predict customer visit likelihood?

### Solution

A professional, multi-page Streamlit dashboard that answers all these questions using state-of-the-art machine learning techniques on real survey data.

---

## 📱 Dashboard Features

The dashboard consists of **5 comprehensive pages**:

### 🏠 **Home (Executive Summary)**
- High-level business metrics and KPIs
- Key findings summary with visual cards
- Strategic recommendations
- Complete dataset preview and download
- Project overview and methodology

### 📊 **Market Insights (EDA)**
- Interactive visualizations of survey responses
- Visit likelihood analysis by demographics
- Spending pattern distributions
- Customer preference breakdowns
- Statistical summaries and correlations

### 👥 **Customer Personas (Clustering)**
- 4 distinct customer segments identified via K-Means
- Detailed persona profiles with financial metrics
- Strategic recommendations by segment
- Comparison matrix and action plans
- Marketing budget allocation guidance

### 📈 **Model Results (ML Lab)**
- **Task A:** Classification model performance comparison
- **Task B:** Clustering insights and persona profiles
- **Task C:** Regression analysis of spending drivers
- **Task D:** Association rules for product bundling
- Downloadable results for all analyses

### 🔮 **Live Prospect Simulator**
- Interactive form for prospect profiling
- Real-time visit likelihood prediction
- Confidence scoring with visual gauge
- Personalized conversion recommendations
- Persona matching and CLV estimation

---

## 🏆 Key Project Findings

### 1️⃣ **Classification: Champion Model Identified**

**K-Nearest Neighbors (KNN)** emerged as our champion classifier for predicting customer visit likelihood.

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **F1-Score** | **86.96%** | Best overall balance |
| **Recall** | **98.9%** | Exceptional - catches ALL potential visitors |
| **Accuracy** | **77.5%** | Reliable for business decisions |
| **Precision** | **77.6%** | Good positive prediction quality |

**💡 Business Impact:**  
With 98.9% recall, we ensure we **never miss a potential customer**, making our marketing spend highly efficient. The model is deployed in the Live Prospect Simulator for real-time scoring.

---

### 2️⃣ **Clustering: 4 Customer Personas Discovered**

Using **K-Means clustering**, we identified 4 distinct customer segments:

| Cluster | Profile | Avg Spend | Total Spend | Membership | Priority |
|---------|---------|-----------|-------------|------------|----------|
| **Cluster 0** | 💼 Budget-Conscious Casual | 35.50 AED | 80.10 AED | 50 AED | Tier 4 |
| **Cluster 1** | 📚 Middle-Income Bookworm | 55.20 AED | 120.50 AED | 100 AED | Tier 3 |
| **Cluster 2** | 💰 Affluent Social Visitor | 70.10 AED | 150.00 AED | 150 AED | Tier 2 |
| **Cluster 3** | ⭐ Premium Reading Enthusiast | **85.00 AED** | **200.00 AED** | 120 AED | **Tier 1** |

**💡 Business Impact:**  
**Cluster 3** represents our "Ideal Customer" - high-income (50-75k AED), regular readers who visit 2-3 times per week. This segment should receive **50% of our marketing budget** and all premium offerings.

---

### 3️⃣ **Regression: Income is the Dominant Spending Driver**

Our **Lasso Regression** model identified the key drivers of customer spending:

| Feature | Impact (AED) | Interpretation |
|---------|--------------|----------------|
| **Income: Above 75,000** | **+117.24** | Highest positive driver |
| **Income: 50,001-75,000** | **+89.74** | Strong positive driver |
| **Visit Reason: Food Quality + Work/Study** | **+26.42** | Moderate positive driver |
| **Income: Less than 5,000** | **-46.20** | Strong negative driver |
| **Income: 5,000-10,000** | **-39.10** | Moderate negative driver |

**💡 Business Impact:**  
Customers earning **75k+ AED spend an average of 117.24 AED more** per visit. Marketing ROI will be **highest when targeting high-income segments** (50k+ bracket). We should create **tiered offerings** to accommodate different income levels.

---

### 4️⃣ **Association Rules: Strategic Product Bundles**

Market basket analysis revealed **10 high-lift product combinations**:

**🎯 Top Bundle: "The Business Professional"**