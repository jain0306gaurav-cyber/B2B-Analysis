import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")

st.title("B2B Client Risk & Churn Prediction Dashboard")

# Load dataset
df = pd.read_csv("B2B_Client_Churn_5000.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# RISK SCORE CALCULATION
# -----------------------------

def calculate_risk(row):
    score = 0
    
    if row["Payment_Delay_Days"] > 30:
        score += 3
        
    if row["Monthly_Usage_Score"] < 40:
        score += 2
        
    if row["Contract_Length_Months"] <= 6:
        score += 2
        
    if row["Support_Tickets_Last30Days"] > 8:
        score += 3
        
    return score

df["Risk_Score"] = df.apply(calculate_risk, axis=1)

def risk_category(score):
    if score <= 2:
        return "Low Risk"
    elif score <= 5:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Level"] = df["Risk_Score"].apply(risk_category)

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filters")

region = st.sidebar.selectbox(
    "Region",
    ["All"] + list(df["Region"].unique())
)

industry = st.sidebar.selectbox(
    "Industry",
    ["All"] + list(df["Industry"].unique())
)

risk = st.sidebar.selectbox(
    "Risk Level",
    ["All"] + list(df["Risk_Level"].unique())
)

filtered_df = df.copy()

if region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == region]

if industry != "All":
    filtered_df = filtered_df[filtered_df["Industry"] == industry]

if risk != "All":
    filtered_df = filtered_df[filtered_df["Risk_Level"] == risk]

# -----------------------------
# KPI METRICS
# -----------------------------

total_clients = len(filtered_df)
high_risk = len(filtered_df[filtered_df["Risk_Level"] == "High Risk"])
avg_revenue = filtered_df["Monthly_Revenue_USD"].mean()
predicted_churn = (1 - filtered_df["Renewal_Status"].map({"Yes":1,"No":0}).mean()) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", high_risk)
col3.metric("Predicted Churn Rate %", round(predicted_churn,2))
col4.metric("Avg Revenue per Client", round(avg_revenue,2))

# -----------------------------
# RISK DISTRIBUTION
# -----------------------------

st.subheader("Risk Category Distribution")

risk_counts = filtered_df["Risk_Level"].value_counts()

fig = px.bar(
    x=risk_counts.index,
    y=risk_counts.values,
    labels={"x":"Risk Category","y":"Number of Clients"}
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INDUSTRY RISK ANALYSIS
# -----------------------------

st.subheader("Industry-wise Risk Analysis")

fig = px.histogram(
    filtered_df,
    x="Industry",
    color="Risk_Level"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# REVENUE VS RISK
# -----------------------------

st.subheader("Revenue vs Risk")

fig = px.scatter(
    filtered_df,
    x="Monthly_Revenue_USD",
    y="Risk_Score",
    color="Risk_Level"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CONTRACT LENGTH VS CHURN
# -----------------------------

st.subheader("Contract Length vs Churn")

fig = px.box(
    filtered_df,
    x="Renewal_Status",
    y="Contract_Length_Months"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# MACHINE LEARNING MODEL
# -----------------------------

st.header("Machine Learning Churn Prediction")

df_ml = df.copy()
df_ml["Renewal_Status"] = df_ml["Renewal_Status"].map({"Yes":1,"No":0})

features = [
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days",
    "Monthly_Revenue_USD"
]

X = df_ml[features]
y = df_ml["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")
st.write(round(accuracy,3))

# Confusion Matrix

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots()
ax.matshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, cm[i,j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature":features,
    "Importance":model.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance,
    x="Feature",
    y="Importance"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TOP HIGH RISK CLIENTS
# -----------------------------

st.subheader("Top 20 High Risk Clients")

high_risk_clients = df[df["Risk_Level"]=="High Risk"]\
    .sort_values("Risk_Score",ascending=False)\
    .head(20)

st.dataframe(high_risk_clients)

# -----------------------------
# RETENTION STRATEGY BUTTON
# -----------------------------

st.header("AI Based Retention Suggestions")

if st.button("Generate Retention Strategy"):

    st.success("Recommended Retention Strategies")

    st.write("• Offer payment discounts for clients with payment delay greater than 30 days")

    st.write("• Assign dedicated account managers for high revenue clients")

    st.write("• Provide incentives for long-term contract renewals")

    st.write("• Improve customer support response time for clients with frequent support tickets")

    st.write("• Conduct product training sessions to increase platform usage")

# -----------------------------
# RESPONSIBLE AI
# -----------------------------

st.header("Ethical Implications of Predicting Client Churn")

st.write("""
1. Predictive models may introduce bias if certain industries or regions are overrepresented in the dataset.

2. Labeling clients as 'High Risk' could negatively influence how account managers treat those clients.

3. Companies must ensure customer data privacy and comply with data protection regulations.

4. AI predictions should support human decision making rather than fully automate client relationship decisions.

5. Responsible AI requires transparency, fairness, and ethical use of customer data.
""")
