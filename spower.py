import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("solar_maintenance_data.csv")
    return df

df = load_data()

# Train model if not already trained
@st.cache_resource
def train_model():
    X = df[["Temperature_C", "Voltage_V", "Current_A", "Efficiency_%", "Dust_Level_%"]]
    y = df["Maintenance_Needed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

model, model_accuracy = train_model()

# Streamlit UI
st.title("Predictive Maintenance for Solar Power Systems")

# Display dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Show dataset statistics
if st.checkbox("Show Data Statistics"):
    st.write(df.describe())

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Histogram of efficiency
st.subheader("Efficiency Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Efficiency_%"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Scatter plot of temperature vs efficiency
st.subheader("Temperature vs Efficiency")
fig, ax = plt.subplots()
sns.scatterplot(x=df["Temperature_C"], y=df["Efficiency_%"], hue=df["Maintenance_Needed"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# User input for prediction
st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("Temperature (°C)", 20, 60, 40)
voltage = st.sidebar.slider("Voltage (V)", 30, 50, 40)
current = st.sidebar.slider("Current (A)", 5, 15, 10)
efficiency = st.sidebar.slider("Efficiency (%)", 70, 95, 85)
dust_level = st.sidebar.slider("Dust Level (%)", 0, 100, 50)

# Predict maintenance need
input_data = np.array([[temp, voltage, current, efficiency, dust_level]])
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error(f"⚠️ Maintenance Required! (Confidence: {prediction_proba[0][1]:.2f})")
else:
    st.success(f"✅ No Maintenance Needed (Confidence: {prediction_proba[0][0]:.2f})")

# Display model accuracy
st.subheader("Model Accuracy")
st.info(f"🔍 The model accuracy is: {model_accuracy:.2f}")

# Feature importance visualization
st.subheader("Feature Importance")
feature_importance = model.feature_importances_
feature_names = ["Temperature_C", "Voltage_V", "Current_A", "Efficiency_%", "Dust_Level_%"]
fig, ax = plt.subplots()
sns.barplot(x=feature_importance, y=feature_names, ax=ax)
st.pyplot(fig)