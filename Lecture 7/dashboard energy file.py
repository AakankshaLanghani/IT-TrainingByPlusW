import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "yearly_full_release_long_format.csv"
df = pd.read_csv(file_path)

# Filter for total energy consumption (Demand)
df_energy = df[df["Variable"] == "Demand"].copy()
df_energy = df_energy[['Area', 'Year', 'Value']].dropna()
df_energy.rename(columns={'Area': 'Country', 'Value': 'Energy Consumption (TWh)'}, inplace=True)

# Feature engineering
df_energy["Years Since Start"] = df_energy["Year"] - df_energy["Year"].min()

# Define features and target variable
X = df_energy[["Years Since Start"]]
y = df_energy["Energy Consumption (TWh)"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("Energy Consumption Prediction Dashboard")

# Display dataset info
st.write("## Dataset Overview")
st.write(df_energy.head())

# Visualization of Global Energy Consumption Trends
st.write("## Global Energy Consumption Trends")
fig, ax = plt.subplots()
sns.lineplot(data=df_energy.groupby("Year")["Energy Consumption (TWh)"].sum().reset_index(), x="Year", y="Energy Consumption (TWh)", marker="o", ax=ax)
plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (TWh)")
st.pyplot(fig)

# Model Evaluation
st.write("## Model Performance")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared Score: {r2:.2f}")

# Prediction Section
st.sidebar.header("Predict Energy Consumption")
years_since_start = st.sidebar.slider("Select Years Since Start:", int(X.min()), int(X.max()))
input_data = np.array([[years_since_start]])
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.sidebar.write(f"Predicted Energy Consumption: {prediction:.2f} TWh")
