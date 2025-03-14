import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset function
@st.cache_data
def load_data():
    df = pd.read_csv("salary_data.csv")  # Ensure this file is in the project folder
    df.dropna(inplace=True)
    return df

# Load data
df = load_data()

# Streamlit UI
st.title("💰 Salary Prediction Web App")
st.write("### Explore Salary Data")

# Show dataset preview
st.write(df.head())

# Identify categorical columns
categorical_columns = ['degree', 'job_role', 'location']
existing_categorical_columns = [col for col in categorical_columns if col in df.columns]

# Apply one-hot encoding if columns exist
if existing_categorical_columns:
    df = pd.get_dummies(df, columns=existing_categorical_columns, drop_first=True)

# Define features and target
if 'Salary' in df.columns:
    X = df.drop(columns=['Salary'])
    y = df['Salary']
else:
    st.error("Error: The 'Salary' column is missing from the dataset.")
    st.stop()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"### Model Performance:")
st.write(f"✅ **Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"✅ **R-squared (R²):** {r2:.2f}")

# Visualization
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted Salaries")
st.pyplot(fig)

# Sidebar for User Input
st.sidebar.header("📊 Enter Data for Prediction:")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(f"{col}", value=float(X[col].mean()))

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
predicted_salary = model.predict(input_df)[0]
st.sidebar.write(f"### 💲 Predicted Salary: {predicted_salary:.2f}")
