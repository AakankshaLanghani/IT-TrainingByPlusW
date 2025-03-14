import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st


# Load dataset function
@st.cache_data
def load_data():
    file_path = "customer_churn_data.csv"  # Ensure correct path
    df = pd.read_csv(file_path)

    # Check if dataset is empty
    if df.empty:
        st.error("Error: Dataset is empty. Please check the file.")
        st.stop()

    # Preprocess data
    df.dropna(inplace=True)  # Remove missing values

    # Convert categorical columns to numeric
    df['churn'] = df['churn'].astype(int)
    df = pd.get_dummies(df, columns=['international_plan', 'voice_mail_plan'], drop_first=True)

    # Drop unnecessary columns
    if {'Id', 'state', 'phone_number'}.issubset(df.columns):
        df.drop(columns=['Id', 'state', 'phone_number'], inplace=True)

    return df


# Load and process data
df = load_data()

# Define features and target variable
X = df.drop(columns=['churn'])
y = df['churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit App UI
st.title("📊 Customer Churn Prediction")
st.write(f"✅ **Model Accuracy:** {accuracy:.2f}")

# Display confusion matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Sidebar for User Input Prediction
st.sidebar.header("🔍 Predict Customer Churn")
features = {col: st.sidebar.number_input(f"{col}:", float(X[col].min()), float(X[col].max()), float(X[col].mean())) for
            col in X.columns}

# Predict churn when user clicks the button
if st.sidebar.button("Predict"):
    input_data = np.array([features[col] for col in X.columns]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    st.sidebar.write(f"🛑 **Predicted Churn:** {'Yes' if prediction == 1 else 'No'}")
