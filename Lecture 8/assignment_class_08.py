# -*- coding: utf-8 -*-
"""Assignment-Class 08.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17ACZT52VAlQJM8xaF6JubzW44E8SdcGr
"""

#Sales Forecasting for a Retail Store

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data_sales = pd.read_csv("retail_sales.csv")

# Encoding categorical variables
encoder = OneHotEncoder(drop='first', sparse_output =False)  # Drop first category to avoid multicollinearity
encoded_columns = encoder.fit_transform(data_sales[['Gender', 'Product Category']])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(['Gender', 'Product Category']))

# Select numerical features
X_sales_numeric = data_sales[['Age', 'Quantity', 'Price per Unit']]

# Combine encoded and numerical features
X_sales = pd.concat([X_sales_numeric, encoded_df], axis=1)

# Target variable
y_sales = data_sales['Total Amount']

# Train-test split
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42)

# Train model
model_sales = LinearRegression()
model_sales.fit(X_train_sales, y_train_sales)

# Predictions
y_pred_sales = model_sales.predict(X_test_sales)

# Evaluation metrics
mse_sales = mean_squared_error(y_test_sales, y_pred_sales)
r2_sales = r2_score(y_test_sales, y_pred_sales)

print(f"Sales Forecast - MSE: {mse_sales:.2f}")
print(f"Sales Forecast - R-squared: {r2_sales:.4f}")

plt.figure(figsize=(10, 6))

# Scatter plot with improved aesthetics
plt.scatter(y_test_sales, y_pred_sales,
            c=y_test_sales, cmap='coolwarm',  # Color gradient based on actual values
            alpha=0.75, edgecolors='black', s=80)  # Larger points with black edges

# Reference line (perfect predictions)
plt.plot([y_test_sales.min(), y_test_sales.max()],
         [y_test_sales.min(), y_test_sales.max()],
         color="red", linestyle="--", linewidth=2, label="Perfect Prediction")

# Titles and labels with better styling
plt.xlabel("Actual Sales", fontsize=14, fontweight='bold')
plt.ylabel("Predicted Sales", fontsize=14, fontweight='bold')
plt.title("Actual vs. Predicted Sales", fontsize=16, fontweight='bold', color='darkblue')

# Grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adding a legend
plt.legend(fontsize=12, loc="upper left")

# Show the improved plot
plt.show()

#Email Spam Detection using SVM

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset with correct column names
df = pd.read_csv("spam.csv", encoding='latin-1', names=["label", "message"], usecols=[0, 1])

# Check for missing values before cleaning
print("Missing values before cleaning:\n", df.isnull().sum())

# Drop rows with missing values in label or message
df = df.dropna(subset=['label', 'message'])

# Verify all NaNs are removed
print("Missing values after cleaning:\n", df.isnull().sum())

# Convert labels to binary (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
corpus = []

for msg in df['message']:
    msg = re.sub(r'[^a-zA-Z]', ' ', msg).lower().split()  # Remove non-alphabetic chars, convert to lowercase
    msg = [ps.stem(word) for word in msg if word not in stop_words]  # Remove stopwords and stem words
    corpus.append(" ".join(msg))

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()
y = df['label'].values

# Remove rows with NaN values in y
# This is the key change to fix the error
X = X[~np.isnan(y)]
y = y[~np.isnan(y)]


# Final check for NaN values in y
print("NaN values in y after processing:", pd.isnull(y).sum())  # Should print 0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)  # Should now work fine

# Predict and Evaluate
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Customer Churn Prediction using SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Telco_customer_churn.csv")
# Drop unnecessary columns
df = df.drop(columns=['customerID'])

# Convert 'TotalCharges' to numeric (handling potential non-numeric values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)  # Replace NaN values with 0

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

# Define features and target variable
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM Model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svm_model.predict(X_test)

# Print Performance Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Fraud Detection in Credit Card Transactions

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load dataset (Replace 'your_dataset.csv' with actual file path)
df = pd.read_csv("credit_card.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Drop rows with missing target values
df.dropna(subset=['Delinquent_Acc'], inplace=True)

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_enc.fit_transform(df[col])

# Define features (X) and target (y)
X = df.drop(columns=['Delinquent_Acc'])  # Predicting Delinquent Account status
y = df['Delinquent_Acc']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM Model
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

