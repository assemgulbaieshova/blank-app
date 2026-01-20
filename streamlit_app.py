import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("student_performance.csv")

st.title("🎓 Student Performance Prediction")

st.write(df.head())

# Select features and target
X = df.drop("Final_Exam_Score", axis=1)
y = df["Final_Exam_Score"]

# Identify categorical vs numeric
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]
)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
score = r2_score(y_test, pred)

st.write("### Model R2 Score:", score)

# Sidebar inputs
st.sidebar.header("Input Student Data")
input_data = {}
for col in cat_cols:
    input_data[col] = st.sidebar.selectbox(col, df[col].unique())
for col in num_cols:
    input_data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write("### Predicted Final Exam Score:", prediction[0])
