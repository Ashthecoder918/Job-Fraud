import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and vectorizer
model = joblib.load("fraud_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load your cleaned dataset (used for visualization)
@st.cache_data
def load_data():
    return pd.read_csv("train_dataset.csv")  # Replace with your actual file

df = load_data()

# ---------- UI Layout ----------
st.title("🕵️‍♂️ Fraudulent Job Posting Detector")
st.write("Enter job details below to check if it's fraudulent.")

# ---------- Input Text Box ----------
user_input = st.text_area("📝 Paste job details here (title, description, etc.):")

if st.button("🔍 Predict"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        result = "Fraudulent ❌" if prediction == 1 else "Legitimate ✅"
        st.subheader(f"Prediction: {result}")
    else:
        st.warning("Please enter job text to predict.")

# ---------- Visualizations ----------
st.header("📊 Data Insights")

# Pie Chart for class distribution
st.subheader("Fraudulent vs Real Job Postings")
fig1, ax1 = plt.subplots()
labels = ['Real', 'Fraudulent']
sizes = df['fraudulent'].value_counts().sort_index()
colors = ['#66bb6a', '#ef5350']
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.axis('equal')
st.pyplot(fig1)

# Histogram: Top job locations
st.subheader("Top Job Locations")
top_locations = df['location'].value_counts().nlargest(10)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_locations.values, y=top_locations.index, ax=ax2, palette="Blues_r")
ax2.set_xlabel("Number of Postings")
ax2.set_ylabel("Location")
st.pyplot(fig2)
