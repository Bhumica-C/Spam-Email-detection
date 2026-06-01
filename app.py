import streamlit as st
import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from datetime import datetime

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Create dataset file
log_file = "email_activity_log.csv"

if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["date","type"])
    df.to_csv(log_file,index=False)

st.title("Spam Email Detection System")

trusted_domains = [
    "coursera.org",
    "edx.org",
    "google.com",
    "microsoft.com",
    "udemy.com"
]

risk_score = 0

# Layout
col1, col2 = st.columns([2,1])

with col1:
    message = st.text_area("Enter Email / Message", height=200)
    sender = st.text_input("Sender Email or Domain")
    analyze = st.button("Analyze Message")

if analyze:

    if message.strip() == "":
        st.warning("Please enter a message")

    else:

        vector = vectorizer.transform([message])
        probability = model.predict_proba(vector)[0][1]

        risk_score = int(probability * 100)

        trusted = False

        if sender:
            for domain in trusted_domains:
                if domain in sender.lower():
                    trusted = True

        # Adjust risk score using domain
        if trusted:
            risk_score -= 20
        else:
            risk_score += 10

        risk_score = max(0, min(risk_score,100))

        # Final classification
        if risk_score >= 80:

            st.error("Result: BLOCK (Spam)")
            email_type = "spam"

        elif risk_score >= 40:

            st.warning("Result: SUSPICIOUS")
            email_type = "suspicious"

        else:

            st.success("Result: SAFE")
            email_type = "safe"

        # Save to dataset
        today = datetime.now()

        new_data = pd.DataFrame({
           "date":[today.strftime("%Y-%m-%d")],
           "day":[today.strftime("%A")],
           "month":[today.strftime("%B")],
           "type":[email_type]
         })

        new_data.to_csv(log_file, mode="a", header=False, index=False)

        # Location detection
        try:
            response = requests.get("https://ipinfo.io/json")
            data = response.json()

            st.write("IP:", data["ip"])
            st.write("City:", data["city"])
            st.write("Region:", data["region"])
            st.write("Country:", data["country"])

        except:
            st.write("Location not detected")

# Risk Meter
with col2:

    st.subheader("Spam Risk Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Spam Risk %"},
        gauge={
            'axis': {'range': [0,100]},
            'steps': [
                {'range': [0,40], 'color': "lightgreen"},
                {'range': [40,70], 'color': "yellow"},
                {'range': [70,100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig)

# Daily Email Graph
st.subheader("Daily Email Detection Activity")

df = pd.read_csv(log_file)

if df.empty:
    st.info("Analyze emails to generate graph")

else:

    counts = df.groupby(["date","type"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots()

    counts.plot(
        kind="bar",
        ax=ax,
        color={
            "safe":"green",
            "suspicious":"yellow",
            "spam":"red"
        }
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Emails")
    ax.set_title("Daily Email Detection")

    st.pyplot(fig)