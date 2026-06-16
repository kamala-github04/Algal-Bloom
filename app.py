import streamlit as st
import pandas as pd
import requests
import joblib

# ==========================
# CONFIGURATION
# ==========================

CHANNEL_ID = "YOUR_CHANNEL_ID"
READ_API_KEY = "YOUR_READ_API_KEY"

model = joblib.load("hab_model.pkl")

risk_map = {
    0: "🟢 Low Risk",
    1: "🟡 Medium Risk",
    2: "🔴 High Risk"
}

# ==========================
# FETCH LATEST THINGSPEAK DATA
# ==========================

def get_latest_data():

    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?results=1"

    if READ_API_KEY:
        url += f"&api_key={READ_API_KEY}"

    response = requests.get(url)
    data = response.json()

    latest = data["feeds"][0]

    return {
        "turbidity": float(latest["field1"]),
        "temperature": float(latest["field2"]),
        "tds": float(latest["field3"]),
        "ph": float(latest["field4"])
    }

# ==========================
# PAGE
# ==========================

st.set_page_config(
    page_title="Smart Aquaculture HAB Dashboard",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Smart Aquaculture HAB Prediction Dashboard")

try:

    data = get_latest_data()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Turbidity", data["turbidity"])
    col2.metric("Temperature", data["temperature"])
    col3.metric("TDS", data["tds"])
    col4.metric("pH", data["ph"])

    input_df = pd.DataFrame({
        "ph_filtered": [data["ph"]],
        "temperature_filtered": [data["temperature"]],
        "turbidity_filtered": [data["turbidity"]],
        "tds_filtered": [data["tds"]]
    })

    prediction = model.predict(input_df)[0]

    st.subheader("Predicted HAB Risk")

    if prediction == 0:
        st.success(risk_map[prediction])

    elif prediction == 1:
        st.warning(risk_map[prediction])

    else:
        st.error(risk_map[prediction])

    st.write("Latest Sensor Data")
    st.dataframe(input_df)

except Exception as e:
    st.error(f"Error fetching ThingSpeak data: {e}")
