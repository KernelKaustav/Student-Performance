import streamlit as st
import joblib
import numpy as np

# ---------------- Load Model ----------------
try:
    model = joblib.load("model.pkl")
except:
    st.error("Model not found! Please run train.py first.")
    st.stop()

# ---------------- UI ----------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Predict final student grade (G3) using machine learning")

st.divider()

# ---------------- Inputs ----------------
st.subheader("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    G1 = st.slider("G1 (First Period Grade)", 0, 20, 10)
    studytime = st.selectbox("Study Time", [1, 2, 3, 4])

with col2:
    G2 = st.slider("G2 (Second Period Grade)", 0, 20, 10)
    failures = st.selectbox("Past Failures", [0, 1, 2, 3])

st.divider()

# ---------------- Prediction ----------------
if st.button("🚀 Predict Performance"):

    features = np.array([[G1, G2, studytime, failures]])
    prediction = model.predict(features)[0]

    st.subheader("📈 Prediction Result")
    st.success(f"Predicted Final Grade (G3): {prediction:.2f} / 20")

    # ---------------- Interpretation ----------------
    st.subheader("🧠 Interpretation")

    if prediction >= 15:
        st.success("🔥 Excellent performance expected!")
    elif prediction >= 10:
        st.info("👍 Average performance expected.")
    else:
        st.warning("⚠️ Risk of low performance.")

    # ---------------- Insight ----------------
    st.subheader("📌 Model Insight")

    st.write("""
    - 📈 **G2 (recent performance)** has the strongest impact  
    - 📊 G1 has moderate influence  
    - 📉 Failures negatively affect performance  
    - ⏳ Study time has limited impact  
    """)

st.divider()

# ---------------- Footer ----------------
st.caption("Built using Machine Learning (Random Forest + SHAP)")