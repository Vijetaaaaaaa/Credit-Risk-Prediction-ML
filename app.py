import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Default Risk Predictor")
st.markdown("Fill in the customer details below to predict credit default risk.")

# --- Input Fields ---
st.subheader("Customer Info")
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=0)
sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
education = st.selectbox("Education", options=[1, 2, 3, 4],
    format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
marriage = st.selectbox("Marriage", options=[1, 2, 3],
    format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
age = st.number_input("Age", min_value=18, max_value=100)

st.subheader("Repayment Status (last 6 months)")
st.caption("-2 = No consumption, -1 = Paid in full, 0 = Minimum paid, 1–8 = Months delayed")
pay_0 = st.slider("Repayment Status - Sep (PAY_0)", -2, 8, 0)
pay_2 = st.slider("Repayment Status - Aug (PAY_2)", -2, 8, 0)
pay_3 = st.slider("Repayment Status - Jul (PAY_3)", -2, 8, 0)
pay_4 = st.slider("Repayment Status - Jun (PAY_4)", -2, 8, 0)
pay_5 = st.slider("Repayment Status - May (PAY_5)", -2, 8, 0)
pay_6 = st.slider("Repayment Status - Apr (PAY_6)", -2, 8, 0)

st.subheader("Bill Amounts (last 6 months)")
bill_amt1 = st.number_input("Bill Amount - Sep (BILL_AMT1)", value=0)
bill_amt2 = st.number_input("Bill Amount - Aug (BILL_AMT2)", value=0)
bill_amt3 = st.number_input("Bill Amount - Jul (BILL_AMT3)", value=0)
bill_amt4 = st.number_input("Bill Amount - Jun (BILL_AMT4)", value=0)
bill_amt5 = st.number_input("Bill Amount - May (BILL_AMT5)", value=0)
bill_amt6 = st.number_input("Bill Amount - Apr (BILL_AMT6)", value=0)

st.subheader("Payment Amounts (last 6 months)")
pay_amt1 = st.number_input("Payment Amount - Sep (PAY_AMT1)", value=0)
pay_amt2 = st.number_input("Payment Amount - Aug (PAY_AMT2)", value=0)
pay_amt3 = st.number_input("Payment Amount - Jul (PAY_AMT3)", value=0)
pay_amt4 = st.number_input("Payment Amount - Jun (PAY_AMT4)", value=0)
pay_amt5 = st.number_input("Payment Amount - May (PAY_AMT5)", value=0)
pay_amt6 = st.number_input("Payment Amount - Apr (PAY_AMT6)", value=0)

# --- Predict ---
if st.button("🔮 Predict Risk"):
    features = np.array([[
        limit_bal, sex, education, marriage, age,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
        bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
        pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")
    st.write(f"**Default Probability: {prob:.2f}**")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk — this customer is likely to default ({prob:.0%})")
    else:
        st.success(f"✅ Low Risk — this customer is unlikely to default ({prob:.0%})")

    st.progress(int(prob * 100))