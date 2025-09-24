import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_xgbmodel.pkl")
gender_encoder = joblib.load("person_gender_encoder.pkl")
default_encoder = joblib.load("previous_loan_defaults_on_file_encoder.pkl")

def main():
    st.set_page_config(page_title="Loan Status Prediction App", page_icon="💸", layout="centered")
    st.title("Loan Status Prediction App")

    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <p style='font-size: 18px; color: #555; font-weight: bold;'>Nadja Nayara Krisna</p>
        <p style='font-size: 16px; color: #555;'>NIM: 2702320425 | Class: LB09</p>
        <p style='font-size: 16px; color: #555;'>Model Deployment Midterms (3A)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
        <h3 style='color: #1f77b4;'>Welcome to the Loan Status Prediction App!</h3>
        <p style='font-size: 16px; color: #333;'>
            This app uses a machine learning model to predict whether your loan application is likely to be <strong>approved</strong> or <strong>rejected</strong>. 
            Simply input your financial details, such as income, credit score, and loan amount, and let our model analyze the data to provide an instant prediction.
        </p>
        <p style='font-size: 16px; color: #333;'>
            <strong>Why use this app?</strong> It’s fast, user-friendly, and helps you understand your loan eligibility before applying formally. 
            Start by entering your details below!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Tip**: Ensure all inputs are accurate for the most reliable prediction.")

    st.subheader("Enter Your Details")
    person_age = st.number_input("Enter Age", min_value=18, max_value=100)
    person_gender = st.radio("Select Gender", options=["male", "female"])
    person_income = st.number_input("Enter Yearly Income", min_value=1000)
    person_education = st.selectbox("Select Latest Education", options=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    previous_loan_defaults_on_file = st.radio("Previous Loan Defaults", options=["No", "Yes"])
    person_emp_exp = st.number_input("Employment Experience (in years)", min_value=0)
    person_home_ownership = st.selectbox("Home Ownership", options=["Own", "Rent", "Mortgage"])
    loan_intent = st.selectbox("Loan Intent", options=['VENTURE', 'EDUCATION', 'MEDICAL', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_amnt = st.number_input("Loan Amount", min_value=500)
    loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0)
    cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0)

    data = {
        'person_age': int(person_age),
        'person_gender': person_gender,
        'person_income': float(person_income),
        'person_education': person_education,
        'previous_loan_defaults_on_file': previous_loan_defaults_on_file,
        'person_emp_exp': float(person_emp_exp),
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent,
        'loan_amnt': float(loan_amnt),
        'loan_int_rate': float(loan_int_rate),
        'cb_person_cred_hist_length': int(cb_person_cred_hist_length),
        'credit_score': int(credit_score),
        'loan_percent_income': float(loan_percent_income)
    }

    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))

    df['person_gender'] = gender_encoder.transform(df['person_gender'])
    df['previous_loan_defaults_on_file'] = default_encoder.transform(df['previous_loan_defaults_on_file'])

    education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    df['person_education'] = pd.Categorical(df['person_education'], categories=education_order, ordered=True).codes

    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'], drop_first=False)

    required_columns = model.get_booster().feature_names
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[required_columns]

    if st.button('Make Prediction'):
        result = make_prediction(df)
        prediction_text = "APPROVED" if result == 1 else "REJECTED"
        bg_color = "green" if result == 1 else "red"
        st.markdown(
            f'<div style="background-color:{bg_color}; color:white; padding:10px; border-radius:5px; text-align:center; width:fit-content;">'
            f'The loan application is likely to be: {prediction_text}</div>',
            unsafe_allow_html=True
        )

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
