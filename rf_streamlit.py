import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load model pickle
model = joblib.load('trained_model.pkl')

def main():
    le = LabelEncoder()
    st.title('UTS Model Deployment')

    # user input untuk di streamlit webnya nnt
    person_age = st.slider('Age', min_value=0, max_value=100, value=0)

    person_gender = st.segmented_control('Gender', ['Female', 'Male'])
    # encode gender berdasarkan encoding yg digunakan di model
    person_gender = {'Female':0, 'Male': 1}.get(person_gender, -1)

    person_education = st.selectbox('Last Education Level', ['Bachelor', 'Associate', 'High School', 'Master', 'Doctorate'])
    person_education = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}.get(person_education, -1)

    person_income = st.number_input('Income', min_value=0, max_value=10000000, value=0)

    person_emp_exp = st.slider('Employment Experience', min_value=0, max_value=50, value=0)

    person_home_ownership = st.selectbox('Home Ownership Status', ['Rent', 'Mortgage', 'Own', 'Other'])
    person_home_ownership = {"Mortgage":0, "Other": 1, "Own": 2, "Rent":3}.get(person_home_ownership, -1)

    loan_amnt = st.number_input('Loan Amount', min_value=0, max_value=100000, value=0)

    loan_intent = st.selectbox('Loan Intent', ['Education', 'Medical', 'Venture', 'Personal', 'Debt Consolidation', 'Home Improvement'])
    loan_intent = {"Debt Consolidation":0, "Education":1, "Home Improvement":2, "Medical":3, "Personal":4, "Venture":5}.get(loan_intent, -1)

    loan_int_rate = st.slider('Loan Interest Rate', min_value=0.0, max_value=30.0, value=0.0)

    loan_percent_income = st.slider('Loan Percent Income', min_value=0.0, max_value=1.0, value=0.0)

    cb_person_cred_hist_length = st.slider("Credit Bureau: Person's Credit History Length", min_value=0, max_value=50, value=0)

    credit_score = st.slider('Credit Score', min_value=0, max_value=850, value=0)

    previous_loan_defaults_on_file = st.segmented_control('Do You Have a Previous Loan On This File', ['Yes', 'No'])
    previous_loan_defaults_on_file = {"No":0, "Yes":1}.get(previous_loan_defaults_on_file, -1)

    if st.button('Make Prediction'):
        features = [person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file]
        result = make_prediction(features)
        result_text = {0: "Loan Not Approved", 1: "Loan Approved"}.get(result, -1)
        st.success(f'The prediction is: {result} ({result_text})')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

