import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the machine learning model
model = joblib.load('trained_model.pkl')

def main():
    le = LabelEncoder()
    st.title('UTS Model Deployment')

    # Add user input components for 5 features
    person_age = st.slider('person_age', min_value=0.0, max_value=150.0, value=0.0)

    person_gender = st.segmented_control('person_gender', ['Female', 'Male'])
    # encode gender
    person_gender = {'Female':0, 'Male': 1}.get(person_gender, -1)

    person_education = st.selectbox('person_education', ['Bachelor', 'Associate', 'High School', 'Master', 'Doctorate'])
    person_education = {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4}.get(person_education, -1)

    person_income = st.number_input('person_income', min_value=0.0, max_value=10000000.0, value=0.0)

    person_emp_exp = st.slider('person_emp_exp', min_value=0.0, max_value=80.0, value=0.0)

    person_home_ownership = st.selectbox('person_home_ownership', ['Rent', 'Mortgage', 'Own', 'Other'])
    person_gender = {"Mortgage":0, "Other": 1, "Own": 2, "Rent":3}.get(person_gender, -1)

    loan_amnt = st.number_input('loan_amnt', min_value=0.0, max_value=100000.0, value=0.0)

    loan_intent = st.selectbox('loan_intent', ['Education', 'Medical', 'Venture', 'Personal', 'Debt Consolidation', 'Home Improvement'])
    person_gender = {"Debt Consolidation":0, "Education":1, "Home Improvement":2, "Medical":3, "Personal":4, "Venture":5}.get(person_gender, -1)

    loan_int_rate = st.slider('loan_int_rate', min_value=0.0, max_value=30.0, value=0.0)

    loan_percent_income = st.slider('loan_percent_income', min_value=0.0, max_value=1.0, value=0.0)

    cb_person_cred_hist_length = st.slider('cb_person_cred_hist_length', min_value=0.0, max_value=50.0, value=0.0)

    credit_score = st.slider('credit_score', min_value=0.0, max_value=850.0, value=0.0)

    previous_loan_defaults_on_file = st.segmented_control('previous_loan_defaults_on_file', ['Yes', 'No'])
    person_gender = {"No":0, "Yes":1}.get(person_gender, -1)
    
# get label mapping
        #print(dict(zip(le.classes_, le.transform(le.classes_))))
        #print(dict(zip(le_edu.classes_, le_edu.transform(le_edu.classes_))))
        #print(dict(zip(le_hownership.classes_, le_hownership.transform(le_hownership.classes_))))
        #print(dict(zip(le_intent.classes_, le_intent.transform(le_intent.classes_))))
        #print(dict(zip(le_prev_loan.classes_, le_prev_loan.transform(le_prev_loan.classes_))))

    if st.button('Make Prediction'):
        features = [person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

