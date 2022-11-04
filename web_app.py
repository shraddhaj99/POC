import pickle
import pandas as pd
from bank import y_pred_base
#decision tree model
pickle.dump(y_pred_base, open('loan_model.pkl', 'wb'))

import streamlit as st


st.header("Bank Loan Prediction")

st.write("""

Data used for this modelling is taken from Kaggle.com. 
It has the following features:

ID-	Customer ID						
Age-	Customer's age in completed years						
Experience-	#years of professional experience						
Income-	Annual income of the customer ($000)						
ZIPCode-	Home Address ZIP code.						
Family-	Family size of the customer						
CCAvg-	Avg. spending on credit cards per month ($000)						
Education-	Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional						
Mortgage-	Value of house mortgage if any. ($000)						
Personal Loan-	Did this customer accept the personal loan offered in the last campaign?						
Securities Account-	Does the customer have a securities account with the bank?						
CD Account-	Does the customer have a certificate of deposit (CD) account with the bank?						
Online-	Does the customer use internet banking facilities?						
CreditCard-	Does the customer use a credit card issued by UniversalBank?

""")

st.sidebar.header("User Input Features")

def user_input():
    id = st.sidebar.slider("ID", 1, 6000, 3000)
    age = st.sidebar.slider("Age", 21, 70, 21)
    experience = st.sidebar.slider("Experience", 1,15, 1)
    income = st.sidebar.slider("Income", 1, 10000, 1)
    zipcode = st.sidebar.slider("Zip Code", 93000, 95010, 93000)
    fam = st.sidebar.slider("Family Size", 1, 4, 1)
    ccavg = st.sidebar.slider("Average expenditure on Credit Cards", 1,15000, 1)
    edu = st.sidebar.slider("Education", 1,3,1)
    mortgage = st.sidebar.slider("Mortgage", 0, 1000, 0)
    personalloan = st.sidebar.selectbox("Personal Loan(Yes - 1, No- 0", (0,1))
    security = st.sidebar.selectbox("Security Account", (0,1))
    CD = st.sidebar.selectbox("Certificate of Deposit available? Yes-1, no-0", (0,1))
    online = st.sidebar.selectbox("Do you need online services?", (0,1))
    ccard = st.sidebar.selectbox("Do you use credit card?",(0,1))

    data = {
        'id' : id,
        'age' : age,
        'experience' : experience,
        'income' : income,
        'zipcode' : zipcode,
        'fam' : fam,
        'ccavg' : ccavg,
        'edu' : edu,
        'mortgage' : mortgage,
        'personalloan' : personalloan,
        'security' : security,
        'CD' : CD,
        'online' : online,
        'ccard' : ccard
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input()

df = pd.read_csv('bank.csv')

X = df.drop('personalloan', axis = 1)
df = pd.concat([input_df, X], axis=0)

#Write out input selection
st.subheader('User Input (Pandas DataFrame)')
st.write(df)

#Load in model
load_mod = pickle.load(open('loan_model.pkl', 'rb'))

#Apply model to make predictions
pred = load_mod.predict(df)
pred_prob = load_mod.predict_proba(df)

st.subheader("prediction")
st.write("""
This is a binary classification. 

1 means yes
0 means no

""")
import numpy as np
readmitted = np.array([0,1])
st.write(readmitted[pred])

st.subheader('Prediction Probability')
st.write("""
0 --> 'NO'
1 -->  'YES'
""")
st.write(pred_prob)