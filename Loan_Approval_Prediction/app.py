from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('Decesion_Tree_Model.pkl','rb'))

# @app.route('/')
# def home():
#     return render_template('RegisterDetails.html')

@app.route('/',methods=['GET' , 'POST'])
def predict():
    if request.method == "POST":
        name = request.form.get('name')

        gender = request.form.get('gender')
        marital_status = request.form.get('marital_status')
        dependents = request.form.get('dependents')
        education = request.form.get('education')
        Self_Employed = request.form.get('Self_Employed')
        ApplicantIncome = request.form.get('ApplicantIncome')
        coapp_income = request.form.get('coapp_income')
        loan_amount = request.form.get('loan_amount')
        term = request.form.get('term')
        credit_history = request.form.get('credit_history')
        property_area = request.form.get('property_area')



        prediction=model.predict(pd.DataFrame(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
                                data=np.array([gender, marital_status, dependents, education, Self_Employed, ApplicantIncome, coapp_income, loan_amount, term, credit_history, property_area]).reshape(1, 11)))
        
        print(prediction)

        if prediction == 1:
            pred = "Loan_Approved"
        else:
            pred = "Not_Approved"

        return pred
    return render_template('RegisterDetails.html')
    


if __name__=='__main__':
    app.run()