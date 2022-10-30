from flask import Flask, request, jsonify
from connectdb import *
from datetime import date

app = Flask(__name__)

def age(birthdate):
    today = date.today()
    age = today.year - int(birthdate.split('-')[0]) - ((today.month, today.day) < (int(birthdate.split('-')[1]), int(birthdate.split('-')[2])))
    return age

def parse_patient(patient_rows):
    patient = {}
    patient['patient_id'] = patient_rows[0][0]
    patient['name'] = patient_rows[0][1]
    patient['sex']= patient_rows[0][2]
    patient['age'] = age(patient_rows[0][3])
    date=[]
    for row in patient_rows:
        date.append(row[4])
    patient['date'] = date
    return patient

@app.route("/get_patient", methods=["GET"])
def get_patient():
    patient_id = request.args.get('patient_id')
    patient_rows = sql_fetch_patient(patient_id)
    patient_data = parse_patient(patient_rows)
    return jsonify(patient_data)

if __name__ == "__main__":
    app.run()