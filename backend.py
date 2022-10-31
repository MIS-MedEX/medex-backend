from flask import Flask, request, jsonify
from connectdb import *
from parse import *

app = Flask(__name__)
@app.route("/get_patient", methods=["GET"])
def get_patient():
    patient_id = request.args.get('patient_id')
    patient_rows = db.sql_fetch_patient(patient_id)
    patient_data = parse_patient(patient_rows)
    return jsonify(patient_data)

if __name__ == "__main__":
    app.run()