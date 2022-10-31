from flask import Flask, request, jsonify
from flask_cors import CORS
from connectdb import *
from parse import *

app = Flask(__name__)
CORS(app)


@app.route("/get_patient", methods=["GET"])
def get_patient():
    patient_id = request.args.get('id')
    patient_rows = db.sql_fetch_patient(patient_id)
    patient_data = parse_patient(patient_rows)
    return jsonify(patient_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
