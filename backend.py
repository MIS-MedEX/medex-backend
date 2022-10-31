from flask import Flask, request, jsonify
from flask_cors import CORS
from connectdb import *
from parse import *
from model.inference import *

app = Flask(__name__)
CORS(app)

"""
RESTful api
/api/patient (get)
/api/patient (insert)
/api/patient/<patient_id> (get)
/api/patient/<patient_id> (update)
/api/patient/<patient_id> (delete)
"""


@app.route("/api/patient", methods=["GET"])
def get_patient_list():
    patient_rows = db.sql_fetch_patient_list()
    patient_data = parse_patient_list(patient_rows)
    return jsonify(patient_data)


@app.route("/api/patient/<id>", methods=["GET"])
def get_patient(id):
    patient_id = id
    patient_rows = db.sql_fetch_patient(patient_id)
    patient_data = parse_patient(patient_rows)
    return jsonify(patient_data)


@app.route("/api/patient/<id>/image", methods=["GET"])
def get_images(id):
    patient_id = id
    date = request.args.get('date')
    type = request.args.get('type')
    image_type, time = parse_get_images_str(type)
    image_row = db.sql_fetch_images(patient_id, date, time)
    res = parse_img_rows(image_row)
    cardio_task_pred, pneumo_task_pred, pleural_task_pred = validate(
        res["img_org_path"])
    res["img_vis_cardio"] = {"prob": float(cardio_task_pred)}
    res["img_vis_pneumo"] = {"prob": float(pneumo_task_pred)}
    res["img_vis_pleural"] = {"prob": float(pleural_task_pred)}
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
