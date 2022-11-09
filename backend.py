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
    date_list = db.sql_fetch_distinct_date(patient_id)
    patient_data = parse_patient(patient_rows, date_list)
    return jsonify(patient_data)


@app.route("/api/patient/<id>/image", methods=["GET"])
def get_images(id):

    patient_id = id
    date = request.args.get('date')
    type = request.args.get('type')
    image_type, time = parse_get_images_str(type)
    image_rows = db.sql_fetch_images(patient_id, date, time)
    label = parse_label(image_rows)
    res = parse_img_rows(image_rows)
    cardio_task_pred, pneumo_task_pred, pleural_task_pred, cardio_vis_path, pneumonia_vis_path, pleural_vis_path = validate(
        res["img_org_path"])
    res["res_our_cardio"] = {"prob": float(cardio_task_pred),
                             "vis_path": cardio_vis_path}
    res["res_our_pneumo"] = {"prob": float(pneumo_task_pred),
                             "vis_path": pneumonia_vis_path}
    res["res_our_pleural"] = {"prob": float(pleural_task_pred),
                              "vis_path": pleural_vis_path}
    # res["res_baseline_cardio"] = {"prob": float(cardio_task_pred)}
    # res["res_baseline_pneumo"] = {"prob": float(pneumo_task_pred)}
    # res["res_baseline_pleural"] = {"prob": float(pleural_task_pred)}
    res["img_label"] = label
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
