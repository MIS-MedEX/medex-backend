from flask import Flask, request, jsonify
from flask_cors import CORS
from connectdb import *
from parse import *
from model.inference import *
import math

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
    validate_ret = validate(res["img_org_path"])
    res["res_our_cardio"] = {"prob": math.floor(float(validate_ret["cardio"]["our"])*100)/100,
                             "vis_path": validate_ret["cardio"]["vis"],
                             "error": math.floor(abs(image_rows[0][1]-float(validate_ret["cardio"]["our"]))*100)/100}
    res["res_our_pneumo"] = {"prob": math.floor(float(validate_ret["pneumonia"]["our"])*100)/100,
                             "vis_path": validate_ret["pneumonia"]["vis"],
                             "error": math.floor(abs(image_rows[0][2]-float(validate_ret["pneumonia"]["our"]))*100)/100}
    res["res_our_pleural"] = {"prob": math.floor(float(validate_ret["pleural"]["our"])*100)/100,
                              "vis_path": validate_ret["pleural"]["vis"],
                              "error": math.floor(abs(image_rows[0][3]-float(validate_ret["pleural"]["our"]))*100)/100}
    res["res_baseline_cardio"] = {
        "prob": math.floor(float(validate_ret["cardio"]["baseline"])*100)/100,
        "error": math.floor(abs(image_rows[0][1]-float(validate_ret["cardio"]["baseline"]))*100)/100}
    res["res_baseline_pneumo"] = {
        "prob": math.floor(float(validate_ret["pneumonia"]["baseline"])*100)/100,
        "error": math.floor(abs(image_rows[0][2]-float(validate_ret["pneumonia"]["baseline"]))*100)/100}
    res["res_baseline_pleural"] = {
        "prob": math.floor(float(validate_ret["pleural"]["baseline"])*100)/100,
        "error": math.floor(abs(image_rows[0][3]-float(validate_ret["pleural"]["baseline"]))*100)/100}
    res["img_label"] = label
    return jsonify(res)


@app.route("/api/patient/save_report", methods=["POST"])
def save_report():
    data = request.get_json()
    patient_id = data["id"]
    date = data["date"]
    type = data["type"]
    image_type, time = parse_get_images_str(type)
    report = data["report"]
    highlight = data["highlight"]  # ["str", "str", "str"]
    highlight = parse_highlight(highlight)
    status = db.sql_update_report(patient_id, report, highlight, date, time)
    return jsonify({"status": status})


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
