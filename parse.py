from util import *


def parse_patient_list(patient_rows):
    patient_list = []
    for row in patient_rows:
        patient = {}
        patient['id'] = row[0]
        patient['name'] = row[1]
        patient['sex'] = row[2]
        patient['age'] = age(row[3])
        patient_list.append(patient)
    return patient_list


def parse_patient(patient_rows, date_list):
    patient = {}
    patient['id'] = patient_rows[0][0]
    patient['name'] = patient_rows[0][1]
    patient['sex'] = patient_rows[0][2]
    patient['age'] = age(patient_rows[0][3])
    date_value = {}
    for date in date_list:
        temp = {"{}".format(date): []}
        date_value[date] = []
    for row in patient_rows:
        date = row[4].split(' ')[0]
        time = row[4].split(' ')[1]
        date_value[date].append("Xray-{}".format(time))
    patient['date'] = date_value
    return patient


def parse_get_images_str(type):
    time = type.split('-')[1]
    image_type = type.split('-')[0]
    return image_type, time


def parse_img_rows(img_rows):
    res = {}
    res["img_org_path"] = img_rows[0][4]
    res["report"] = img_rows[0][5]
    if res["report"] is None:
        res["report"] = ""
    res["highlight"] = img_rows[0][6]
    if res["highlight"] is None:
        res["highlight"] = []
    else:
        res["highlight"] = res["highlight"].split("--")
    return res


def parse_label(img_rows):
    cardio = img_rows[0][1]
    pneumo = img_rows[0][2]
    pleural = img_rows[0][3]
    if cardio == 1:
        return "Cardiomegaly"
    elif pneumo == 1:
        return "Pneumonia"
    elif pleural == 1:
        return "Pleural Effusion"


def parse_pred_label(cardio_prob, pneumo_prob, pleural_prob):
    print(cardio_prob, pneumo_prob, pleural_prob)
    res = []
    if cardio_prob > 0.5:
        res.append("Cardiomegaly")
    if pneumo_prob > 0.5:
        res.append("Pneumonia")
    if pleural_prob > 0.5:
        res.append("Pleural Effusion")
    if len(res) == 0:
        return "none"
    elif len(res) == 1:
        return res[0]
    else:
        return res


def parse_highlight(highlight):
    '''
    :param highlight: list of highlight
    '''
    if highlight[0] == "" and len(highlight) == 1:
        return ""
    highlight = "--".join(highlight)
    return highlight
