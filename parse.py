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


# def parse_patient(patient_rows):
#     patient = {}
#     patient['id'] = patient_rows[0][0]
#     patient['name'] = patient_rows[0][1]
#     patient['sex'] = patient_rows[0][2]
#     patient['age'] = age(patient_rows[0][3])
#     date_list = []
#     temp = {}
#     for row in patient_rows:
#         date = row[4].split(' ')[0]
#         if date not in temp.keys():
#             temp[date] = []
#         time = row[4].split(' ')[1]
#         temp[date].append(time)
#     date_list.append(temp)
#     patient['date'] = date_list
#     return patient

def parse_patient(patient_rows, date_list):
    patient = {}
    patient['id'] = patient_rows[0][0]
    patient['name'] = patient_rows[0][1]
    patient['sex'] = patient_rows[0][2]
    patient['age'] = age(patient_rows[0][3])
    date_value = []
    for date in date_list:
        temp = {"{}".format(date): []}
        date_value.append(temp)
    for row in patient_rows:
        date = row[4].split(' ')[0]
        time = row[4].split(' ')[1]
        date_value[date_list.index(date)][date].append(time)
    patient['date'] = date_value
    return patient


def parse_get_images_str(type):
    time = type.split('-')[1]
    image_type = type.split('-')[0]
    return image_type, time


def parse_img_rows(img_rows):
    res = {}
    res["img_org_path"] = img_rows[0][4]
    return res


def parse_label(img_rows):
    cardio = img_rows[0][1]
    pneumo = img_rows[0][2]
    pleural = img_rows[0][3]
    if cardio == 1:
        return "cardio"
    elif pneumo == 1:
        return "pneumonia"
    elif pleural == 1:
        return "pleural"
