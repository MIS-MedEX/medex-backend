from util import *
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