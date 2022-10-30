import sqlite3


def sql_fetch_patient(patient_id):
    con = sqlite3.connect('./db/MedEX.db')
    cursorObj = con.cursor()
    cursorObj.execute('SELECT Patient.ID, Name, Sex, Birthdate, Date  FROM Patient JOIN Image ON Patient.ID = Image.PatientID WHERE Patient.ID={}'.format(patient_id))
    rows = cursorObj.fetchall()
    for row in rows:
        print(row)
    con.close()
    return rows

sql_fetch_patient(1)
