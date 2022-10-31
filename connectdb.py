import sqlite3


class Database:
    def __init__(self):
        self.con = sqlite3.connect('./db/MedEX.db', check_same_thread=False)
        self.cursorObj = self.con.cursor()

    def sql_fetch_patient(self, patient_id):
        self.cursorObj.execute('SELECT Patient.ID, Name, Sex, Birthdate, Date  FROM Patient JOIN Image ON Patient.ID = Image.PatientID WHERE Patient.ID={}'.format(patient_id))
        rows = self.cursorObj.fetchall()
        self.con.commit()
        for row in rows:
            print(row)
        return rows
db=Database()
