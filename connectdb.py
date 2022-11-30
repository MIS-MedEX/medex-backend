import sqlite3

from cv2 import sort


class Database:
    def __init__(self):
        self.con = sqlite3.connect('./db/MedEX.db', check_same_thread=False)
        self.cursorObj = self.con.cursor()

    def sql_fetch_patient_list(self):
        self.cursorObj.execute(
            'SELECT ID, Name, Sex, Birthdate FROM Patient;')
        rows = self.cursorObj.fetchall()
        self.con.commit()
        for row in rows:
            print(row)
        return rows

    def sql_fetch_patient(self, patient_id):
        self.cursorObj.execute(
            'SELECT Patient.ID, Name, Sex, Birthdate, Date  FROM Patient JOIN Image ON Patient.ID = Image.PatientID WHERE Patient.ID={};'.format(patient_id))
        rows = self.cursorObj.fetchall()
        self.con.commit()
        for row in rows:
            print(row)
        return rows

    def sql_fetch_distinct_date(self, patient_id):
        date_list = []
        self.cursorObj.execute(
            'SELECT distinct(date(Date)) FROM Image WHERE PatientID={} ORDER BY Date;'.format(patient_id))
        rows = self.cursorObj.fetchall()
        self.con.commit()
        for row in rows:
            date_list.append(row[0])
        return date_list

    def sql_fetch_images(self, patient_id, date, time):
        print(patient_id, date, time)
        self.cursorObj.execute(
            'SELECT ID, Cardio, Pneumo, Pleural, Path, Report, Keyword FROM Image WHERE (PatientID=? AND datetime(Date)=?);', (patient_id, date+" "+time))
        rows = self.cursorObj.fetchall()
        self.con.commit()
        for row in rows:
            print(row)
        return rows

    def sql_update_report(self, patient_id, report, highlight, date, time):
        if highlight == "":
            if report == "":
                return "Not update report or highlight"
            sql = 'UPDATE Image SET Report="{}", Finish=1 WHERE (PatientID=? AND datetime(Date)=?);'.format(
                report)
        else:
            sql = 'UPDATE Image SET Report="{}",Keyword="{}", Finish=1 WHERE (PatientID=? AND datetime(Date)=?);'.format(
                report, highlight)
        self.cursorObj.execute(sql, (patient_id, date+" "+time))
        self.con.commit()
        if self.cursorObj.rowcount == 1:
            return "success"
        else:
            return "fail"


db = Database()
