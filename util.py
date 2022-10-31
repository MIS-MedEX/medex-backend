from datetime import date
def age(birthdate):
    today = date.today()
    age = today.year - int(birthdate.split('-')[0]) - ((today.month, today.day) < (int(birthdate.split('-')[1]), int(birthdate.split('-')[2])))
    return age