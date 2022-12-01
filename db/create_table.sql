CREATE TABLE Patient (
    ID int NOT NULL,
    Name varchar(50) NOT NULL,
    Sex varchar(50) NOT NULL,
    Birthdate date NOT NULL,
    PRIMARY KEY (ID)
);
CREATE TABLE Image (
    ID int NOT NULL,
    PatientID int NOT NULL,
    Date datetime NOT NULL,
    AIType varchar(50),
    Cardio int,
    Pneumo int,
    Pleural int,
    Path varchar(50) NOT NULL,
    Keyword TEXT,
    Report TEXT,
    Finish boolean NOT NULL,
    PRIMARY KEY (ID),
    FOREIGN KEY (PatientID) REFERENCES Patient(ID)
);