CREATE TABLE Patient (
    PatientID int NOT NULL,
    PatientName varchar(50) NOT NULL,
    PatientSex varchar(50) NOT NULL,
    PatientBirthdate date NOT NULL,
    PRIMARY KEY (PatientID)
);
CREATE TABLE Image (
    ImageID int NOT NULL,
    PatientID int NOT NULL,
    ImageDate datetime NOT NULL,
    AIType varchar(50),
    ImagePath varchar(50) NOT NULL,
    Report varchar(100) NOT NULL,
    Finish boolean NOT NULL,
    PRIMARY KEY (ImageID),
    FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
);