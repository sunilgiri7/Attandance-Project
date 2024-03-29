import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://attandance-recorder-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "elon1":
    {
        "Name":"Elon Musk",
        "Major":"Robotics",
        "Starting_year":2019,
        "Total_Attandance":6,
        "Year":4,
        "Last_Attandance_Time":"2023-01-27 00:54:34"
    },
    "jef1":
    {
        "Name":"jef bezos",
        "Major":"E-commerce",
        "Starting_year":2020,
        "Total_Attandance":10,
        "Year":3,
        "Last_Attandance_Time":"2023-01-25 00:52:44"
    },
    "mark2":
    {
        "Name":"mark zugerberg",
        "Major":"E-commerce",
        "Starting_year":2019,
        "Total_Attandance":15,
        "Year":4,
        "Last_Attandance_Time":"2022-01-27 00:54:38"
    },
    "sunil":
    {
        "Name":"Sunil Giri",
        "Major":"Computer Application",
        "Starting_year":2021,
        "Total_Attandance":20,
        "Year":3,
        "Last_Attandance_Time":"2022-01-24 00:24:54"
    },
    "gaurav":
    {
        "Name":"Gaurav Bohra",
        "Major":"CSE",
        "Starting_year":2020,
        "Total_Attandance":0,
        "Year":4,
        "Last_Attandance_Time":"2022-01-24 00:24:54"
    },
    "bishal":
    {
        "Name":"Bishal Kumar Yadav",
        "Major":"CSE",
        "Starting_year":2020,
        "Total_Attandance":0,
        "Year":4,
        "Last_Attandance_Time":"2022-01-24 00:24:54"
    },
    "bikash":
    {
        "Name":"Bikash Kumar",
        "Major":"CSE",
        "Starting_year":2020,
        "Total_Attandance":0,
        "Year":4,
        "Last_Attandance_Time":"2022-01-24 00:24:54"
    },
    "gautam":
    {
        "Name":"Gautam Kumar",
        "Major":"CSE",
        "Starting_year":2020,
        "Total_Attandance":0,
        "Year":4,
        "Last_Attandance_Time":"2022-01-24 00:24:54"
    }
}

for key,value in data.items():
    ref.child(key).set(value)