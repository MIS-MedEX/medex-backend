# API1

```
route: /api/patient
response:
[
    {
        "age": int,
        "name": str,
        "id": int,
        "sex": int
    },
    {
        "age": int,
        "name": str,
        "id": int,
        "sex": int
    },
    ...
]
```

![Postman測試](API1_test.png)

# API2

```
route: /api/patient/<id>
request: id
response:
{
    "age": int,
    "date": [
        {date:[time, time, ...]},
        {date:[time, time, ...]},
        {date:[time, time, ...]}]
    "name": str,
    "id": int,
    "sex": int
}
```

![Postman測試](API2_test.png)

# API3

```
route: /api/patient/<id>/image
request: id, date, type
response:
{
    "img_label": str,
    "img_org_path": str,
    "report": str
    "res_our_cardio":{
        prob:float,
        path:str
    },
    "res_our_pneumo":{
        prob:float,
        path:str
    },
    "res_our_pleural":{
        prob:float,
        path:str
    },
    "res_baseline_cardio":{
        prob:float
    },
    "res_baseline_pneumo":{
        prob:float
    },
    "res_baseline_pleural":{
        prob:float
    },
}
```

<!--![Postman測試](API1_test.png)-->
