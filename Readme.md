# API1

```
method: GET
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
method: GET
route: /api/patient/<id>
request: id
response:
{
    "age": int,
    "date": {
        date:[time, time, ...],
        date:[time, time, ...],
        date:[time, time, ...]}
    "name": str,
    "id": int,
    "sex": int
}
```

![Postman測試](API2_test.png)

# API3

```
method: GET
route: /api/patient/<id>/image
request: id, date, type
response:
{
    "highlight": [str, str, str],
    "img_label": str,
    "image_base64": str,
    "img_org_path": str,
    "pred_label": str/list
    "report": str,
    "res_our_cardio":{
        error: float,
        prob:float,
        vis_path:str,
        vis_base64: str
    },
    "res_our_pneumo":{
        error: float,
        prob:float,
        vis_path:str,
        vis_base64: str
    },
    "res_our_pleural":{
        error: float,
        prob:float,
        vis_path:str,
        vis_base64: str
    },
    "res_baseline_cardio":{
        error: float,
        prob:float
    },
    "res_baseline_pneumo":{
        error: float,
        prob:float
    },
    "res_baseline_pleural":{
        error: float,
        prob:float
    },
}
```

![Postman測試](API3_test.png)

# API4

```
method: POST
route: /api/patient/save_report
request:
{
    "id": int,
    "date": str,
    "type": str,
    "report": str,
    "highlight": [str, str, str]
}
response:
{
    "status":success/fail
}
```

![Postman測試](API4_test.png)
