# SmartGearMLmodel
The Smart Gear backend uses an Isolation Forest model to predict the behaviour of the users transactions. It predicts whether the users current is different from all thier previous transactions. This feature can be modified to detect fraud purchases by scammers.

# API_URL
The model's api was created using fastapi and is hosted on https://smartgearmlmodel.onrender.com


# Features
The model is trained using the amount, hour of the purchase, the day of the week the purchase was made and the type of transaction that was made from the transaction table in the database. The model takes these attributes as input and makes predictions based on the input. 
- amount (decimal)
- hour (integer)
- dayofweek (integer)
- transaction_type_encoded (integer)

# Endpoint
| Method | Endpoint                  | Description                             |
|--------|---------------------------|-----------------------------------------|
| POST   | `/predict`                | Sends a request to the ML model to make a prediction  | 

# Output
The model gives a prediction whether the transaction was normal (1) or an anomaly (-1).

Example Input:
```
{
  "values": [
    5030, 12, 3, 2 
  ]
}
```

Expected Response:
```
{
  "Preditction": [
    1
  ]
}
```