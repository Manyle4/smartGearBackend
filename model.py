import pandas as pd 
import random
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

#Generating fake data
num_samples = 500
users = [f"user{i}" for i in range(1, 11)]
transaction_types = ["card", "bank_account", "mobile_money"]
statuses = ["completed", "pending", "failed"]

#Random timestamp generator
def random_date_generator():
    start = datetime(2024, 1, 1)
    end = datetime(2025, 6, 30)
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

data = []
for i in range(num_samples):
    data.append({
        "transaction_id": i,
        "owner": random.choice(users),
        "amount": round(random.uniform(5.0, 10000.0), 2),
        "transaction_type": random.choice(transaction_types),
        "status": random.choice(statuses),
        "timestamp": random_date_generator()
    })

df = pd.DataFrame(data)

#Processing the data to use to train the model
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

encoder = LabelEncoder()
df["transaction_type_encoded"] = encoder.fit_transform(df["transaction_type"])

#Try and add new features to imporve the effectiveness of the random data
#Like frequency and average amount

#Training the model
x = df[["amount", "hour", "dayofweek", "transaction_type_encoded"]]

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(x)

df["prediction"] = model.predict(x)

df["avg_amount"] = df.groupby("owner")["amount"].transform("mean")

anomaly_count = (df["prediction"] == -1).sum()

joblib.dump(model, 'model.pkl')