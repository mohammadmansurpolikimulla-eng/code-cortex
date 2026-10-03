import requests
import json
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
payload = {'forecast_date': today, 'is_exam': 0}
print(f'Sending payload: {payload}')

res = requests.post('http://127.0.0.1:5000/api/predict', json=payload)
data = res.json()
print(data.get('success'))
if not data.get('success'):
    print(data.get('error'))
