import json 
from Flask import requests

url = 'http://127.0.0.1:8006/model'
request_data = json.dumps({'model': 'Logistic Regression'})
my_response = requests.post(url, request_data)
print(my_response.text)


