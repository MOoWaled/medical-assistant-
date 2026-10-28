import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "text": "can’t sleep at night"
}

print("sending request to API...")
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response JSON:")
print(f"condition: {response.json()['medical_details']['condition']}")
print("-"*120)
print(f"warnings: {response.json()['medical_details']['warnings']}")
print("-"*120)
print(f"causes: {response.json()['medical_details']['causes']}")
print("-"*120)
print(f"recommendations: {response.json()['medical_details']['recommendations']}")
print("-"*80)