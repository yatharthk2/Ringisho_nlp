import requests
API_TOKEN = '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'

API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-hate"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({"inputs": "I like you. I love you"})
print(output)