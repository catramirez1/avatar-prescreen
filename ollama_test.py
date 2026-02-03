import requests

url = "http://localhost:11434/api/chat"
payload = {
  "model": "llama3.2",
  "stream": False,
  "messages": [
    {"role": "system", "content": "You are a clinic pre-screening intake assistant. Ask ONE short intake question."},
    {"role": "user", "content": "I have a sore throat and a fever."}
  ]
}

r = requests.post(url, json=payload, timeout=120)
r.raise_for_status()
print(r.json()["message"]["content"])
