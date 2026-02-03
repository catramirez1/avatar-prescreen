import json
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

SYSTEM = """
You are a nurse pre-screening intake assistant.
Do NOT diagnose. Ask concise intake questions only.
Return ONLY valid JSON with keys:
{
  "assistant_text": string,
  "next_question": string,
  "intake": object,
  "done": boolean,
  "safety_flag": boolean
}
"""

state = {"symptoms": None, "duration": None, "severity": None}

payload = {
    "model": MODEL,
    "stream": False,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Current intake state: {json.dumps(state)}"},
        {"role": "user", "content": "Patient said: I've had a headache for two days."}
    ]
}

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()
content = r.json()["message"]["content"].strip()

# parse JSON even if model adds extra text
start, end = content.find("{"), content.rfind("}")
out = json.loads(content[start:end+1])

print(json.dumps(out, indent=2))
