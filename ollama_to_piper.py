import json
import subprocess
from pathlib import Path
import requests

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

PROJECT = Path.home() / "research" / "avatar-prescreen"
VOICE_MODEL = PROJECT / "assets" / "voices" / "en_US-amy-medium.onnx"
VOICE_CONFIG = PROJECT / "assets" / "voices" / "en_US-amy-medium.onnx.json"
OUT_WAV = PROJECT / "assets" / "audio" / "llama_reply.wav"

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
patient = "I have a sore throat and a fever."

payload = {
    "model": MODEL,
    "stream": False,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Current intake state: {json.dumps(state)}"},
        {"role": "user", "content": f"Patient said: {patient}"}
    ]
}

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()
data = r.json()
content = data["message"]["content"]
start, end = content.find("{"), content.rfind("}")
out = json.loads(content[start:end+1])

spoken = (out.get("assistant_text","") + " " + out.get("next_question","")).strip()
if not spoken:
    spoken = "Thanks. Can you tell me more about what brings you in today?"

print("LLM says:", spoken)

cmd = [
    "python", "-m", "piper",
    "--model", str(VOICE_MODEL),
    "--config", str(VOICE_CONFIG),
    "--output_file", str(OUT_WAV),
]
subprocess.run(cmd, input=spoken.encode("utf-8"), check=True)

subprocess.run(["afplay", str(OUT_WAV)], check=False)
print("Saved:", OUT_WAV)
