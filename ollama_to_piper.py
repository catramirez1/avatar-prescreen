import json
import subprocess
from pathlib import Path
import requests

# ---------------- CONFIG ----------------

URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"

PROJECT = Path.home() / "research" / "avatar-prescreen"

VOICE_MODEL = PROJECT / "assets" / "voices" / "en_US-amy-medium.onnx"
VOICE_CONFIG = PROJECT / "assets" / "voices" / "en_US-amy-medium.onnx.json"

OUT_WAV = PROJECT / "assets" / "audio" / "llama_reply.wav"
TRANSCRIPT_FILE = PROJECT / "assets" / "audio" / "transcript.txt"
STATE_FILE = PROJECT / "state.json"

# ---------------- SYSTEM PROMPT ----------------

SYSTEM = """
You are a nurse pre-screening intake assistant.
Do NOT diagnose. Ask concise intake questions only.

Always ask the next missing field:
- symptoms
- duration
- severity

Do not repeat already answered questions.

Return ONLY valid JSON with keys:
{
  "assistant_text": string,
  "next_question": string,
  "intake": object,
  "done": boolean,
  "safety_flag": boolean
}
"""

# ---------------- LOAD STATE ----------------

if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())
else:
    state = {
        "turn": 0,
        "intake": {
            "symptoms": [],
            "duration": None,
            "severity": None
        },
        "next_question": None,
        "done": False
    }

# ---------------- LOAD PATIENT INPUT ----------------

if TRANSCRIPT_FILE.exists():
    patient = TRANSCRIPT_FILE.read_text().strip()
    print(f"📝 Patient said: {patient}")
else:
    print("⚠️ No transcript found, using fallback.")
    patient = "I am not feeling well."

# ---------------- LLM CALL ----------------

payload = {
    "model": MODEL,
    "stream": False,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Current intake state:\n{json.dumps(state, indent=2)}"},
        {"role": "user", "content": f"Patient said: {patient}"}
    ]
}

print("🧠 Sending to LLM...")

r = requests.post(URL, json=payload, timeout=120)
r.raise_for_status()
resp = r.json()

# ---------------- PARSE RESPONSE (SAFE JSON) ----------------

raw_text = resp.get("message", {}).get("content", "").strip()

print("\n📦 RAW LLM OUTPUT:")
print(raw_text)

# Extract JSON even if model adds text
start = raw_text.find("{")
end = raw_text.rfind("}")

if start != -1 and end != -1:
    try:
        data = json.loads(raw_text[start:end+1])
    except Exception:
        print("❌ JSON parse failed.")
        data = {"assistant_text": raw_text}
else:
    print("❌ No JSON found.")
    data = {"assistant_text": raw_text}

# ---------------- FIX INTAKE FORMAT ----------------

# ensure symptoms is always a list
if "intake" in data and "symptoms" in data["intake"]:
    if isinstance(data["intake"]["symptoms"], str):
        data["intake"]["symptoms"] = [data["intake"]["symptoms"]]

# ---------------- UPDATE STATE ----------------

if "intake" in data:
    for key, value in data["intake"].items():
        if isinstance(value, list):
            state["intake"].setdefault(key, [])
            state["intake"][key] = list(set(state["intake"][key] + value))
        else:
            state["intake"][key] = value

state["next_question"] = data.get("next_question")
state["done"] = data.get("done", False)
state["turn"] = state.get("turn", 0) + 1

with open(STATE_FILE, "w") as f:
    json.dump(state, f, indent=2)

# ---------------- USE ASSISTANT TEXT ----------------

text = data.get("assistant_text", "").strip()

# Clean up tone
text = text.replace("Can you describe", "Can you tell me a bit more about")
text = text.replace("Are you experiencing", "Have you been experiencing")
text = text.replace("Please describe", "Can you describe")
text = text.replace("Kindly", "")
text = text.capitalize()

print("\n💬 FINAL ASSISTANT TEXT:")
print(text)

# ---------------- TTS (PIPER) ----------------

print("\n🔊 Generating speech...")

cmd = [
    "piper",
    "--model", str(VOICE_MODEL),
    "--config", str(VOICE_CONFIG),
    "--output_file", str(OUT_WAV)
]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
proc.communicate(text.encode("utf-8"))

print(f"\n✅ Saved audio to: {OUT_WAV}")