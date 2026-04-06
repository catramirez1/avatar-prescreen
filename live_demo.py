import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import requests
import subprocess

# -------- SETTINGS --------
MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/chat"
AUDIO_FILE = "mic_input.wav"
OUTPUT_AUDIO = "assets/audio/llama_reply.wav"

conversation_history = []

VOICE_MODEL = "/Users/catherineramirez/research/avatar-prescreen/en_US-lessac-medium.onnx"
VOICE_CONFIG = "/Users/catherineramirez/research/avatar-prescreen/en_US-lessac-medium.onnx.json"

whisper = WhisperModel("base")

# -------- RECORD AUDIO --------
def record_audio(seconds=5):
    print("🎤 Speak now...")
    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1)
    sd.wait()
    wav.write(AUDIO_FILE, 16000, audio)
    print("✅ Recorded")

# -------- TRANSCRIBE --------
def transcribe():
    segments, _ = whisper.transcribe(AUDIO_FILE)
    text = " ".join([seg.text for seg in segments])
    print("📝 You said:", text)
    return text

# -------- LLM --------
def ask_llm(user_text):
    global conversation_history

    conversation_history.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a clinical intake assistant.\n\n"

                    "Your responsibilities:\n"
                    "- Detect the patient's emotional tone (e.g., anxious, neutral, frustrated)\n"
                    "- Adapt your response tone accordingly\n"
                    "- Show empathy when appropriate\n"
                    "- Reiterate the patient's symptoms briefly\n"
                    "- Ask short, natural follow-up questions\n\n"

                    "Tone adaptation rules:\n"
                    "- If patient sounds anxious or worried → be reassuring and gentle\n"
                    "- If patient is calm → be professional and neutral\n"
                    "- If patient is not concerned → be concise and direct\n\n"

                    "Rules:\n"
                    "- DO NOT diagnose\n"
                    "- DO NOT give medical advice\n"
                    "- DO NOT use bullet points, symbols, or lists\n"
                    "- Keep responses under 2 sentences\n\n"

                    "Always sound human and conversational."
                )
            }
        ] + conversation_history
    }

    r = requests.post(OLLAMA_URL, json=payload)
    response = r.json()["message"]["content"]

    conversation_history.append({"role": "assistant", "content": response})

    print("🤖 AI:", response)
    return response

# -------- TTS --------
def speak(text):
    # clean formatting
    text = text.replace("*", "")
    text = text.replace("#", "")
    
    cmd = [
        "python", "-m", "piper",
        "--model", VOICE_MODEL,
        "--config", VOICE_CONFIG,
        "--output_file", OUTPUT_AUDIO,
    ]
    
    subprocess.run(cmd, input=text.encode(), check=True)

    print("🔊 Playing audio...")
    subprocess.run(["afplay", OUTPUT_AUDIO])
# -------- MAIN LOOP --------
while True:
    record_audio()
    user_text = transcribe()
    
    if "stop" in user_text.lower():
        break

    reply = ask_llm(user_text)
    speak(reply)

    print("📁 Saved response to:", OUTPUT_AUDIO)
    print("----")
