import whisper
from pathlib import Path

PROJECT = Path.home() / "research" / "avatar-prescreen"

AUDIO_FILE = PROJECT / "assets" / "audio" / "mic_silence.wav"
TRANSCRIPT_FILE = PROJECT / "assets" / "audio" / "transcript.txt"

print("📝 Transcribing audio...")

# Load model (first time downloads it)
model = whisper.load_model("base")

# Transcribe
result = model.transcribe(str(AUDIO_FILE))

text = result["text"].strip()

# Save transcript
with open(TRANSCRIPT_FILE, "w") as f:
    f.write(text)

print("\n📝 TRANSCRIPTION:")
print(text)

print(f"\n✅ Saved to: {TRANSCRIPT_FILE}")
