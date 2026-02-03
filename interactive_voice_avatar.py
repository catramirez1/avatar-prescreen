import json
import subprocess
from pathlib import Path
import requests

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wavwrite

PROJECT = Path.home() / "research" / "avatar-prescreen"
ASSETS = PROJECT / "assets"
AUDIO_DIR = ASSETS / "audio"
OUTPUT_DIR = PROJECT / "output"

# Whisper.cpp (offline STT) via Homebrew whisper-cpp formula
WHISPER_BIN = "whisper-cli"
WHISPER_MODEL = PROJECT / "models" / "whisper" / "ggml-base.en.bin"

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"

# Piper
PIPER_MODEL = ASSETS / "voices" / "en_US-amy-medium.onnx"
PIPER_CONFIG = ASSETS / "voices" / "en_US-amy-medium.onnx.json"


# SadTalker
SADTALKER_DIR = PROJECT / "SadTalker"
FACE_IMAGE = ASSETS / "faces" / "testimage.jpeg"  # <-- change if needed

STATE_PATH = PROJECT / "state.json"

SYSTEM = """
You are a nurse pre-screening intake assistant for a clinic kiosk.

Goal:
- Collect structured intake info and ask ONE question at a time.
- Do NOT diagnose.

You must:
- Look at the existing intake object.
- Ask the most relevant NEXT question to fill missing fields.
- Avoid repeating a question that was just asked.

Emergency rule:
If severe chest pain, trouble breathing, stroke signs, severe allergic reaction, or suicidal intent are mentioned:
- set safety_flag=true
- say: "Please seek immediate medical attention / call emergency services."

You MUST return ONLY valid JSON in this exact schema:
{
  "assistant_text": "brief acknowledgement (1 short sentence max)",
  "next_question": "one question only",
  "intake": {
    "chief_complaint": "",
    "duration": "",
    "severity_1_to_10": "",
    "associated_symptoms": "",
    "medications_tried": "",
    "allergies": "",
    "medical_history": ""
  },
  "done": false,
  "safety_flag": false
}

Set done=true ONLY when these are filled:
chief_complaint, duration, severity_1_to_10, allergies
"""

# Toggle video generation (SadTalker is slow). Keep False for smooth demos.
MAKE_VIDEO = True


def record_until_silence(
    out_wav: Path,
    samplerate: int = 16000,
    max_seconds: int = 20,
    start_threshold: float = 0.015,
    stop_threshold: float = 0.010,
    silence_seconds: float = 1.2,
    chunk_ms: int = 30
):
    chunk_samples = int(samplerate * chunk_ms / 1000)
    max_chunks = int(max_seconds * 1000 / chunk_ms)
    silence_chunks_needed = int(silence_seconds * 1000 / chunk_ms)

    audio_chunks = []
    started = False
    silent_chunks = 0

    print("\n🎙️ Speak now... (auto-stops after silence)")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk[:, 0]
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)

            if not started:
                if rms >= start_threshold:
                    started = True
                    audio_chunks.append(chunk.copy())
                continue

            audio_chunks.append(chunk.copy())

            if rms < stop_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= silence_chunks_needed:
                break

    if not audio_chunks:
        raise RuntimeError("No speech detected. Speak louder/closer or lower the start_threshold.")

    audio = np.concatenate(audio_chunks)
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavwrite(str(out_wav), samplerate, audio_int16)
    print(f"✅ Saved mic audio: {out_wav}")


def transcribe_whisper(wav_path: Path) -> str:
    base = wav_path.with_suffix("")          # e.g., assets/audio/mic_1
    txt_path = Path(str(base) + ".txt")      # e.g., assets/audio/mic_1.txt
    if txt_path.exists():
        txt_path.unlink()

    cmd = [
        WHISPER_BIN,
        "-m", str(WHISPER_MODEL),
        "-f", str(wav_path),
        "-otxt",
        "-of", str(base),
        "-nt",   # no timestamps (cleaner)
        "-np",   # fewer prints
    ]
    subprocess.run(cmd, check=True)

    if not txt_path.exists():
        raise RuntimeError(f"Whisper output not found: {txt_path}")

    return txt_path.read_text(encoding="utf-8").strip()


def ask_ollama(messages: list[dict]) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()

    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"Ollama did not return JSON. Raw output:\n{content}")

    return json.loads(content[start:end+1])

    # Parse JSON even if it wraps it with text
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"Ollama did not return JSON. Raw output:\n{content}")

    return json.loads(content[start:end+1])


def piper_tts(text: str, out_wav: Path):
    cmd = [
        "python", "-m", "piper",
        "--model", str(PIPER_MODEL),
        "--config", str(PIPER_CONFIG),
        "--output_file", str(out_wav),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


def sadtalker_video(driven_audio: Path, source_image: Path, out_dir: Path):
    cmd = [
        "python", "inference.py",
        "--driven_audio", str(driven_audio),
        "--source_image", str(source_image),
        "--result_dir", str(out_dir)
    ]
    subprocess.run(cmd, cwd=str(SADTALKER_DIR), check=True)


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # AUTO-RESET EVERY RUN
    intake = {}
    turn = 0
    max_turns = 10

    # Track last asked question to prevent repetition
    last_question = None

    # A simple fallback sequence if the LLM repeats
    fallback_questions = [
        "What brings you in today?",
        "How long have you been experiencing this?",
        "On a scale of 1 to 10, how severe is it?",
        "Are you having any other symptoms along with that?",
        "Have you taken any medication or tried anything for it?",
        "Do you have any allergies to medications or foods?",
        "Do you have any relevant medical history we should know about?",
    ]
    fallback_i = 0

    # Conversation memory for Ollama
    messages = [{"role": "system", "content": SYSTEM}]

    # -----------------------------
    # SYSTEM TALKS FIRST
    # -----------------------------
    greeting_question = "Hello! I’m going to ask you a few quick questions to help the nurse prepare for your visit. What brings you in today?"
    messages.append({"role": "assistant", "content": greeting_question})

    avatar_wav = AUDIO_DIR / "avatar_0.wav"
    print("🔊 Generating greeting audio...")
    piper_tts(greeting_question, avatar_wav)

    print("🔊 Playing greeting audio...")
    subprocess.run(["afplay", str(avatar_wav)], check=False)

    if MAKE_VIDEO:
        print("🎥 Generating greeting video...")
        sadtalker_video(avatar_wav, FACE_IMAGE, OUTPUT_DIR)
        open_latest_video(OUTPUT_DIR)

    last_question = "What brings you in today?"
    turn = 1

    # -----------------------------
    # LOOP
    # -----------------------------
    while True:
        if turn > max_turns:
            print("✅ Reached max turns. Ending session.")
            (PROJECT / "output" / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            break

        mic_wav = AUDIO_DIR / f"mic_{turn}.wav"
        record_until_silence(mic_wav)

        user_text = transcribe_whisper(mic_wav)
        print(f"📝 Transcribed: {user_text}")

        if not user_text.strip():
            print("Heard nothing. Try speaking a bit louder/closer to the mic.")
            continue

        messages.append({"role": "user", "content": user_text})

        # Ask LLM for next step
        reply = ask_ollama(messages)

        # Update intake
        intake = reply.get("intake", intake)

        assistant_text = reply.get("assistant_text", "").strip()
        next_question = reply.get("next_question", "").strip()

        # If the model repeats or gives nothing, fallback
        if (not next_question) or (last_question and next_question.lower() == last_question.lower()):
            if fallback_i < len(fallback_questions):
                next_question = fallback_questions[fallback_i]
                fallback_i += 1
            else:
                next_question = "Thank you. Is there anything else you think the nurse should know today?"

        # Stop criteria (required fields)
        required = ["chief_complaint", "duration", "severity_1_to_10", "allergies"]
        if all(str(intake.get(k, "")).strip() for k in required):
            reply["done"] = True

        # Speak out loud
        spoken = (assistant_text + " " + next_question).strip()
        if not spoken:
            spoken = "Thanks. Can you tell me more about your symptoms?"

        messages.append({"role": "assistant", "content": spoken})
        last_question = next_question

        avatar_wav = AUDIO_DIR / f"avatar_{turn}.wav"
        print("🔊 Generating reply audio...")
        piper_tts(spoken, avatar_wav)

        print("🔊 Playing reply audio...")
        subprocess.run(["afplay", str(avatar_wav)], check=False)

        if MAKE_VIDEO:
            print("🎥 Generating reply video...")
            sadtalker_video(avatar_wav, FACE_IMAGE, OUTPUT_DIR)
            open_latest_video(OUTPUT_DIR)

        print("✅ Intake so far:\n", json.dumps(intake, indent=2))

        # Stop conditions
        if bool(reply.get("safety_flag", False)):
            print("⚠️ Safety flag true — stopping flow.")
            (PROJECT / "output" / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            break

        if bool(reply.get("done", False)):
            print("✅ Done. Saving nurse handoff.")
            (PROJECT / "output" / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            print("🩺 Nurse handoff saved to output/final_intake.json")
            break

        turn += 1


if __name__ == "__main__":
    main()
