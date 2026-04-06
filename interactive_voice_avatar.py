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
ONTOLOGY_PATH = PROJECT / "ontology.json"

# Whisper.cpp (offline STT)
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
FACE_IMAGE = ASSETS / "faces" / "testimage.jpeg"

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

MAKE_VIDEO = False   # set False if SadTalker is too slow


# -------------------------------------------------
# AUDIO RECORDING
# -------------------------------------------------

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


# -------------------------------------------------
# WHISPER TRANSCRIPTION
# -------------------------------------------------

def transcribe_whisper(wav_path: Path) -> str:
    base = wav_path.with_suffix("")
    txt_path = Path(str(base) + ".txt")

    if txt_path.exists():
        txt_path.unlink()

    cmd = [
        WHISPER_BIN,
        "-m", str(WHISPER_MODEL),
        "-f", str(wav_path),
        "-otxt",
        "-of", str(base),
        "-nt",
        "-np",
    ]
    subprocess.run(cmd, check=True)

    if not txt_path.exists():
        raise RuntimeError(f"Whisper output not found: {txt_path}")

    return txt_path.read_text(encoding="utf-8").strip()


# -------------------------------------------------
# OLLAMA CALL
# -------------------------------------------------

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
        raise RuntimeError(f"Ollama did not return JSON:\n{content}")

    return json.loads(content[start:end+1])


# -------------------------------------------------
# PIPER TTS
# -------------------------------------------------

def piper_tts(text: str, out_wav: Path):
    cmd = [
        "python", "-m", "piper",
        "--model", str(PIPER_MODEL),
        "--config", str(PIPER_CONFIG),
        "--output_file", str(out_wav),
    ]
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)


# -------------------------------------------------
# SADTALKER VIDEO
# -------------------------------------------------

# def sadtalker_video(driven_audio: Path, source_image: Path, out_dir: Path):
  #  cmd = [
      #  "python", "inference.py",
        # "--driven_audio", str(driven_audio),
        # "--source_image", str(source_image),
        # "--result_dir", str(out_dir)
   # ]
    #subprocess.run(cmd, cwd=str(SADTALKER_DIR), check=True)
from live_avatar import play_and_animate


# -------------------------------------------------
# ONTOLOGY LOADER
# -------------------------------------------------

def load_ontology():
    if not ONTOLOGY_PATH.exists():
        raise FileNotFoundError(f"Ontology file missing: {ONTOLOGY_PATH}")
    return json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ontology = load_ontology()
    print("📚 Loaded ontology:", list(ontology.keys()))

    intake = {}
    turn = 0
    max_turns = 10
    last_question = None

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

    messages = [{"role": "system", "content": SYSTEM}]

    # ---- SYSTEM TALKS FIRST ----
    greeting = (
        "Hello! I’m going to ask you a few quick questions to help the nurse prepare for your visit. "
        "What brings you in today?"
    )

    messages.append({"role": "assistant", "content": greeting})

    avatar_wav = AUDIO_DIR / "avatar_0.wav"
    print("🔊 Generating greeting audio...")
    piper_tts(greeting, avatar_wav)

    print("🔊 Playing greeting audio + live animation...")
    play_and_animate(avatar_wav, FACE_IMAGE)


    # ---- MAIN LOOP ----
    while True:
        if turn > max_turns:
            print("✅ Reached max turns. Ending session.")
            (OUTPUT_DIR / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            break

        mic_wav = AUDIO_DIR / f"mic_{turn}.wav"
        record_until_silence(mic_wav)

        user_text = transcribe_whisper(mic_wav)
        print(f"📝 Transcribed: {user_text}")

        if not user_text.strip():
            print("Heard nothing. Try speaking a bit louder/closer to the mic.")
            continue

        messages.append({"role": "user", "content": user_text})

        reply = ask_ollama(messages)
        intake = reply.get("intake", intake)

        assistant_text = reply.get("assistant_text", "").strip()
        next_question = reply.get("next_question", "").strip()

        # fallback if model repeats or fails
        if (not next_question) or (last_question and next_question.lower() == last_question.lower()):
            if fallback_i < len(fallback_questions):
                next_question = fallback_questions[fallback_i]
                fallback_i += 1
            else:
                next_question = "Is there anything else you think the nurse should know?"

        required = ["chief_complaint", "duration", "severity_1_to_10", "allergies"]
        if all(str(intake.get(k, "")).strip() for k in required):
            reply["done"] = True

        spoken = (assistant_text + " " + next_question).strip()
        if not spoken:
            spoken = "Thanks. Can you tell me more about your symptoms?"

        messages.append({"role": "assistant", "content": spoken})
        last_question = next_question

        avatar_wav = AUDIO_DIR / f"avatar_{turn}.wav"
        print("🔊 Generating reply audio...")
        piper_tts(spoken, avatar_wav)

        print("🔊 Playing reply audio...")
        play_and_animate(avatar_wav, FACE_IMAGE)

        if MAKE_VIDEO:
            print("🎥 Generating reply video...")
            play_and_animate(avatar_wav, FACE_IMAGE)

        print("✅ Intake so far:\n", json.dumps(intake, indent=2))

        if bool(reply.get("safety_flag", False)):
            print("⚠️ Safety flag true — stopping.")
            (OUTPUT_DIR / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            break

        if bool(reply.get("done", False)):
            print("✅ Done. Saving nurse handoff.")
            (OUTPUT_DIR / "final_intake.json").write_text(json.dumps(intake, indent=2), encoding="utf-8")
            print("🩺 Nurse handoff saved to output/final_intake.json")
            break

        turn += 1


if __name__ == "__main__":
    main()
