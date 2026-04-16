import os
import json
import time
from pathlib import Path

PROJECT = Path.home() / "research" / "avatar-prescreen"
STATE_FILE = PROJECT / "state.json"

print("🚀 Starting interactive avatar...\n")

while True:

    # ---------------- STEP 1: RECORD ----------------
    print("🎤 Step 1: Recording audio...")
    os.system("python record_until_silence.py")

    # ---------------- STEP 2: TRANSCRIBE ----------------
    print("📝 Step 2: Transcribing...")
    os.system("python transcribe.py")  # make sure this creates transcript.txt

    # ---------------- STEP 3: LLM + TTS ----------------
    print("🧠 Step 3: Processing with LLM + generating speech...")
    os.system("python ollama_to_piper.py")

    # ---------------- STEP 4: VIDEO ----------------
    print("🎥 Step 4: Generating video...")
    os.system("python run_pipeline.py")

    # ---------------- STEP 5: CHECK STATE ----------------
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())

        print("\n📊 Current state:")
        print(json.dumps(state, indent=2))

        # End conversation if done
        if state.get("done"):
            print("\n✅ Intake complete. Ending conversation.")
            break

    # ---------------- STEP 6: LOOP CONTROL ----------------
    user_input = input("\n🔁 Continue conversation? (y/n): ").strip().lower()
    if user_input == "n":
        print("👋 Exiting pipeline.")
        break

    print("\n----------------------------------------\n")
    time.sleep(1)