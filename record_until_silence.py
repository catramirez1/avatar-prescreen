import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wavwrite
from pathlib import Path

def record_until_silence(
    out_wav: Path,
    samplerate: int = 16000,
    max_seconds: int = 20,
    start_threshold: float = 0.015,   # how loud to start recording
    stop_threshold: float = 0.010,    # how quiet counts as silence
    silence_seconds: float = 1.2,     # stop after this much silence
    chunk_ms: int = 30
):
    chunk_samples = int(samplerate * chunk_ms / 1000)
    max_chunks = int(max_seconds * 1000 / chunk_ms)
    silence_chunks_needed = int(silence_seconds * 1000 / chunk_ms)

    audio_chunks = []
    started = False
    silent_chunks = 0

    print("\n🎙️ Speak whenever you're ready... (auto-stops after silence)")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk[:, 0]  # mono
            rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)

            if not started:
                if rms >= start_threshold:
                    started = True
                    audio_chunks.append(chunk.copy())
                    # small pre-roll feel is nice, but optional; we skip for simplicity
                continue

            audio_chunks.append(chunk.copy())

            if rms < stop_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= silence_chunks_needed:
                break

    if not audio_chunks:
        raise RuntimeError("No speech detected. Try again and speak a bit louder/closer to the mic.")

    audio = np.concatenate(audio_chunks)
    # Convert float32 [-1,1] to int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    wavwrite(str(out_wav), samplerate, audio_int16)
    print(f"✅ Saved: {out_wav}  ({len(audio)/samplerate:.2f}s)")

if __name__ == "__main__":
    out = Path("assets/audio/mic_silence.wav")
    record_until_silence(out)
