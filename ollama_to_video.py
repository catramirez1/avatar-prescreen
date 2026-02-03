import subprocess
from pathlib import Path

PROJECT = Path.home() / "research" / "avatar-prescreen"
SADTALKER = PROJECT / "SadTalker"
FACE = PROJECT / "assets" / "faces" / "testimage.jpeg"     # change if needed
AUDIO = PROJECT / "assets" / "audio" / "llama_reply.wav"
OUTDIR = PROJECT / "output"

cmd = [
    "python", "inference.py",
    "--driven_audio", str(AUDIO),
    "--source_image", str(FACE),
    "--result_dir", str(OUTDIR)
]

subprocess.run(cmd, cwd=str(SADTALKER), check=True)
subprocess.run(["open", str(OUTDIR)], check=False)
print("Output folder:", OUTDIR)
