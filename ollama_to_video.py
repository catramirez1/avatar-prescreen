import subprocess
import uuid
import os

RUN_ID = str(uuid.uuid4())

LOCAL_RUN = f"runs/{RUN_ID}"
REMOTE_RUN = f"~/avatar_runs/{RUN_ID}"

def submit_to_hpc(audio_path, image_path):

    os.makedirs(f"{LOCAL_RUN}/input", exist_ok=True)

    # copy inputs locally into run folder
    subprocess.run(["cp", audio_path, f"{LOCAL_RUN}/input/audio.wav"])
    subprocess.run(["cp", image_path, f"{LOCAL_RUN}/input/face.png"])

    # create remote folder
    subprocess.run([
        "ssh", "kean-hpc",
        f"mkdir -p {REMOTE_RUN}/input {REMOTE_RUN}/output"
    ])

    # upload inputs
    subprocess.run([
        "rsync", "-av",
        f"{LOCAL_RUN}/input/",
        f"kean-hpc:{REMOTE_RUN}/input/"
    ])

    # submit slurm job
    subprocess.run([
        "ssh", "kean-hpc",
        f"cd {REMOTE_RUN} && sbatch run_sadtalker.sbatch"
    ])

    # wait loop
    while True:
        result = subprocess.run(
            ["ssh", "kean-hpc", f"test -f {REMOTE_RUN}/output/DONE && echo done"],
            capture_output=True, text=True
        )
        if "done" in result.stdout:
            break

    # download result
    subprocess.run([
        "rsync", "-av",
        f"kean-hpc:{REMOTE_RUN}/output/",
        f"{LOCAL_RUN}/output/"
    ])

    return f"{LOCAL_RUN}/output/video.mp4"
