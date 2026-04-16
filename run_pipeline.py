import os
import time

USERNAME = "ramicath"   
HPC_HOST = "login.kean-hpc.kean.edu"
REMOTE_PATH = "~/research/avatar-prescreen"

AUDIO_FILE = "assets/audio/llama_reply.wav"
IMAGE_FILE = "assets/faces/testimage.jpeg"
OUTPUT_FILE = "result.mp4"

print("🚀 Starting pipeline...")

# 1. Upload files
print("📤 Uploading files...")
os.system(f"scp {AUDIO_FILE} {USERNAME}@{HPC_HOST}:{REMOTE_PATH}/assets/audio/")
os.system(f"scp {IMAGE_FILE} {USERNAME}@{HPC_HOST}:{REMOTE_PATH}/assets/faces/")

# 2. Submit job
print("🖥️ Submitting job...")
submit_cmd = f'ssh {USERNAME}@{HPC_HOST} "cd {REMOTE_PATH} && sbatch hpc_job.sbatch"'
job_output = os.popen(submit_cmd).read()
print(job_output)

# Get job ID
print(job_output)

if "Submitted batch job" not in job_output:
    print("❌ Job submission failed. Check sbatch file.")
    exit()

job_id = job_output.strip().split()[-1]
print(f"📌 Job ID: {job_id}")

# 3. Wait for job
print("⏳ Waiting...")
while True:
    check_cmd = f'ssh {USERNAME}@{HPC_HOST} "squeue -j {job_id}"'
    result = os.popen(check_cmd).read()

    if job_id not in result:
        break

    print("...still running...")
    time.sleep(10)

print("✅ Job finished!")

# Download latest video
REMOTE_BASE = "/home/ramicath/research/avatar-prescreen"

print("📥 Downloading latest video...")

# Get newest mp4 directly (not folder)
get_latest_cmd = f'ssh {USERNAME}@{HPC_HOST} "ls -t {REMOTE_BASE}/output/*.mp4 | head -n 1"'
latest_file = os.popen(get_latest_cmd).read().strip()

print(f"🎬 Latest file: {latest_file}")

# Download it
os.system(f'scp "{USERNAME}@{HPC_HOST}:{latest_file}" .')

# Rename locally
os.system("ls -t *.mp4 | head -n 1 | xargs -I {} mv {} latest.mp4")

print("🎉 Saved as latest.mp4")
os.system("open latest.mp4")