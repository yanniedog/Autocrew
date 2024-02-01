import subprocess
import time
import re

# Define the port to create a tunnel to
port = 80

# Start an Ngrok tunnel to the specified port and capture the output
ngrok_process = subprocess.Popen(["ngrok", "http", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Wait for a few seconds to ensure the tunnel is established
time.sleep(5)

# Try to find the Ngrok URL from the process output
ngrok_url = None
if ngrok_process.stdout:
    for line in ngrok_process.stdout:
        match = re.search(r"https://[0-9a-z-]+\.ngrok.io", line.decode("utf-8"))
        if match:
            ngrok_url = match.group(0)
            break

if ngrok_url:
    print("Ngrok URL:", ngrok_url)
else:
    print("Failed to obtain Ngrok URL")

try:
    # Keep the script running to keep the tunnel open
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Terminate the Ngrok process when you stop the script
    ngrok_process.terminate()
