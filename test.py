import requests
import json

def pull_llama2():
    url = "http://localhost:11434/api/pull"
    data = {
        "name": "llama2"
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json.dumps(data), headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success:", response.json())  # Assuming JSON response, parse it.
    else:
        print("Error:", response.status_code, response.text)

if __name__ == '__main__':
    pull_llama2()