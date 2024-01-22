# Filename: check_latest_version.py

import requests

def check_latest_version():
    try:
        response = requests.get('https://raw.githubusercontent.com/yanniedog/autocrew/main/autocrew.py')
        response.raise_for_status()
        script_content = response.text
        version_line = next(line for line in script_content.split('\n') if line.startswith('autocrew_version = '))
        latest_version = version_line.split('=')[1].strip().strip('"')

        if version.parse(latest_version) > version.parse(autocrew_version):
            return latest_version
        else:
            return None

    except Exception as e:
        print(f'Error checking the latest version: {e}')
        return None
