# CrewAI AutoCrew Script

## Introduction
Welcome to the CrewAI AutoCrew Script, an innovative Python tool designed to automate the creation and evaluation of virtual agent teams. Integrating with Ollama for AI-driven decision-making, this script streamlines processes within the CrewAI framework, making it an indispensable resource for developers and researchers in AI and machine learning.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Versioning and Updates](#versioning-and-updates)

## Features
- **Agent Team Creation**: Automates generation of agent teams with specified roles and tasks.
- **Integration with Ollama and CrewAI**: Leverages Ollama for decision-making and CrewAI for agent management.
- **CSV Data Management**: Facilitates agent data handling in CSV format.
- **Multiple Script Generation**: Supports creation of various scripts for different objectives.
- **Team Ranking Functionality**: Evaluates and ranks agent teams based on effectiveness and goal alignment.

## Prerequisites
- Python 3.x
- OpenAI API key (for Ollama interactions)
- Basic knowledge of Python and command-line operations.

## Installation
1. Clone the repository to your local machine.
2. Install Python 3.x if not already installed: [Python Installation Guide](https://www.python.org/downloads/).
3. Obtain an OpenAI API key from [OpenAI](https://openai.com/).
4. Install necessary dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Execute the script in a terminal as follows:
```
python3 crewai-autocrew.py [options] "overall_goal"
```

### Options
- `"overall_goal"`: Main objective for the agent crew (in quotes).
- `-a`: Automatically run the generated script.
- `-m[NUM]`: Create multiple crews for the same goal. Replace `[NUM]` with the number required.
- `-r`: (*experimental*): Rank crews generated with the "-m" option.

## Examples
### Basic Command
```
python3 crewai-autocrew.py "create a smartphone app with a Voice Chatbot for scam calls"
```

### Automatic Execution
```
python3 crewai-autocrew.py "Summarise the latest tech news" -a
```

### Multiple Scripts
```
python3 crewai-autocrew.py "Develop a handheld quantum computer" -m3
```

### Ranking Crews
```
python3 crewai-autocrew.py "Environmental Cleanup" -r
```

## Troubleshooting
For common issues, refer to the [Troubleshooting Guide](Troubleshooting.md).

## FAQ
Answers to frequently asked questions can be found in the [FAQ section](FAQ.md).

## Contributing
Contributions are welcome. Please fork the repository and submit pull requests for enhancements.

## License
This project is under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to all contributors and users of the CrewAI community.

## Versioning and Updates
Regular updates are made to this script. Check the [Releases](https://github.com/yourrepository/crewai-autocrew/releases) page for the latest version.

---

**Disclaimer**: This script is updated regularly. Ensure you're using the latest version for optimal functionality.

---
