# CrewAI AutoCrew Script

This repository contains the CrewAI AutoCrew Script, a Python tool designed for automating the creation and evaluation of virtual agent teams within the CrewAI framework, integrating with Ollama for enhanced decision-making.

## Features

- **Agent Team Creation**: Automates the generation of agent teams, specifying roles and tasks aligned with overarching goals.
- **Integration with Ollama and CrewAI**: Utilizes Ollama for AI-driven decisions and CrewAI for effective agent management.
- **CSV Data Management**: Supports handling of agent data in CSV format, facilitating data processing and analysis.
- **Multiple Script Generation**: Allows for the creation of various scripts tailored to different objectives.
- **Team Ranking Functionality**: Evaluates and ranks agent teams based on effectiveness and goal alignment.

## Installation

Ensure Python 3.x is installed along with necessary dependencies like CrewAI, Ollama, and others. An OpenAI API key is required for Ollama interactions.

## Usage

Execute the script in a terminal:

```bash
python3 crewai-autocrew.py "overall_goal" options
```

### Options

- `"overall_goal"`: Define the main objective for the agent crew, enclosed within double quotation marks.
- `-a`: Option to automatically run the generated script.
- `-m[NUM]`: Create multiple alternative crews for the same goal. Replace `[NUM]` with the number of alternatives required.
- `-r`: _(* experimental/unstable *)_: Use an LLM to identify the best crew from the alternatives generated by the "-m" parameter.

## Example Usage

### Basic Command

```bash
python3 crewai-autocrew.py "create a smartphone app which uses a Voice Chatbot to hold autonomous conversations with scam callers"
```

This command generates a script for a crew with the goal of "create a smartphone app which uses a Voice Chatbot to hold autonomous conversations with scam callers"

### Automatic Execution

```bash
python3 crewai-autocrew.py "Summarise the latest tech news" **-a**
```

Automatically execute the generated script once it is created
_(note: "-a" cannot be used with "-m" or "-r" currently, but this is under development)_

### Creating Multiple Scripts

```bash
python3 crewai-autocrew.py "Develop an handheld quantum computer for consumers" **-m3**
```

Generates three different scripts for the goal of urban development.

### Ranking Existing Crews

```bash
python3 crewai-autocrew.py "Environmental Cleanup" **-r**
```

Ranks existing crews based on the goal of environmental cleanup.

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: Regular updates are made to this script. Ensure you have the latest version for optimal functionality.
