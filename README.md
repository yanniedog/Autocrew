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
python3 crewai-autocrew.py [overall_goal] [options]
```

### Options

- `overall_goal`: Define the main objective for the agent crew.
- `-a`, `--auto_run`: Option to automatically run the generated script.
- `-m`, `--multiple NUM`: Create multiple scripts for a single goal.
- `-r`, `--ranking`: Rank agent teams based on provided CSV data.

## Example Usage

### Basic Command

```bash
python3 crewai-autocrew.py "Space Exploration"
```

This command generates a script for a crew with the goal of space exploration.

### Creating Multiple Scripts

```bash
python3 crewai-autocrew.py "Urban Development" -m 3
```

Generates three different scripts for the goal of urban development.

### Automatic Execution

```bash
python3 crewai-autocrew.py "Deep Sea Research" -a
```

Automatically runs the generated script for the goal of deep sea research.

### Ranking Existing Crews

```bash
python3 crewai-autocrew.py "Environmental Cleanup" -r
```

Ranks existing crews based on the goal of environmental cleanup.

## Contributing

Contributions are welcome. Please fork the repository and submit pull requests for any enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: Regular updates are made to this script. Ensure you have the latest version for optimal functionality.
