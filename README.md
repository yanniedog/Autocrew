# CrewAI Autocrew Script

This script automates the process of creating a CrewAI team with agents, tasks, and tools, using the Ollama language model to generate the required data in CSV format. The script then parses the CSV data, defines agents and tasks, and writes a CrewAI script that can be executed to run the generated team.

## Features

- Initialize Ollama with a specified model (default is 'openhermes').
- Get agent data from Ollama as a CSV response.
- Save the CSV output to a file.
- Parse the CSV data and extract agent information.
- Define agents and tasks for the CrewAI script.
- Write the CrewAI script based on the agent and task data.
- Option to run the generated CrewAI script automatically.

## Requirements

- Python 3.6 or higher
- `langchain_community` package
- `crewai` package

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yanniedog/crewai-autocrew.git
   ```

2. Change to the project directory:

   ```
   cd crewai-autocrew
   ```

3. Install the required packages:

   ```
   pip install langchain_community crewai
   ```

## Usage

1. Run the script with the overall goal as an argument:

   ```
   python autocrew.py "Find the best pizza restaurants in New York"
   ```

   Alternatively, you can run the script without an argument and provide the overall goal when prompted:

   ```
   python autocrew.py
   ```

2. The script will generate a CSV file with agent data and a CrewAI script based on the parsed agent data.

3. To run the generated CrewAI script automatically, use the `-a` or `--autorun` option:

   ```
   python autocrew.py -a "Find the best pizza restaurants in New York"
   ```

## Customization

You can modify the script to use different models, tools, or processes for the CrewAI team. Simply update the relevant parts of the code to reflect your desired configuration.

## Contributing

1. Fork the repository on GitHub.
2. Create a new branch for your changes.
3. Commit your changes and push them to your fork.
4. Create a pull request with a description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
