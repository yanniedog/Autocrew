# Autocrew: Where AI Builds Itself

Autocrew is a Python script designed to simplify the process of generating scripts for [CrewAI](https://github.com/joaomdmoura/crewAI), a collaborative language AI platform. It automates the creation of CrewAI agent and task scripts based on your specified overall goal, and it can also rank crews based on their suitability for completing a given task.

![Autocrew_logo](./docs/autocrew_logo.png)

[Autocrew Discord channel](https://discord.gg/ZGxdjVSPA3)

## Key Features

- **Script Generation**: Autocrew can generate CrewAI agent and task scripts for a specified overall goal. It communicates with the Ollama language model to generate agent details, including roles, goals, backstories, assigned tasks, and delegation abilities. The generated scripts are ready for execution within CrewAI.

- **Ranking Mode**: Autocrew can rank existing crews based on their suitability for a particular task. It collects CSV data files that represent different crews, communicates with Ollama to analyze and rank them, and provides a summary of the ranking.

- **Multiple Script Generation**: You can specify the number of scripts to generate for the same overall goal, allowing you to create multiple versions of your crew for comparison or experimentation.

- **Automatic Execution**: Autocrew can automatically execute the generated scripts within CrewAI, saving you time and effort.

## Prerequisites

Before using Autocrew, make sure you have the following prerequisites installed:

- Python 3.x
- Required Python packages (install using `pip`):
  - `argparse`
  - `csv`
  - `io`
  - `json`
  - `requests`
  - `crewai`
  - `langchain_community`
- [Ollama](Ollama.ai)
  - NOTE: Ollama is currently available for Linux and Mac only
  - In order to run Ollama on windows, you need to install the Ubuntu 20.04 distribution within WSL. Please see [this guide](https://www.jeremymorgan.com/blog/generative-ai/how-to-run-llm-local-windows/) for more information.

## Getting Started

To get started with Autocrew, you'll need to clone the Git repository to your local machine. Follow these steps:

1. Open your terminal or command prompt.

2. Clone the Autocrew repository from GitHub using the following command:

```bash
git clone https://github.com/yanniedog/autocrew.git
```

3. Once the repository is cloned, you can navigate into the Autocrew directory:

```bash
cd autocrew
```

Now you have successfully cloned the Autocrew repository to your local machine, and you can start using and contributing to the project.

If you encounter any issues, have questions, or want to contribute to Autocrew, please refer to the "Contributing" section in this README for guidelines on making changes and submitting pull requests.

## Usage

### Script Generation

To generate CrewAI scripts for a specified overall goal, use the following command:

```bash
python3 autocrew.py <overall_goal> [-a] [-m NUM] [-v]
```

- `<overall_goal>`: Specify the overall goal for your crew.
- `-a` (optional): Automatically run the generated script(s) after generation.
- `-m NUM` (optional): Generate NUM number of scripts for the same overall goal.
- `-v` (optional): Enable verbose output for detailed information.

### Ranking Mode

To rank existing crews based on their suitability for a task, use the following command:

```bash
python3 autocrew.py -r <overall_goal> [-v]
```

- `-r`: Activate ranking mode.
- `<overall_goal>`: Specify the overall goal for ranking.
- `-v` (optional): Enable verbose output for detailed information.

## Example

Here's an example of generating CrewAI scripts for a project management task:

```bash
python3 autocrew.py "Project Management" -a -m 3 -v
```

This command generates three CrewAI scripts for the "Project Management" goal, automatically runs them, and provides detailed output.

## Contributing

If you would like to contribute to Autocrew, please fork the repository, make your changes, and submit a pull request. We welcome contributions and suggestions.

## Community

Join the [Autocrew Discord channel](https://discord.gg/ZGxdjVSPA3)
## License

Autocrew is licensed under the [MIT License](LICENSE).

## Acknowledgments

Autocrew was created by [Yanniedog](https://github.com/yanniedog). It was initially built to interact with the [CrewAI](https://github.com/joaomdmoura/crewAI) platform, but compatibility with other platforms (such as [Autogen](https://microsoft.github.io/autogen/)) is currently in development.
