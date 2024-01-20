# Autocrew

![Autocrew_logo](./docs/logo.png)
# Autocrew: Automated CrewAI Script Generation and Ranking

![Autocrew Logo](https://example.com/autocrew_logo.png)

Autocrew is a Python script designed to simplify the process of generating scripts for CrewAI, a collaborative language AI platform. It automates the creation of CrewAI agent and task scripts based on your specified overall goal, and it can also rank crews based on their suitability for completing a given task.

## Features

- **Script Generation:** Autocrew can generate CrewAI agent and task scripts for a specified overall goal. It communicates with the Ollama language model to generate agent details, including roles, goals, backstories, assigned tasks, and delegation abilities. The generated scripts are ready for execution within CrewAI.

- **Ranking Mode:** Autocrew can rank existing crews based on their suitability for a particular task. It collects CSV data files that represent different crews, communicates with Ollama to analyze and rank them, and provides a summary of the ranking.

- **Multiple Script Generation:** You can specify the number of scripts to generate for the same overall goal, allowing you to create multiple versions of your crew for comparison or experimentation.

- **Automatic Execution:** Autocrew can automatically execute the generated scripts within CrewAI, saving you time and effort.

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

## License

Autocrew is licensed under the [MIT License](LICENSE).

## Acknowledgments

Autocrew was created by [Your Name] and is based on the [CrewAI](https://example.com/crewai) platform.

## Contact

For any questions or issues, please contact [Your Email Address].

**Disclaimer:** Autocrew is not affiliated with or endorsed by CrewAI or any other third-party services mentioned in this script. It is provided as an open-source project for script generation and ranking purposes.

- [CrewAI](https://www.crewai.com): The AI platform that Autocrew is currently compatible with.
- [Autogen](https://www.autogen.com): A potential future alternative to CrewAI.
