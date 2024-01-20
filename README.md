# Autocrew

![Autocrew_logo](./docs/logo.png)

Autocrew is a standalone project that serves as a front-end application for CrewAI. It's designed to automate the process of generating scripts for CrewAI. Although it's currently compatible with CrewAI, future plans include expanding Autocrew to be compatible with other alternatives such as Autogen. Please note that Autocrew is not affiliated with CrewAI.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Command Line Parameters](#command-line-parameters)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

Before you install Autocrew, ensure you have Python 3.7 or later installed on your system.

To install Autocrew, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yanniedog/autocrew.git
   ```
2. Navigate to the project directory:
   ```
   cd autocrew
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To use Autocrew, run the main script `autocrew.py` with your desired command-line parameters. For example:

```
python3 autocrew.py "Save the world"
```

## Command Line Parameters

Autocrew supports various command-line parameters to customize its behavior:

- `overall_goal`: The overall goal for the crew (required).
- `-a`, `--auto_run`: Automatically run the generated script.
- `-m`, `--multiple`: Create a specified number of CrewAI scripts for the same overall goal. Example: `-m 3`.
- `-r`, `--ranking`: Perform ranking only based on existing CSV files (currently experimental).
- `-v`, `--verbose`: Enable verbose output.

## Examples

Here are some examples of how to use Autocrew:

- Generate a single script with an overall goal of "Save the world":
  ```
  python3 autocrew.py "Save the world"
  ```
- Generate three scripts with the same overall goal and automatically run them:
  ```
  python3 autocrew.py "Save the world" -m 3 -a
  ```
- Perform ranking based on existing CSV files for a given overall goal:
  ```
  python3 autocrew.py "Save the world" -r
  ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or create an Issue.

## License

Autocrew is released under the [MIT License](https://opensource.org/licenses/MIT).

## Resources

- [CrewAI](https://www.crewai.com): The AI platform that Autocrew is currently compatible with.
- [Autogen](https://www.autogen.com): A potential future alternative to CrewAI.
