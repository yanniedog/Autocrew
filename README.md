# Autocrew: Where AI Builds Itself (Beginner Friendly Guide)
## Introduction
Welcome to AutoCrew! This application uses advanced language models like Ollama and OpenAI to create automated workflows and manage data. It's designed to be user-friendly, even if you're not a tech expert.

![Autocrew_logo](./docs/autocrew_logo.png)

[Autocrew Discord channel](https://discord.gg/ZGxdjVSPA3)

## Setting up Your Environment

### Step 1: Install Windows Subsystem for Linux (WSL)
AutoCrew runs on Ubuntu, so Windows users will need to set up WSL. Here's how:

1. **Open PowerShell as Administrator**: Search for PowerShell in your Start menu, right-click it, and select "Run as administrator".
2. **Install WSL**: In the PowerShell window, type the following command and press Enter: 
   ``` 
   wsl --install
   ```
   This will install WSL with the default Ubuntu distribution.
3. **Restart Your Computer**: Once the installation is complete, restart your computer.

### Step 2: Setting Up Ubuntu
After your computer restarts, follow these steps:

1. **Open Ubuntu**: Search for Ubuntu in your Start menu and open it. The first launch will take a few minutes as it completes the setup.
2. **Create a User Account**: You'll be prompted to create a username and a password. Remember these, as you'll need them for accessing Ubuntu.

### Step 3: Install Python on Ubuntu
AutoCrew requires Python. Here's how to install it:

1. **Update Ubuntu**: In your Ubuntu window, type:
   ```
   sudo apt update && sudo apt upgrade
   ```
   Enter your password when prompted. This updates the software sources and gets the latest software packages.
2. **Install Python**: Next, install Python by typing:
   ```
   sudo apt install python3
   ```

## Downloading and Running AutoCrew

### Step 1: Download AutoCrew
1. **Clone the AutoCrew Repository**: In Ubuntu, use the `git clone` command to download AutoCrew:
   ```
   git clone https://github.com/yanniedog/autocrew.git
   ```
2. **Navigate to the AutoCrew Directory**: Once the download is complete, navigate to the AutoCrew directory:
   ```
   cd autocrew
   ```

### Step 2: Run AutoCrew
1. **Start AutoCrew**: In the same Ubuntu window, type:
   ```
   python3 autocrew.py
   ```
2. **Follow the On-Screen Instructions**: AutoCrew will guide you through the rest!

## AutoCrew and CrewAI

AutoCrew is tightly integrated with CrewAI, which is a platform for automating tasks and generating scripts using advanced language models. Here's how they connect:

- **CrewAI Integration**: AutoCrew leverages CrewAI's capabilities to generate data-driven scripts and automate tasks.
- **Advanced Language Models**: CrewAI utilizes powerful language models like Ollama and OpenAI, which AutoCrew seamlessly integrates into its workflow.
- **User-Friendly Interface**: AutoCrew provides an easy-to-use interface to interact with CrewAI's features, making it accessible to users with minimal tech knowledge.

## What Can You Do with AutoCrew?
AutoCrew is designed to be a friendly assistant in creating automated workflows. Here's what you can do:

- **Generate Data-Driven Scripts**: AutoCrew can create scripts based on specific goals you provide.
- **Automate Tasks**: It can automate various tasks using advanced language models.
- **Customize Workflows**: Even if you're not tech-savvy, AutoCrew guides you in customizing workflows to suit your needs.
- **Easy Interaction**: With simple commands, interact with AutoCrew to perform complex operations.
- **No Tech Expertise Needed**: AutoCrew is designed for ease of use, regardless of your tech background.

## Conclusion
AutoCrew is your go-to tool for automating tasks and managing data in an easy-to-use environment. Whether you're a tech newbie or an enthusiast, AutoCrew makes your work simpler and more efficient.
