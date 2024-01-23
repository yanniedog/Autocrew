#!/bin/bash
# setup.sh

# Function to install langchain-community
install_langchain_community() {
    echo "Installing langchain-community..."
    pip install -U langchain-community
    if [ $? -eq 0 ]; then
        echo "langchain-community installed successfully."
    else
        echo "Failed to install langchain-community."
        exit 1
    fi
}

# Function to update import statements in Python script
update_imports_in_python_script() {
    local script_path="$1"
    echo "Updating import statements in $script_path..."
    sed -i 's/from langchain.callbacks import CallbackManager, StreamingStdoutCallbackHandler/from langchain_community.callbacks import CallbackManager, StreamingStdoutCallbackHandler/g' "$script_path"
    if [ $? -eq 0 ]; then
        echo "Import statements updated successfully in $script_path."
    else
        echo "Failed to update import statements in $script_path."
        exit 1
    fi
}

# Main execution
install_langchain_community

# Update the autocrew.py script
autocrew_script_path="/home/ai/autocrew/autocrew.py"
update_imports_in_python_script "$autocrew_script_path"

# Uncomment and modify the line below if you have other scripts to update
# update_imports_in_python_script "/path/to/other/script.py"

echo "Setup completed successfully."
