import requests
from bs4 import BeautifulSoup
import re

# Function to scrape and list URLs
def scrape_and_list_urls(base_url):
    try:
        print("Requesting the webpage...")
        response = requests.get(base_url)
        response.raise_for_status()
        print("Webpage accessed successfully.")

        soup = BeautifulSoup(response.content, 'html.parser')
        all_links = soup.find_all('a', href=True)
        print(f"Total links found: {len(all_links)}")

        # Filtering links that start with "/library/"
        library_links = [a['href'] for a in all_links if a['href'].startswith('/library/')]
        print(f"Filtered links: {len(library_links)}")

        # Removing the "/library/" part from each link and sorting
        trimmed_links = sorted([link.replace('/library/', '') for link in library_links])
        print("Trimmed and sorted links:")

        # Displaying the filtered and trimmed links
        for idx, link in enumerate(trimmed_links, 1):
            print(f"{idx}. {link}")

        # User selection
        choice = int(input("Enter the number of the link you want to choose: "))
        if 1 <= choice <= len(trimmed_links):
            selected_model = trimmed_links[choice - 1]
            print(f"You selected: {selected_model}")

            # Formulate the URL in the specified format
            model_url = f"{base_url}{selected_model}/tags"
            print(f"Formulated URL: {model_url}")

            # Scraping the model URL
            ollama_run_strings = scrape_ollama_run_strings(model_url)

            # User selection of ollama_library_model_and_quant
            if ollama_run_strings:
                ollama_model_to_pull = select_ollama_run_string(ollama_run_strings)
                if ollama_model_to_pull:
                    print(f"You selected: {ollama_model_to_pull}")
                    # Store the selected value for later use in other scripts or functions
                else:
                    print("Invalid choice. Please enter a number from the list.")
            else:
                print("No 'ollama run' strings found on the model page.")
        else:
            print("Invalid choice. Please enter a number from the list.")
    
    except requests.RequestException as e:
        print(f"Error occurred: {e}")

# Function to scrape ollama run strings
def scrape_ollama_run_strings(model_url):
    try:
        print("Requesting the model webpage...")
        response = requests.get(model_url)
        response.raise_for_status()
        print("Model webpage accessed successfully.")

        # Extracting and parsing the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Finding all input elements with the class 'command'
        command_inputs = soup.find_all('input', class_='command')

        # Extracting the 'value' attribute from each input element
        ollama_run_strings = [input_element['value'] for input_element in command_inputs]

        # Sorting the strings alphabetically
        ollama_run_strings = sorted(set(ollama_run_strings))  # Using set to remove duplicates

        return ollama_run_strings
    
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return []

# Function to select ollama_library_model_and_quant
def select_ollama_run_string(ollama_run_strings):
    if not ollama_run_strings:
        return None
    
    # Printing the sorted strings as a numbered list
    print("Text strings starting with 'ollama run [ollama_library_model_and_quant]':")
    for idx, run_string in enumerate(ollama_run_strings, 1):
        print(f"{idx}. {run_string}")

    # User selection
    try:
        choice = int(input("Enter the number of the string you want to choose: "))
        if 1 <= choice <= len(ollama_run_strings):
            selected_value = ollama_run_strings[choice - 1]
            return selected_value
        else:
            print("Invalid choice. Please enter a number from the list.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return None

# Ensure that the base_url is correct and that you have permission to scrape the website
base_url = "https://ollama.ai/library/"
scrape_and_list_urls(base_url)