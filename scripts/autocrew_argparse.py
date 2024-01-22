# autocrew_argparse.py

import argparse
from autocrew_core import main

def parse_arguments():
    parser = argparse.ArgumentParser(description='CrewAI Autocrew Script')
    parser.add_argument('overall_goal', nargs='?', type=str, help='The overall goal for the crew')
    parser.add_argument('-a', '--auto_run', action='store_true', help='Automatically run the generated script')
    parser.add_argument('-m', '--multiple', type=int, metavar='NUM', help='Create NUM number of CrewAI scripts for the same overall goal. Example: -m 3')
    parser.add_argument('-r', '--ranking', action='store_true', help='Perform ranking only based on existing CSV files --> currently EXPERIMENTAL')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--use_ollama_host', action='store_true', help='Use OLLAMA_HOST from the original script in the generated script')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
