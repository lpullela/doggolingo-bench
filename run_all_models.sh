#!/bin/bash

# model name parameter (first argument passed to the script)
model_name=$1

# Check if model_name was provided
if [ -z "$model_name" ]; then
  echo "Error: No model name provided."
  echo "Usage: ./run_prompt_llm.sh <model_name>"
  exit 1
fi

# Run the Python script with "mini" argument
echo "Running $model_name with mini argument..."
python3 $model_name mini

# Run the Python script with "regular" argument
echo "Running $model_name with regular argument..."
python3 $model_name regular

# Run the Python script with "llama" argument
echo "Running $model_name with llama argument..."
python3 $model_name llama

echo "All tasks completed."

# run like this: ./run_all_models.sh prompt_llm2.py