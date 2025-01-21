import os
import pandas as pd
import argparse
import json
from vllm import LLM, SamplingParams
from langchain_openai import ChatOpenAI

MAX_TOKEN_LENGTH = 512

# Load API key
with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

os.environ['HF_HOME'] = '/data/kenantang/llama'
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

# Initialize Llama model
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=2, max_model_len=MAX_TOKEN_LENGTH)
sampling_params = SamplingParams(temperature=0.5, max_tokens=MAX_TOKEN_LENGTH, seed=0)

def query_gpt4_with_lookup(model_type, query, use_rag=False, definition=""):
    if model_type == "mini":
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4"

    llm = ChatOpenAI(model=model_name, temperature=0, max_completion_tokens=MAX_TOKEN_LENGTH)

    if use_rag and definition:
        query = f"Using the true definition of the word:\n{definition}\n\n{query}"

    response = llm.invoke(query)
    return response.content

def query_llama_directly(query, use_rag=False, definition=""):
    if use_rag and definition:
        query += f"\nDefinition: {definition}"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]

    outputs = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

def evaluate_responses_with_gpt4o(word, results_df, prompts, definition):
    """
    Evaluate the responses using regular GPT-4 as the evaluator.

    Parameters:
        word (str): The word being evaluated.
        results_df (pd.DataFrame): DataFrame with results to evaluate.
        prompts (dict): Prompts used for querying models.

    Returns:
        dict: Updated DataFrame columns with ratings and evaluation comments.
    """
    # read the json file eval_criteria.json 
    with open('eval_criteria.json', 'r') as f:
        eval_criteria = json.load(f)

    evaluator_model = ChatOpenAI(model="gpt-4o", temperature=0)
    ratings = {}
    ratings_full = {}

    for prompt_type in prompts.keys():
        if prompt_type == "Translate": 
            evaluation_query = (
                f"You are an AI agent whose task is to grade a response to a task. "
                f"The task is to: {prompts[prompt_type]} {word}. "
                f"You will rate the response on the following scale: BAD, OKAY, GOOD, EXCELLENT "
                f"based on these criteria: {eval_criteria[prompt_type]['criteria']} "
                f"The true definition of the word is: {definition}. "
                f"The sentences to translate are: {results_df.loc[results_df['word'] == word, 'Create'].iloc[0]} "
                f"The translated responses are: {results_df.loc[results_df['word'] == word, prompt_type].iloc[0]} "
            )
        else:
            evaluation_query = (
                f"You are an AI agent whose task is to grade a response to a task. "
                f"The task is to: {prompts[prompt_type]} {word}. You will rate the response on the following scale: "
                f"BAD, OKAY, GOOD, EXCELLENT based on these criteria: {eval_criteria[prompt_type]['criteria']} "
                f"The true definition of the word is: {definition}. The response is: {results_df.loc[results_df['word'] == word, prompt_type].iloc[0]}."
            )

        # Evaluate using GPT-4
        evaluation_response = evaluator_model.invoke(evaluation_query)

        rating_words = ["excellent", "good", "okay", "bad"]
        matched_ratings = [word.upper() for word in rating_words if word in evaluation_response.content.lower()]

        if len(matched_ratings) == 1:
            ratings[prompt_type] = matched_ratings[0].upper() # upper case
        else:
            ratings[prompt_type] = "unknown"

        ratings_full[prompt_type] = evaluation_response.content

    return ratings, ratings_full

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run queries on different models.")
    parser.add_argument("model", choices=["regular", "mini", "llama"], help="Model type to use: regular, mini, or llama")
    args = parser.parse_args()

    # Load the DoggoLingo dictionary
    df = pd.read_csv("data/doggolingo_dict.csv")
    results_df = pd.DataFrame()

    # Prompts
    prompts = {
        "Interpret": "Interpret the doggolingo word: ",
        "Create": "Create sentences containing the doggolingo word: ",
        "Translate": "Translate the following doggolingo sentences to professional English: ",
        "Generate": "Generate new words similar to the doggolingo word: ",
    }

    # Iterate over words and query models
    for _, row in df.iterrows():
        word = row["word"]
        definition = row["definition"]

        responses = {}
        for prompt_type, prompt_text in prompts.items():
            if prompt_type == "Translate":
                query = prompt_text + responses.get("Create", "")
            else:
                query = prompt_text + word

            if args.model in ["regular", "mini"]:
                response = query_gpt4_with_lookup(
                    model_type=args.model, query=query, use_rag=False, definition=definition
                )
                print(response)
            elif args.model == "llama":
                response = query_llama_directly(query=query, use_rag=False, definition=definition)
                print(response)

            responses[prompt_type] = response

        entry = {"word": word, "definition": definition, "model_name": args.model}
        entry.update(responses)
        results_df = pd.concat([results_df, pd.DataFrame([entry])], ignore_index=True)
        # break

    # Evaluate responses with GPT-4
    for _, row in df.iterrows():
        word = row["word"]
        ratings, ratings_full = evaluate_responses_with_gpt4o(word, results_df, prompts, row['definition'])

        for prompt_type, rating in ratings.items():
            column_name = f"{prompt_type}_rating"
            full_rating_column = f"{prompt_type}_full_rating"

            if column_name not in results_df.columns:
                results_df[column_name] = None
            if full_rating_column not in results_df.columns:
                results_df[full_rating_column] = None

            results_df.loc[results_df["word"] == word, column_name] = rating
            results_df.loc[results_df["word"] == word, full_rating_column] = ratings_full[prompt_type]
        #break   

    # Save the responses to a CSV file
    output_file = f"responses_output_prompt_with_eval_criteria_{args.model}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Responses saved to {output_file}")

# run this command
# python3 prompt_llm.py llama ( or mini or regular )