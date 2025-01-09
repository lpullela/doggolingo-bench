import os
import pandas as pd
import argparse
from vllm import LLM, SamplingParams
from langchain_openai import ChatOpenAI

# Load API key
with open("api_key.txt") as f:
    OPENAI_KEY = f.read().strip()
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

os.environ['HF_HOME'] = '/data/kenantang/llama'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# Initialize Llama model
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=2, max_model_len=4096)
sampling_params = SamplingParams(temperature=0.5, max_tokens=1024, seed=0)

def query_gpt4_with_lookup(model_type, query, use_rag=False, definition=""):
    if model_type == "mini":
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4"

    llm = ChatOpenAI(model=model_name, temperature=0, max_completion_tokens=1024)

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
    evaluator_model = ChatOpenAI(model="gpt-4o", temperature=0)
    ratings = {}
    ratings_full = {}

    for prompt_type in prompts.keys():
        # Prepare the evaluation query
        query = (
                prompts[prompt_type]
                + ":"
                + word
                + "\n"
                + results_df.loc[results_df["word"] == word, prompt_type].iloc[0]
            )

        prompt = (
            f"Using the true definition of the word:\n{definition}\n\nRate the proceeding content on the following scale: BAD, OKAY, GOOD, EXCELLENT. Content: "
            + query
        )
    
        evaluation_query = f"Rate the following content on a scale of BAD, OKAY, GOOD, EXCELLENT:\n\n{prompt}"

        # Evaluate using GPT-4
        evaluation_response = evaluator_model.invoke(evaluation_query)

        if "excellent" in evaluation_response.content.lower():
            ratings[prompt_type] = "EXCELLENT"
        elif "good" in evaluation_response.content.lower():
            ratings[prompt_type] = "GOOD"
        elif "okay" in evaluation_response.content.lower():
            ratings[prompt_type] = "OKAY"
        elif "bad" in evaluation_response.content.lower():
            ratings[prompt_type] = "BAD"
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
        "Interpret": "Interpret the word: ",
        "Create": "Create sentences containing the word: ",
        "Translate": "Translate the following sentences to professional English: ",
        "Generate": "Generate new words similar to the word: ",
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
    output_file = f"responses_output_{args.model}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Responses saved to {output_file}")

# run this command
# python3 prompt_llm.py llama ( or mini or regular )