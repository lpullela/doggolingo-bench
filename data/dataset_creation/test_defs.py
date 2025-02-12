import os
import pandas as pd
import time
import csv
import json
from functools import lru_cache
from langchain_openai import ChatOpenAI

MAX_TOKEN_LENGTH = 1000
BATCH_SIZE = 7

with open("../../../api_key2.txt") as f:
    OPENAI_KEY = f.read().strip()

csv_file = "gpt_define_test.csv"
    
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

TASK_PROMPT = "You will be given a word sourced from the internet that is a form of doggolingo. The definition of \
doggolingo is as follows: DoggoLingo is an Internet language that is created from word conversion, meme lexicon, \
and onomatopoeia. Emerging in the 2010s, DoggoLingo is implied to be a dog's own idiom, and is presented as a canine's \
thought process. \n Words that fall into this category may include the following: \n \
    1) puns related to dogs, cats, or other pet activities, such as 'pupset' (pup & upset, meaning 'upset') or fur-real (fur & real, meaning 'for real') \n \
    2) cutesy ways to say pet related words or emotions, such as 'snoozles', 'woofer', or 'angery' \n \
    3) portmanteaus such as petcation (pet & vacation) or doggolicious (doggo & delicious) \n \
This list is not exhaustive and you may encounter more examples of doggolingo not listed here. \n \
Your task is to define the doggolingo word, and to provide an equivalent word in traditional English. Here are some examples: \n \
    1) word: 'doin', definition: 'Popular expression in DoggoLingo that is a rudimentary way of creating verb phrases.', equivalent: 'doing' \n \
    2) word: 'thicc', definition: 'Based on the word thick, thicc is used affectionately to describe doggos who are chunky, pudgy, or slightly overweight.' \
equivalent: 'thick' \n \
    3) word: 'petcation', definition: 'The term 'petcation' in doggolingo refers to a vacation or getaway that includes pets, particularly dogs. It's a play on \
the words 'pet' and 'vacation,' suggesting a trip where pets are not only allowed but are a central part of the experience', equivalent: 'vacation' \n \
    4) word: 'cronch', definition: 'In doggolingo, the word 'cronch' is often used to describe the sound or action of a dog chewing on something, typically something \
crunchy like a bone or a treat. It's an onomatopoeic term that captures the satisfying noise and experience of a dog enjoying a good chew.', equivalent: 'crunch' \n \
In your defintions, aim to capture the meaning of the word, how it is formed (onomatopoeia, pun, etc), and the context it may be used in. \n \
Please give your response for the given worse following this json formatting: \n \
[{\"word\": \"thicc\", \"definition\": \" Based on the word thick, thicc is used affectionately to describe doggos who are chunky, pudgy, or slightly overweight.\" \
\"equivalent\": \"thick\"},\n \
{\"word\": \"petcation\", \"definition\": \"The term 'petcation' in doggolingo refers to a vacation or getaway that includes pets, particularly dogs. It's a play on \
the words 'pet' and 'vacation,' suggesting a trip where pets are not only allowed but are a central part of the experience\", \"equivalent\": \"vacation\"}] \n \
Do not use markdown. Only provide the json data as shown above."



def query_gpt4_batch(words):
    model_name = "gpt-4o"
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=MAX_TOKEN_LENGTH)
    
    batched_prompt = TASK_PROMPT + "\n\n" + "\n\n".join([f"Word {i+1}: {word}" for i, word in enumerate(words)])
    response = llm.invoke(batched_prompt)
    print(response)
    parsed_response = json.loads(response.content.strip())
        
    return parsed_response 
@lru_cache(maxsize=1000)
def cached_query(word):
    return query_gpt4_batch([word])[0]

def process_words_sequentially(words, csv_file):
    results = []
    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i:i + BATCH_SIZE]
        responses = query_gpt4_batch(batch)
        
        with open(csv_file, 'a', newline ='') as f:
            fieldnames = ["word", "definition", "equivalent"]
            writer = csv.DictWriter(f, fieldnames = fieldnames) 

            if f.tell() == 0:
                writer.writeheader()

            if isinstance(responses, dict):
                    writer.writerow(responses)

            elif isinstance(responses, list):
                writer.writerows(responses)
        time.sleep(1)  
    return results

if __name__ == "__main__":
    print("Loading csv file")
    gpt_data = pd.read_csv('cleaned_gpt_id.csv')
    # sample = gpt_data.sample(n=30)
    all_words = gpt_data['word'].dropna().tolist()

    print("Defining words")
    results = process_words_sequentially(all_words, csv_file)

    results_df = pd.DataFrame(results)
    print(f'Saved words and definitions to {csv_file}')