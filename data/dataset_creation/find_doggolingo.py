import os
import pandas as pd
import time
from functools import lru_cache
from langchain_openai import ChatOpenAI

MAX_TOKEN_LENGTH = 512
BATCH_SIZE = 10

with open("../../../api_key2.txt") as f:
    OPENAI_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

TASK_PROMPT = "You will be given a word sourced from the internet. Your task is to decide if this word might be considered 'doggolingo' or not. \n \
        The definition of doggolingo is as follows: DoggoLingo is an Internet language that is created from word conversion, meme lexicon, \
        and onomatopoeia. Emerging in the 2010s, DoggoLingo is implied to be a dog's own idiom, and is presented as a canine's \
        thought process. \n Words that fall into this category may include the following: \n \
            1) puns related to dogs, cats, or other pet activities, such as 'pupset' (pup & upset, meaning 'upset') or fur-real (fur & real, meaning 'for real') \n \
            2) cutesy ways to say pet related words or emotions, such as 'snoozles', 'woofer', or 'angery' \n \
            3) portmanteaus such as petcation (pet & vacation) or doggolicious (doggo & delicious) \n \
        This list is not exhaustive and you may encounter more examples of doggolingo not listed here.\
        Words that would not be considered doggolingo would be typos that do not seem related to pet content or words that have no discernable meaning related to pets. \
        If you believe that the word is doggolingo, respond with TRUE. Otherwise, respond with FALSE. \n" 

def query_gpt4_batch(words):
    model_name = "gpt-4"
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=MAX_TOKEN_LENGTH)
    
    batched_prompt = TASK_PROMPT + "\n".join([f"The word for this task is {word}." for word in words])
    response = llm.invoke(batched_prompt)
    
    return response.content.strip().split("\n")

@lru_cache(maxsize=500)
def cached_query(word):
    return query_gpt4_batch([word])[0]

def process_words_sequentially(words):
    results = []
    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i:i + BATCH_SIZE]
        responses = query_gpt4_batch(batch)
        for word, response in zip(batch, responses):
            results.append({"word": word, "response": response})
            print({"word": word, "response": response})
        time.sleep(1)  
    return results

if __name__ == "__main__":
    print("Loading csv file")
    misspelled_words = pd.read_csv('misspelled_words.csv')
    all_words = misspelled_words['misspelled_word'].dropna().tolist()

    print("processing words")
    results = process_words_sequentially(all_words)

    results_df = pd.DataFrame(results)
    results_df.to_csv('gpt_doggolingo.csv', index=False)
    print("saved new words to csv")








