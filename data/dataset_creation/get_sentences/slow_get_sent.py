import re
import numpy as np
import pandas as pd

def create_list(text):
    text = text.lower()
    pattern = r"[ \t\n\r\f\v,./\-|+!?*&|#'$()^%[\]@+=~`{}:;<>\"“”‘’]"

    result = re.split(pattern, text)
    result = [word for word in result if word]

    return result

doggo_dict = pd.read_csv('/home/smirrashidi/doggolingo-bench/data/final_doggolingo_dict.csv')
words = doggo_dict['word'].tolist()

reddit_data = pd.read_csv('../old_csvs/misspelled_words.csv')
sentences = reddit_data["context"].tolist()

res = []

for word in words: 
    print(f'Now searching for: {word}')
    for sentence in sentences:
        sent_list = create_list(sentence)
        if word in sent_list:
            res.append({
                'word' : word,
                'sentence' : sentence
})

sentences_df = pd.DataFrame(res).drop_duplicates().sort_values(by=["word"])
sentences_df.to_csv('slow_words_with_sentences.csv',  index=False)