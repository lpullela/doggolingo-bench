import csv
import numpy as np
import pandas as pd

print('here')
doggo_dict = pd.read_csv('/home/smirrashidi/doggolingo-bench/data/final_doggolingo_dict.csv')
words = doggo_dict['word'].tolist()

reddit_data = pd.read_csv('../old_csvs/misspelled_words.csv')
sentences = reddit_data["context"].tolist()

res = []

for word in words: 
    print(f'Now searching for: {word}')
    for sentence in sentences:
        if word in sentence:
            res.append({
                'word' : word,
                'sentence' : sentence
})

sentences_df = pd.DataFrame(res).drop_duplicates().sort_values(by=["word"])
sentences_df.to_csv('slow_words_with_sentences.csv',  index=False)