import re
import numpy as np
import pandas as pd

doggo_dict = pd.read_csv('/home/smirrashidi/doggolingo-bench/data/final_doggolingo_dict.csv')
words = doggo_dict['word'].tolist()

reddit_data = pd.read_csv('../old_csvs/misspelled_words.csv')

pattern = r'\b(' + '|'.join(map(re.escape, words)) + r')\b'

reddit_data['matched_word'] = reddit_data['context'].str.extract(pattern, expand=False)

reddit_data = reddit_data.dropna(subset="matched_word")
reddit_data = reddit_data.drop_duplicates(['context', 'matched_word'])

final_df = reddit_data.drop(['misspelled_word', 'corpus'], axis=1)
final_df = final_df.sort_values(by=['matched_word'])
final_df.to_csv('words_with_sentences.csv', index=False)