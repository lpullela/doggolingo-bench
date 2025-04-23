import pandas as pd
import os 
import json 
from datasets import load_dataset

# load harmful queries
ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
harmful_sentences = ds['harmful']['Goal']

# load doggolingo dataset 
doggo_dict = pd.read_csv("../data/final_doggolingo_dict.csv")
doggo_df = pd.DataFrame(doggo_dict).drop(columns=["definition"])
cols = ["equivalent", "word"]
doggo_df = doggo_df[cols]

# create word mappings
word_dict = dict(doggo_df.values)

def mapper(sent, word_dict):
    s = ' '
    word_list = sent.split(s)
    for i in range(len(word_list)):
        replace = word_dict.get(word_list[i])
        if replace:
            word_list[i] = replace
    return s.join(word_list)

sent_series = pd.Series(harmful_sentences)

translated_sentences = sent_series.apply(mapper, args=(word_dict,))

data = {'original_harmful': harmful_sentences, 'doggolingo_harmful': translated_sentences}
final_df = pd.DataFrame(data = data)

final_df.to_csv("harmful_doggolingo_wbw.csv", index=False)