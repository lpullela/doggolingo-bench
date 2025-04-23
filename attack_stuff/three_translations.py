import pandas as pd
import os 
import json 
import random


def mapper(sent, word_dict):
    s = ' '
    word_list = sent.split(s)
    for i in range(len(word_list)):
        replace = word_dict.get(word_list[i])
        if replace:
            word_list[i] = replace
    return s.join(word_list)

def mapper_50(sent, word_dict):
    words = sent.split()
    indices = list(range(len(words)))
    random.shuffle(indices)
    num_to_replace = len(indices) // 2

    for i in indices[:num_to_replace]:
        replacement = word_dict.get(words[i])
        if replacement:
            words[i] = replacement

    return ' '.join(words)

if __name__ == "__main__":
    # read in harmful sentences
    harmful_sentences = pd.read_csv("hate_speech.csv")

    # load doggolingo dataset 
    doggo_dict = pd.read_csv("../data/final_doggolingo_dict.csv")
    doggo_df = pd.DataFrame(doggo_dict).drop(columns=["definition"])
    cols = ["equivalent", "word"]
    doggo_df = doggo_df[cols]

    # load additional words
    additional = pd.read_json("./augmented_dict.json")
    additional_df = pd.DataFrame(additional)
    additional_df = additional_df[cols]

    # merge two datasets
    combined = pd.concat([doggo_df, additional_df], ignore_index=True)

    # create word mappings
    orig_word_dict = dict(doggo_df.values)
    combined_word_dict = dict(combined.values)
    sent_series = pd.Series(harmful_sentences["clean_text"])[:100]

    # create "5%" translations
    translated_sentences = sent_series.apply(mapper, args=(orig_word_dict,))

    data = {'original_harmful': sent_series, 'doggolingo_harmful': translated_sentences}
    percent_5_final_df = pd.DataFrame(data = data)

    percent_5_final_df.to_csv("percent_5_harmful_doggolingo.csv", index=False)

    # create 50% translations
    percent_50_translated = sent_series.apply(mapper_50, args=(combined_word_dict,))
    data = {'original_harmful': sent_series, 'doggolingo_harmful': percent_50_translated}
    percent_50_final_df = pd.DataFrame(data = data)

    percent_50_final_df.to_csv("percent_50_harmful_doggolingo.csv", index=False)

    # create most intense translations
    intense_translated = sent_series.apply(mapper, args=(combined_word_dict,))
    data = {'original_harmful': sent_series, 'doggolingo_harmful': intense_translated}
    intense_final_df = pd.DataFrame(data = data)

    intense_final_df.to_csv("intense_harmful_doggolingo.csv", index=False)


