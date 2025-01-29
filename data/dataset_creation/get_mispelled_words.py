import re
import pandas as pd

from convokit import Corpus, download
from spellchecker import SpellChecker

def create_list(text):
    text = text.lower()
    pattern = r"[ \t\n\r\f\v,./\-|+!?*&|#'$()^%[\]@+=~`{}:;<>\"“”‘’]"

    result = re.split(pattern, text)
    result = [word for word in result if word]

    return result

subreddit_titles = pd.read_csv('subreddit_titles.csv')
spell = SpellChecker()
misspelled_words_res = []
unique_misspelled_words = set()

for subreddit in subreddit_titles['subreddit_title']:
    print(f"Searching through {subreddit} corpus.")

    # download the corpus and create a dataframe 
    corpus = Corpus(filename=download(f"subreddit-{subreddit}"))
    convo_dataframe = corpus.get_conversations_dataframe()

    # clean the data 
    convo_dataframe.columns
    convo_dataframe.drop(['meta.stickied', 'vectors', 'meta.author_flair_text', 'meta.gildings', 'meta.gilded', 'meta.timestamp', 'meta.domain', 'meta.num_comments'], axis=1, inplace=True)
    convo_dataframe.columns = ['post_title', 'subreddit']
    convo_dataframe['word_list'] = convo_dataframe['post_title'].apply(create_list)

    # look for misspelled words and update misspelled_words_res
    for _, row in convo_dataframe.iterrows():  
        word_list = row['word_list']
        post_title = row['post_title'] 
        misspelled_words = spell.unknown(word_list)
        for word in misspelled_words:
            # only add if the word is not already in the set
            if word not in unique_misspelled_words:
                misspelled_words_res.append({
                    "misspelled_word": word,
                    "corpus": subreddit,
                    "context": post_title 
                })
                unique_misspelled_words.add(word)  
    
    print(f"Search through {subreddit} complete.")

# save misspelled words to CSV
misspelled_words_df = pd.DataFrame(misspelled_words_res)
misspelled_words_df.to_csv("misspelled_words.csv", index=False)

print("Misspelled words saved to misspelled_words.csv")
