import os
import pandas as pd
import re
from convokit import Corpus, download


def create_list(text):
    text = text.lower()
    pattern = r"[A-Za-z]+"

    result = re.findall(pattern, text)

    return result

def pd_strat(word_list, text_data):
    pattern = r'\b(' + '|'.join(map(re.escape, word_list)) + r')\b'
    
    text_data['word'] = text_data['text'].str.extract(pattern, expand=False)
    
    text_data = text_data.dropna(subset="word")
    text_data = text_data.drop_duplicates(['text', 'word'])

    return text_data 

def loop_strat(word_list, text_data):
    sentences = text_data["text"].tolist()
    res = []

    for word in word_list: 
        print(f'Now searching for: {word}')
        for sentence in sentences:
            sent_list = create_list(sentence)
            if word in sent_list:
                res.append({
                    'word' : word,
                    'text' : sentence
    })

    sentences_df = pd.DataFrame(res).drop_duplicates()

    return sentences_df


if __name__ == "__main__":
    subreddit_titles = pd.read_csv('/home/smirrashidi/doggolingo-bench/data/dataset_creation/old_csvs/subreddit_titles.csv')
    doggo_dict = pd.read_csv('/home/smirrashidi/doggolingo-bench/data/final_doggolingo_dict.csv')
    words = doggo_dict['word'].tolist()
    file = 'sents_from_utterances.json'

    for subreddit in subreddit_titles['subreddit_title']:
        print(f"Searching through {subreddit} utterances.")

        # download the corpus and create a dataframe 
        corpus = Corpus(filename=download(f"subreddit-{subreddit}"))
        convo_dataframe = corpus.get_utterances_dataframe()
        convo_dataframe.drop(['timestamp', 'reply_to', 'speaker', 'conversation_id', 'meta.top_level_comment', 'meta.score', 'meta.retrieved_on','meta.gilded', 'meta.gildings', 'meta.subreddit', 'meta.stickied',
        'meta.permalink', 'meta.author_flair_text', 'vectors'], axis=1, inplace=True)

        # gather sentences using both strategies
        pd_df = pd_strat(words, convo_dataframe)
        loop_df = loop_strat(words, convo_dataframe)

        sent_df = pd.concat([pd_df, loop_df])
        sent_df = sent_df.drop_duplicates(['word', 'text']).sort_values(by=['word'])
        
        if os.path.isfile(file):
            df_read = pd.read_json(file, orient='index')
            df_read = pd.concat([df_read, sent_df], ignore_index=True)
            df_read.drop_duplicates(inplace=True)
            df_read.to_json(file, orient='index')
        else:
            sent_df.to_json(file, orient='index')
