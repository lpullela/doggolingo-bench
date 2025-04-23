import pandas as pd
import re

def create_sent(text):
    text = text.lower()
    pattern = r"[0-9 \t\n\r\f\v,./\-|+!?*&|#'$()^%[\]@+=~`{}:;<>]"
    result = re.split(pattern, text)
    result = [word for word in result if word]
    return ' '.join(result)

def check_len(sent):
    return len(sent.split(' ')) >= 15

data = pd.read_csv("/home/smirrashidi/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/data/reddit.csv")
data = pd.DataFrame(data)

data["clean_text"] = data["text"].apply(create_sent)
filtered_data = data[data["clean_text"].apply(check_len)]

filtered_data["clean_text"].to_csv("hate_speech.csv", index=False)
