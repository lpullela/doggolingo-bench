import pandas as pd
import json
from collections import Counter
import os
from openai import OpenAI
import datetime
from nltk.corpus import stopwords

with open("../../api_key.txt") as f:
    OPENAI_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
client = OpenAI()

BASE_PROMPT = """Here are some English words and their DoggoLingo version:
1. human --> hooman
2. dog --> doggo
3. awesome --> pawsome

Do not output anything else except for the translated word. Therefore, your response should only be one word long. 
Please provide the DoggoLingo version for the following word: \n
"""
stop = stopwords.words('english')
hate_speech = pd.read_csv("./hate_speech.csv")
hs_series = hate_speech["clean_text"].dropna().apply(lambda sent: sent.split())

words = [word for sent in hs_series for word in sent if word not in stop]

freq = Counter(words)
sorted_freq = freq.most_common()

client = OpenAI()
model_name = "gpt-4o"
current_date = datetime.datetime.now()
formatted_date = current_date.strftime("%m-%d-%Y_%H-%M")
output_file_path = f'./augmented_dict.json'

response_list = []

for word_tuple in sorted_freq[:900]:
    prompt = BASE_PROMPT + word_tuple[0]
    print(f'Now processing {word_tuple[0]}')  
    raw_response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ]
            }
        ]
    )

    response = raw_response.output_text
    response_dict = {
                "word": word_tuple[0],
                "equivalent": response
            }
    response_list.append(response_dict)
    print(response_dict)

with open(output_file_path, "w", encoding="utf-8") as file:
    json.dump(response_list, file, indent=4, ensure_ascii=False)

print("Data saved to", output_file_path)