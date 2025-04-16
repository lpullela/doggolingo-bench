import pandas as pd
import datetime
import json
import os
from openai import OpenAI

with open("../api_key.txt") as f:
    OPENAI_KEY = f.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_KEY
client = OpenAI()

BASE_PROMPT = """"""

class Processor:
    def __init__(self, input_file_path, output_file_path, model_name):

        self.client = OpenAI()
        self.model_name = model_name
        self.input_file_path = input_file_path
        current_date = datetime.datetime.now()
        formatted_date = current_date.strftime("%m-%d-%Y_%H-%M")
        self.output_file_path = f'{output_file_path}/{self.model_name}-{formatted_date}.json'
        
    def main(self):
        response_list = []

        data = pd.read_csv(self.input_file_path)
        for idx, row in data.iterrows():
            prompt = row["dg_sent"]
            
            raw_response = client.responses.create(
                model=self.model_name,
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
            print("Model response:", response)
            response_dict = {
                "prompt": prompt, 
                "response": response
            }
            response_list.append(response_dict)

        with open(self.output_file_path, "w", encoding="utf-8") as file:
            json.dump(response_list, file, indent=4, ensure_ascii=False)

        print("Data saved to", self.output_file_path)

if __name__ == "__main__":
    processor = Processor("prelim_translations.csv", ".", "gpt-4o-mini")
    processor.main()