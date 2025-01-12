import pandas as pd
from spellchecker import SpellChecker

spell = SpellChecker()

doggolingo = pd.read_csv('doggolingo_dict.csv')
dl_df = pd.DataFrame(doggolingo)

in_dict = []
not_in_dict = []
for word in dl_df['word']:
    if word in spell.word_frequency:
        in_dict.append(word)
    else:
        not_in_dict.append(word)

print(f"doggolingo words that are in the pyspellchecker dictionary:{in_dict} \n")
print(f"doggolingo words that are not in the pyspellchecker dictionary:{not_in_dict}")