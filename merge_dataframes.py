import pandas as pd
import glob

# Get list of all responses_output* csv files
csv_files = glob.glob('responses_output*.csv')

# Initialize an empty list to store dataframes
dataframes = []

# Read each csv file and append to the list
for file in csv_files:
    df = pd.read_csv(file)
    model_name = df['model_name'].iloc[0]
    df = df.rename(columns={
        'Interpret': f'Interpret_{model_name}',
        'Create': f'Create_{model_name}',
        'Translate': f'Translate_{model_name}',
        'Generate': f'Generate_{model_name}',
        'Interpret_rating': f'Interpret_rating_{model_name}',
        'Interpret_full_rating': f'Interpret_full_rating_{model_name}',
        'Create_rating': f'Create_rating_{model_name}',
        'Create_full_rating': f'Create_full_rating_{model_name}',
        'Translate_rating': f'Translate_rating_{model_name}',
        'Translate_full_rating': f'Translate_full_rating_{model_name}',
        'Generate_rating': f'Generate_rating_{model_name}',
        'Generate_full_rating': f'Generate_full_rating_{model_name}'
    })
    dataframes.append(df)

# Merge all dataframes on 'word' column
merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on=['word', 'definition'], how='outer')

# drop the 'model_name' column
merged_df = merged_df.drop(columns='model_name_x')

# Save the merged dataframe to a new csv file
merged_df.to_csv('merged_responses_output.csv', index=False)