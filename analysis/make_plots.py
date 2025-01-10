import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('merged_responses_output.csv')

# Assuming 'df' is your DataFrame
rating_map = {'EXCELLENT': 4, 'GOOD': 3, 'OKAY': 2, 'BAD': 1}

# Convert rating columns to numerical values
for col in df.columns:
    if any(col.startswith(prefix) for prefix in ['Interpret_rating_', 'Generate_rating_', 'Create_rating_', 'Translate_rating_']):
        df[col] = df[col].map(rating_map)

# Initialize the dictionary to store results
results = {}

# Calculate mean and std for each relevant column
for col in df.columns:
    if any(col.startswith(prefix) for prefix in ['Interpret_rating_', 'Generate_rating_', 'Create_rating_', 'Translate_rating_']):
        prompt_key, model_name = col.split('_rating_')
        mean_val = df[col].mean()
        std_val = df[col].std()
        results[(prompt_key, model_name)] = (mean_val, std_val)

# make bar graphs for each prompt
for prompt in ['Interpret', 'Generate', 'Create', 'Translate']:
    prompt_results = {model: results[(prompt, model)] for prompt, model in results.keys() if prompt == prompt}
    prompt_df = pd.DataFrame(prompt_results).T
    prompt_df.columns = ['mean', 'std']
    prompt_df = prompt_df.sort_values(by='mean', ascending=False)
    prompt_df.plot(kind='bar', y='mean', yerr='std', legend=False)
    plt.title(f'{prompt} ratings')
    plt.ylabel('Rating')
    plt.yticks([1, 2, 3, 4], ['BAD', 'OKAY', 'GOOD', 'EXCEL.'])

    # save these in the analysis folder
    plt.savefig(f'analysis/plots/{prompt}_ratings.png')
    plt.show()

# use the results dict to make a bar graph comparing the prompts regardless of the model (group together the ratings if the prompt is the same)
interpret_mean = 0
generate_mean = 0
create_mean = 0
translate_mean = 0
interpret_std = 0
generate_std = 0
create_std = 0
translate_std = 0

for key, value in results.items(): 
    if key[0] == 'Interpret': 
        interpret_mean += value[0]
        interpret_std += value[1]
    elif key[0] == 'Generate':
        generate_mean += value[0]
        generate_std += value[1]
    elif key[0] == 'Create':
        create_mean += value[0]
        create_std += value[1]
    elif key[0] == 'Translate':
        translate_mean += value[0]
        translate_std += value[1]
    
interpret_mean /= 3
generate_mean /= 3
create_mean /= 3
translate_mean /= 3
interpret_std /= 3
generate_std /= 3
create_std /= 3
translate_std /= 3

# make a bar graph
prompt_means = [interpret_mean, generate_mean, create_mean, translate_mean]
prompt_stds = [interpret_std, generate_std, create_std, translate_std]
prompts = ['Interpret', 'Generate', 'Create', 'Translate']
plt.figure(figsize=(10, 6))
plt.bar(prompts, prompt_means, yerr=prompt_stds, capsize=5)
plt.title('Prompt ratings')
plt.ylabel('Rating')
plt.yticks([1, 2, 3, 4], ['BAD', 'OKAY', 'GOOD', 'EXCEL.'])
plt.savefig('analysis/plots/prompt_ratings.png')
plt.show()

print(results)
