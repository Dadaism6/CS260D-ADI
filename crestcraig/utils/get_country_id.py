import pandas as pd
df = pd.read_csv('/Users/duanchenda/Desktop/gitplay/CS260D-ADI/full_cleaned_data.tsv',sep='\t')
unique_countries = df['country'].unique()
country2id = {c:i for i,c in enumerate(list(unique_countries))}
id2country = {i:c for i,c in enumerate(list(unique_countries))}
print(country2id)
print(id2country)