import pandas as pd

df = pd.read_json('search.json')
print(df)
df.to_csv('universities.csv')