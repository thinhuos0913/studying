import pandas as pd
import numpy as np

vg_df = pd.read_csv('vgsales.csv', encoding='utf-8')
# print(vg_df.head())
# print(vg_df[['Name', 'Platform', 'Year', 'Genre', 'Publisher']].iloc[1:7])
# vg_df[['Name', 'Platform', 'Year', 'Genre', 'Publisher']].iloc[1:7]
genres = np.unique(vg_df['Genre'])
print('Genres:\n', genres)

from sklearn.preprocessing import LabelEncoder

gle = LabelEncoder()
genre_labels = gle.fit_transform(vg_df['Genre'])
# print(enumerate(gle.classes_))
# for index, label in enumerate(gle.classes_):
# 	print(index, label)
genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
print(genre_mappings)

vg_df['GenreLabel'] = genre_labels
print(vg_df[['Name', 'Platform', 'Year', 'Genre', 'GenreLabel']].iloc[1:7])

one_hot_features = pd.get_dummies(vg_df['Genre'])

print('one_hot_features:\n', one_hot_features.iloc[1:7])

# print([vg_df[['Name', 'Genre']], one_hot_features])
df_one_hot = pd.concat([vg_df[['Name', 'Genre']], one_hot_features], axis = 1)

# df_one_hot.to_csv('one_hot_df.csv')
