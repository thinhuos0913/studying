import pandas as pd

df = pd.read_csv("book_sales.csv", 
	index_col='Date', 
	parse_dates=['Date'],
).drop('Paperback', axis=1)

print(df.head())

# Time-step features
import numpy as np

df['Time'] = np.arange(len(df.index))

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use("seaborn-whitegrid")
# plt.rc(
#     "figure",
#     autolayout=True,
#     figsize=(11, 4),
#     titlesize=18,
#     titleweight='bold',
# )
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=16,
#     titlepad=10,
# )

# # %config InlineBackend.figure_format = 'retina'

# fig, ax = plt.subplots()
# ax.plot('Time', 'Hardcover', data=df, color='0.75')
# ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_title('Time Plot of Hardcover Sales');

# plt.show()

# Lag features
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

# print(df.head())

fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');

plt.show()