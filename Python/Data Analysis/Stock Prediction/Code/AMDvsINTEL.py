import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from plotly.offline import iplot

# data loading
# nse_df = pd.read_csv('NSE-TATAGLOBAL11.csv')
intel_df = pd.read_csv('INTC.csv')
amd_df = pd.read_csv('AMD.csv')
# convert date string to pandas timestamp
intel_df['Date'] = intel_df['Date'].apply(pd.Timestamp)
amd_df['Date'] = amd_df['Date'].apply(pd.Timestamp)
print(intel_df.head())
print(intel_df.tail())
print(amd_df.head())
print(amd_df.tail())

plt.figure(figsize=(15, 5))
plt.plot(intel_df['Date'], intel_df['Close'], label='Intel Close Price', c='b')
plt.plot(amd_df['Date'], amd_df['Close'], label='AMD Close Price', c='r')
plt.legend()
plt.show()

# trace_1 = go.Scatter(x=intel_df['Date'], y=intel_df['Close'], mode='lines', name='Intel Close Price')
# trace_2 = go.Scatter(x=amd_df['Date'], y=amd_df['Close'], mode='lines', name='AMD Close Price')

# layout = dict(title = 'Intel and AMD Stock Prices Comparison',
#               xaxis = dict(title = 'Date'),
#               yaxis = dict(title = 'Prices (USD)'),
#               )

# data = [trace_1, trace_2]
# fig = dict(data=data, layout=layout)
# iplot(fig, filename='styled-line')