import pandas_datareader
# Yahoo recently has become an unstable data source. If it gives an error, you may run the cell again, or try yfinance
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# Set the start and end date
start_date = '1990-01-01'
end_date = '2022-06-28'
# Set the ticker
ticker = 'AMZN'
# Get the data
data = data.get_data_yahoo(ticker, start_date, end_date)
print(data.head(10))

# %matplotlib inline
# data['Adj Close'].plot()
# plt.show()
# Plot the adjusted close price
data['Adj Close'].plot(figsize=(10, 7))
# Define the label for the title of the figure
plt.title("Adjusted Close Price of %s" % ticker, fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
# Show the plot
plt.show()
