# from pandas_datareader import data
# import datetime
# from datetime import date
# from pandas import read_csv
# from pandas import set_option

# today = date.today()
# start_date = datetime.datetime(2012,1,1)
# end_date = today
# df = data.DataReader(name='ACB',data_source='yahoo',start = start_date,end=end_date)
# df = df.drop(["Close"], axis = 1)
# print(df.head())
# df.to_csv('acb_stock.csv')
#------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv("acb_stock.csv", 
# 	index_col='Date', 
# 	parse_dates=['Date'],
# ).drop('Paperback', axis=1)
# LOAD DATA
# df = pd.read_csv('acb_stock.csv', index_col = 'Date', 
# 	parse_dates = ['Date'],)
# print(df.head())

# TIME-STEP FEATURES
# df['Time'] = np.arange(len(df.index))
# print(df.head())

# import seaborn as sns

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
# ax.plot('Time', 'Adj Close', data=df, color='0.25')
# ax = sns.regplot(x='Time', y='Adj Close', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_title('Time Plot of VCB stock price');

# plt.show()
# data = df.copy()

# from sklearn.linear_model import LinearRegression

# Training data
# X = data.loc[:,['High'],['Low'],['Open'],['Volume']]
# X = data.loc[:, ['Time']]  # features
# y = data.loc[:, 'Adj Close']  # target

# print(X.head())
# print(y.head())

# Train the model
# model = LinearRegression()
# model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
# y_pred = pd.Series(model.predict(X), index=X.index)
# print(y_pred.head())

# print('Weights:', model.coef_)
# print('Bias:', model.intercept_)

# Set Matplotlib defaults
# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11, 4))
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=14,
#     titlepad=10,
# )
# plot_params = dict(
#     color="1.5",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )

# ax = y.plot(**plot_params)
# ax = y_pred.plot(ax=ax, linewidth=3)
# ax.set_title('Time Plot of VCB stock price');

# plt.show()

# LAG FEATURES
# data = df.copy()
# data['Lag_1'] = data['Adj Close'].shift(1)
# data = data.reindex(columns=['Adj Close', 'Lag_1'])

# print(data.head())

# X = data.loc[:, ['Lag_1']]
# print(X.head())
# X.dropna(inplace=True)  # drop missing values in the feature set
# print(X.head())
# y = data.loc[:, 'Adj Close']  # create the target
# print(y.head())
# y, X = y.align(X, join='inner')  # drop corresponding values in target

# print(X.head())
# print(y.head())

# model = LinearRegression()
# model.fit(X, y)

# y_pred = pd.Series(model.predict(X), index=X.index)
# print(y_pred.head())

# print('Weights:', model.coef_)
# print('Bias:', model.intercept_)

# fig, ax = plt.subplots()
# ax.plot(X['Lag_1'], y, '.', color='0.25')
# ax.plot(X['Lag_1'], y_pred)
# ax.set_aspect('equal')
# ax.set_ylabel('Adj Close')
# ax.set_xlabel('Lag_1')
# ax.set_title('Lag Plot of ACB stock');

# plt.show()
# fig, ax = plt.subplots()
# ax = sns.regplot(x='Lag_1', y='Adj Close', data=df, ci=None, scatter_kws=dict(color='0.25'))
# ax.set_aspect('equal')
# ax.set_title('Lag Plot of VCB stock price');

# plt.show()

# TREND:
# from pathlib import Path
# from warnings import simplefilter
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.deterministic import DeterministicProcess

# simplefilter("ignore")

# df = pd.read_csv('acb_stock.csv', index_col = 'Date', 
# 	parse_dates = ['Date'],)

# df['Day'] = np.arange(len(df.index))
# print(df.head())

# Set Matplotlib defaults
# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11, 5))
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=14,
#     titlepad=10,
# )
# plot_params = dict(
#     color="0.75",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )

# stock = df.loc[:,['Adj Close']]
# print(stock.head())

# moving_average = stock.rolling(
#     window=365,       # 365-day window
#     center=True,      # puts the average at the center of the window
#     min_periods=183,  # choose about half the window size
# ).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = stock.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="ACB Stock - 365-Days Moving Average", legend=False,);

# plt.show()

# dp = DeterministicProcess(
#     index=stock.index,  # dates from the training data
#     constant=True,       # dummy feature for the bias (y_intercept)
#     order=5,             # the time dummy (trend)
#     drop=True,           # drop terms if necessary to avoid collinearity
# )
# # # `in_sample` creates features for the dates given in the `index` argument
# X = dp.in_sample()
# print(X.head())
# # print(dp)
# print(X.head())
# y = stock["Adj Close"]  # the target
# print(y.head())

# from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=False)
# model.fit(X, y)
# y_pred = pd.Series(model.predict(X), index=X.index)
# print(y_pred.head())

# ax = stock.plot(style=".", color="0.5", title="Stock price - Linear Trend")
# _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# plt.show()

# X = dp.out_of_sample(steps=365)
# print('X_forecast:\n', X.head())
# y_forecast = pd.Series(model.predict(X), index=X.index)
# print('y_forecast:\n', y_forecast.head(10))

# ax = stock["2022-03":].plot(title="VCB stock - Trend Forecast", **plot_params)
# ax = y_pred["2022-03":].plot(ax=ax, linewidth=3, label="Trend")
# ax = y_forecast.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
# _ = ax.legend()

# plt.show()

# from pathlib import Path
# from warnings import simplefilter
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.deterministic import DeterministicProcess

# simplefilter("ignore")

# df = pd.read_csv('acb.csv', parse_dates = ["Date"],)
# print(df.head())
# df = df.set_index("Date").to_period()
# print(df.head())


# # print(data.head())
# # Set Matplotlib defaults
# plt.style.use("seaborn-whitegrid")
# plt.rc("figure", autolayout=True, figsize=(11, 5))
# plt.rc(
#     "axes",
#     labelweight="bold",
#     labelsize="large",
#     titleweight="bold",
#     titlesize=14,
#     titlepad=10,
# )
# plot_params = dict(
#     color="0.75",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )

# # stock = df.loc[:,['Adj Close']]
# # print(stock.head())

# # moving_average = df.rolling(
# #     window=365,       # 365-day window
# #     center=True,      # puts the average at the center of the window
# #     min_periods=183,  # choose about half the window size
# # ).mean()              # compute the mean (could also do median, std, min, max, ...)

# # ax = df.plot(style=".", color="0.5")
# # moving_average.plot(
# #     ax=ax, linewidth=3, title="ACB Stock - 365-Days Moving Average", legend=False,);

# # plt.show()

# dp = DeterministicProcess(
#     index=df.index,  # dates from the training data
#     constant=True,       # dummy feature for the bias (y_intercept)
#     order=1,             # the time dummy (trend)
#     drop=True,           # drop terms if necessary to avoid collinearity
# )

# X = dp.in_sample()
# print(X.head())

# y = df["Adj Close"]  # the target
# print(y.head())

# from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=False)
# model.fit(X, y)
# y_pred = pd.Series(model.predict(X), index=X.index)
# print(y_pred.head())

# # ax = df.plot(style=".", color="0.5", title="Stock price - Linear Trend")
# # _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# # plt.show()

# X = dp.out_of_sample(steps=60)
# # X = pd.DataFrame()
# print('X_forecast:\n', X.head())

# y_forecast = pd.Series(model.predict(X), index=X.index)
# print('y_forecast:\n', y_forecast.head(10))

# ax = df.plot(title="ACB stock - Trend Forecast", **plot_params)
# ax = y_pred.plot(ax=ax, linewidth=3, label="Trend")
# ax = y_forecast.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
# _ = ax.legend()

# plt.show()

# SEASONALITY
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

simplefilter("ignore")

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
# %config InlineBackend.figure_format = 'retina'


# annotations: https://stackoverflow.com/a/49238256/5769929
def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


df = pd.read_csv('acb.csv', parse_dates = ["Date"],)
print(df.head())
df = df.set_index("Date").to_period("D")
print(df.head())

X = df.copy()

# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

print(X.head())

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year
print(X.head())

# plot seasonal
# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
# seasonal_plot(X, y="Adj Close", period="week", freq="day", ax=ax0)
# seasonal_plot(X, y="Adj Close", period="year", freq="dayofyear", ax=ax1);

# plt.show()
# let's look at the periodogram:
# plot_periodogram(df.AdjClose);
# plt.show()

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=10)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=df.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index
print(X.head())

# Create model and make predictions in next 90 days:
y = df["AdjClose"]

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index,name = 'Fitted')
X_fore = dp.out_of_sample(steps=90)
print(X_fore.head())
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
print(y_fore.head())


ax = y.plot(color='0.25', style='.', title="ACB stock - Seasonal Forecast")
ax = y_pred.plot(ax=ax, label="Seasonal")
ax = y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
_ = ax.legend()

plt.show()