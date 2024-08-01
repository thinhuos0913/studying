# from pathlib import Path
# from warnings import simplefilter

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# simplefilter("ignore")  # ignore warnings to clean up output cells

# # Set Matplotlib defaults
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
#     color="0.75",
#     style=".-",
#     markeredgecolor="0.25",
#     markerfacecolor="0.25",
#     legend=False,
# )
# # %config InlineBackend.figure_format = 'retina'


# # Load Tunnel Traffic dataset
# # data_dir = Path("../input/ts-course-data")
# tunnel = pd.read_csv("tunnel.csv", parse_dates=["Day"])

# # Create a time series in Pandas by setting the index to a date
# # column. We parsed "Day" as a date type by using `parse_dates` when
# # loading the data.
# tunnel = tunnel.set_index("Day")

# # By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# # (equivalent to `np.datetime64`, representing a time series as a
# # sequence of measurements taken at single moments. A `PeriodIndex`,
# # on the other hand, represents a time series as a sequence of
# # quantities accumulated over periods of time. Periods are often
# # easier to work with, so that's what we'll use in this course.
# tunnel = tunnel.to_period()

# print(tunnel.head())

# # Time-step features
# df = tunnel.copy()

# df['Time'] = np.arange(len(tunnel.index))

# print(df.head())

# # from sklearn.linear_model import LinearRegression

# # # Training data
# # X = df.loc[:, ['Time']]  # features
# # y = df.loc[:, 'NumVehicles']  # target

# # print(X)
# # print(y)

# # Train the model
# # model = LinearRegression()
# # model.fit(X, y)

# # Store the fitted values as a time series with the same time index as
# # the training data
# # y_pred = pd.Series(model.predict(X), index=X.index)
# # print(y_pred)

# # print('Weights:', model.coef_)
# # print('Bias:', model.intercept_)

# # ax = y.plot(**plot_params)
# # ax = y_pred.plot(ax=ax, linewidth=3)
# # ax.set_title('Time Plot of Tunnel Traffic');

# # plt.show()

# # Lag features

# df['Lag_1'] = df['NumVehicles'].shift(1)
# print(df.head())

# from sklearn.linear_model import LinearRegression

# X = df.loc[:, ['Lag_1']]
# X.dropna(inplace=True)  # drop missing values in the feature set
# y = df.loc[:, 'NumVehicles']  # create the target
# print(y)
# y, X = y.align(X, join='inner')  # drop corresponding values in target

# print(X)
# print(y)

# model = LinearRegression()
# model.fit(X, y)

# y_pred = pd.Series(model.predict(X), index=X.index)

# # fig, ax = plt.subplots()
# # ax.plot(X['Lag_1'], y, '.', color='0.25')
# # ax.plot(X['Lag_1'], y_pred)
# # ax.set_aspect('equal')
# # ax.set_ylabel('NumVehicles')
# # ax.set_xlabel('Lag_1')
# # ax.set_title('Lag Plot of Tunnel Traffic');

# # plt.show()

# ax = y.plot(**plot_params)
# ax = y_pred.plot()

# plt.show()

# L2: TREND
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from statsmodels.tsa.deterministic import DeterministicProcess

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
# # %config InlineBackend.figure_format = 'retina'


# # Load Tunnel Traffic dataset
# data_dir = Path("../input/ts-course-data")
# tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = pd.read_csv('tunnel.csv', parse_dates = ["Day"])
print(tunnel.head())

tunnel = tunnel.set_index("Day").to_period()

print(tunnel.head())

moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

# ax = tunnel.plot(style=".", color="0.5")
# moving_average.plot(
#     ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
# );
# plt.show()

from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()
# print(dp)
print(X.head())
y = tunnel["NumVehicles"]  # the target
print(y.head())

from sklearn.linear_model import LinearRegression

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)

# Plot graph
# ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
# _ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
# plt.show()

X = dp.out_of_sample(steps=30)
print('X_forecast:\n', X.head(10))
y_fore = pd.Series(model.predict(X), index=X.index)
print('y_forecast:\n', y_fore.head(10))

# Let's plot a portion of the series to see the trend forecast for the next 30 days:
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()

plt.show()
