import pandas as pd
import matplotlib.pyplot as plt

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

df_season = pd.read_csv('USAElectricAndGas.csv', 
	parse_dates=['DATE'], index_col='DATE')

print(df_season.head())

df_season.columns = ['IPG2211A2N']
# df_season.plot(figsize=(16, 4))
# plt.show()
print('data frame shape: ', df_season.shape)

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df_season, model='multiplicative')
# fig = result.plot()
# fig.set_size_inches(16, 12)
# plt.show()

# SARIMA model
train, test = df_season[df_season.index < '2020-01-01'], df_season[df_season.index >= '2020-01-01']
print('train shape: ', train.shape)
print('test shape: ', test.shape)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
import matplotlib.pyplot as plt
# plot_pacf(train)
# plot_acf(train)
# plt.show()

# from pyramid.arima import auto_arima

# model_sarima = auto_arima(train, start_p=0, start_q=0,
#                            max_p=5, max_q=5, m=12,
#                            start_P=0, seasonal=True,
#                            d=1, D=1, trace=True,
#                            error_action='ignore',  
#                            suppress_warnings=True, 
#                            stepwise=True)

# print(model_sarima.aic())

from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

fourier = CalendarFourier(freq="A", order=12)  # 10 sin/cos pairs for "A"nnual seasonality

dp = DeterministicProcess(
    index=train.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X = dp.in_sample()  # create features for dates in tunnel.index
print(X.head())
print(X.shape)
y = train['IPG2211A2N']
print(y.head())
print(y.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
print(y_pred.head())
# X_fore = dp.out_of_sample(steps=90)
# print(X_fore.head())
# y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)
# print(y_fore.head())

# ax = y.plot(color='0.25', style='.', title="USAElectricAndGas - Seasonal Forecast")
# ax = y_pred.plot(ax=ax, label="Seasonal")
# plt.show()

dp_test = DeterministicProcess(
    index=test.index,
    constant=True,               # dummy feature for bias (y-intercept)
    order=1,                     # trend (order 1 means linear)
    seasonal=True,               # weekly seasonality (indicators)
    additional_terms=[fourier],  # annual seasonality (fourier)
    drop=True,                   # drop terms to avoid collinearity
)

X_test = dp_test.in_sample()
print(X_test.shape)
y_test = test['IPG2211A2N']
print(y_test.shape)

y_fore = pd.Series(model.predict(X_test),index=X_test.index)

ax = y_test.plot(color='0.25', style='.', title="USAElectricAndGas - Seasonal Forecast")
ax = y_fore.plot(ax=ax, label="Seasonal")
# plt.show()