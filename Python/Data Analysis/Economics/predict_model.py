import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
# LOAD DỮ LIỆU
df = pd.read_csv('Economic_S4.csv')

# 1. Kiểm tra thông số thống kê của mỗi biến
print("--------------------------")
print(df['Inflation'].describe())
print("--------------------------")
print(df['Unemployment'].describe())
print("--------------------------")
print(df['Lending_Interest_Rate'].describe())
print("--------------------------")
print(df['GDP'].describe())
print("--------------------------")
print(df['NPL'].describe())
print("--------------------------")

# 2. Vẽ phân phối dữ liệu dạng box plots
# Tạo DF mới loại bỏ Biến Year và NPL trong DF gốc sau đó vẽ các biểu đồ box với từng biến
df_box = df[['Inflation', 'Lending_Interest_Rate', 'Unemployment', 'GDP']]

# Chạy hàm boxplot để vẽ biểu đồ Box và lấy các giá trị của box. Hàm này trả về 2 giá trị ax và bp.
# Ý nghĩa tham số return type:
# 1. ‘axes’ returns the matplotlib axes the boxplot is drawn on
# 2. ‘dict’ returns a dictionary whose values are the matplotlib Lines of the boxplot.
# 3. ‘both’ returns a namedtuple with the axes and dict.
# ax, bp = pd.DataFrame.boxplot(df_box, return_type='both')

# # In dữ liệu của box và viết giá trị của các điểm này vào biểu đồ box
# # outliner: trả về mảng giá trị nằm bên ngoài. Mảng chưa mỗi giá trị của chuỗi outliner
# # boxes: trả về mảng gồm 5 giá trị của box sẽ nối thành hình box gồm cận trên cận dưới box
# # medians: trả về mảng gồm 2 giá trị nối 2 điểm nằm ở giữa đường line mầu xanh lá cây
# # whiskers: trả về 2 mảng mỗi mảng gồm 2 giá trị vẽ từ cạnh box đến ria 2 đầu
# outliers = [flier.get_ydata() for flier in bp["fliers"]]
# boxes = [box.get_ydata() for box in bp["boxes"]]
# medians = [median.get_ydata() for median in bp["medians"]]
# whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]
# print("outliers:")
# print(outliers)
# for i in range(4):
#     for j in range(len(outliers[i])):
#         ax.text(i+1, outliers[i][j], "{:.4f}".format(outliers[i][j]))
# print("boxes:")
# print(boxes)
# pad = 0.2
# for i in range(4):
#     ax.text(i+1+pad, boxes[i][1], "{:.4f}".format(boxes[i][1]))
#     ax.text(i+1+pad, boxes[i][2], "{:.4f}".format(boxes[i][2]))
# print("medians:")
# print(medians)
# #print(medians[0][0], medians[0][1])
# for i in range(4):
#     ax.text(i+1, medians[i][0], "{:.2f}".format(medians[i][0]))
# print("whiskers:")
# print(whiskers)
# for i in range(4): #[0,3]
#     for j in range(2): #[0,1]
#         ax.text(i + 1, whiskers[i*2+j][1], "{:.2f}".format(whiskers[i*2+j][1]))
    # ax.text(i + 1, whiskers[i][1], "{:.2f}".format(whiskers[i][1]))

# plt.show()

# QUANTILES

# In quantiles theo các quãng 1%, 25%, 50%, 75%, 99%
quantiles = df_box.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
print(quantiles)

# HISTOGRAM
# 3. Vẽ phân phối dữ liệu dạng historical, 
# Hàm của Python mặc định chia dữ liệu thành 10 bin, 
# muốn thay đổi có thể điều chỉnh giá trị bin
# df_hist = df[['GDP']]
# hist = df_hist.hist(bins=10) 
# plt.show()

# SCATTER
# df_scatter = df[['GDP','NPL']]

# x = df_scatter['GDP'][:13]
# y = df_scatter['NPL'][:13]

# plt.scatter(x, y)

# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# plt.plot(x, p(x), "r--")

# plt.show()

# CORRELATION
# 5. Ma trận tương quan giữa các biến trên 3 method khác nhau
df_corr = df[['Inflation', 'Lending_Interest_Rate', 'Unemployment', 'GDP','NPL']]
print("-----------Kết quả pearson---------------")
print(df_corr.corr(method='pearson', min_periods=1))
print("-----------Kết quả kendall---------------")
print(df_corr.corr(method='kendall'))
print("------------Kết quả spearman--------------")
print(df_corr.corr(method='spearman'))

# MÔ HÌNH VÀ PHÂN TÍCH KẾT QUẢ
# Chọn biến đưa vào mô hình: Biến X = GDP và Y = NPL
X_Train = df['GDP'][:14].values.reshape(-1,1)
Y_Train = df['NPL'][:14].values.reshape(-1,1)

# Xây dựng mô hình linear với sklearn
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_Train, Y_Train)

# Dự báo kết quả Tỷ lệ nợ xấu trên số liệu dự báo của GPD 4 năm tiếp theo với sklearn
X_Test = df['GDP'][-4:].values.reshape(-1, 1)
print('GDP dự tính cho 4 năm tiếp theo: \n', X_Test)
Y_Pred = regr.predict(X_Test)
print('Dự đoán tỷ lệ nợ xấu NPL theo biến động của GDP: \n', Y_Pred)

# In các giá trị của hàm Hồi quy để kiểm tra mô hình
# In giá trị Hệ số chặn/ Hằng số Intercept và Hệ số góc Coefficients
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# In kết quả hồi quy tuyến tính thông qua phương pháp OLS với statsmodels
import statsmodels.api as sm
X_Train = sm.add_constant(X_Train)  # adding a constant
model = sm.OLS(Y_Train, X_Train , alpha=0.05).fit()
print_model = model.summary()
print(print_model)

# Tính phần dư Residual từ OLS
res_ols = model.resid  # residuals
print(res_ols)

# Vẽ phân phối dữ liệu của phần dư Residual để phân tích kết quả
df_hist_ols =pd.DataFrame(res_ols)
df_hist_ols.rename(columns={df_hist_ols.columns[0]: "Residual"}, inplace=True)
print(df_hist_ols)
print(df_hist_ols.describe())
df_gdp = df ['GDP'][:14]
df_res = pd.concat([df_hist_ols, df_gdp], axis=1, sort=False)
print("-----------Kết quả pearson giữa Residual và GDP---------------")
print(df_res.corr(method='pearson', min_periods=1))
print("------------Kết quả spearman giữa Residual và GDP--------------")
print(df_res.corr(method='spearman'))
# df_hist_ols.plot(kind='hist', stacked=True, bins=5)
# df_hist_ols.plot(kind='box')
# df_hist_ols.plot.density(figsize=(8, 6), linewidth=4)
# fig = sm.qqplot(res_ols, line='r')
# plt.show()
# sys.exit()

# MULTI REGRESSION
# print(df)
df_combine = df[:14]
# print(df_combine)
X = df_combine[['Inflation', 'Lending_Interest_Rate', 'Unemployment', 'GDP']] 
Y = df_combine['NPL']
print(X)
print(Y)
regr = LinearRegression()
regr.fit(X,Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X)  # adding a constant
model = sm.OLS(Y, X).fit()
        
print_model = model.summary()
print(print_model)
# Tính phần dư Residual từ OLS
res_ols = model.resid  # residuals
print(res_ols)
# Vẽ phân phối dữ liệu của phần dư Residual để phân tích kết quả
df_hist_ols =pd.DataFrame(res_ols)
df_hist_ols.rename(columns={df_hist_ols.columns[0]: "Residual"}, inplace=True)
print(df_hist_ols)
print(df_hist_ols.describe())
df_unemployment = df ['Unemployment'][:14]
df_res = pd.concat([df_hist_ols, df_unemployment], axis=1, sort=False)
print("-----------Kết quả pearson giữa Residual và Unemployment---------------")
print(df_res.corr(method='pearson', min_periods=1))
print("------------Kết quả spearman giữa Residual và Unemployment--------------")
print(df_res.corr(method='spearman'))
# df_hist_ols.plot(kind='hist', stacked=True, bins=5)
# df_hist_ols.plot(kind='box')
# df_hist_ols.plot.density(figsize=(8, 6), linewidth=4)
# fig = sm.qqplot(res_ols, line='r')
# plt.show()
# sys.exit()

#--------------------------------IN TỔ HỢP CÁC KẾT QUẢ HỒI QUY THEO TỪNG BIẾN
arr = ['Inflation', 'Lending_Interest_Rate', 'Unemployment', 'GDP']
i = 0
# for r in range(1,5):
for sub in (arr):
	i = i+1
	print("Model : ", i, np.asarray(sub))
	X = df_combine[np.asarray(sub)]
	print(X)
	Y = df_combine['NPL']
	print(Y)
	X = X.values.reshape(-1,1)
	# with sklearn
	regr = LinearRegression()
	regr.fit(X, Y)

	print('Intercept: \n', regr.intercept_)
	print('Coefficients: \n', regr.coef_)

	# with statsmodels
	X = sm.add_constant(X)  # adding a constant
	model = sm.OLS(Y, X).fit()
	print_model = model.summary()
	print(print_model)

sys.exit()