import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('car.csv', delimiter='|',engine='python',error_bad_lines=False,warn_bad_lines=False, index_col=False, encoding='utf8')
print(df.head(10))

# Report theo kiểu xe
df_car_type = df.groupby('car_type').count()
df_car_type.sort_values(by='car_model',inplace=True)
print(df_car_type[-7:])

# Vẽ theo kiểu xe
# ax = df_car_type[-7:].plot.pie(y='car_model',legend=3,figsize=(20, 10), title='Tỷ lệ các loại xe',autopct='%.2f')
# ax.set_ylabel('')
# plt.show()

# Tổng hợp theo hãng xe
df_car_brand = df.copy()
df_car_brand['tmp'] = df_car_brand['car_model'].str.find('-')

df_car_brand['car_brand'] = df_car_brand.apply(lambda x: x['car_model'][:x['tmp']] if x['tmp']!= -1 else 'others', 1) 
print(df_car_brand['car_brand'].unique())

# Vẽ theo loại xe
# df_draw = df_car_brand[['car_model','car_brand']].groupby('car_brand').count()
# df_draw.sort_values(by='car_model',inplace=True)
# ax = df_draw[-8:].plot.pie(y='car_model',legend=3,figsize=(30, 10), title='Tỷ lệ các hãng xe',autopct='%.2f')
# ax.set_ylabel('')
# plt.show()

# Vẽ theo màu xe out_color
# df_color = df.groupby('out_color').count() # Group by ngoại thất color

# df_color.sort_values(by='car_model',inplace=True)

# ax = df_color[-10:].plot.pie(y='car_model',legend=3,figsize=(20, 10), title='Tỷ lệ màu sắc',autopct='%.2f')
# ax.set_ylabel('')
# plt.show()

# Xét theo số km xe cũ?
df_old_car = df[df['new_old']=='Xe cũ']
#Xoá Km
df_old_car['km_number'] = df_old_car['km'].str.replace(' Km','')
# Xoá ','
df_old_car['km_number'] = df_old_car['km_number'].str.replace(',','')
# Chuyển về kiểu số
df_old_car['km_number'] = df_old_car['km_number'].astype(float)

# Bỏ đi các dữ liệu lỗi
df_old_car = df_old_car[df_old_car['km_number']!=0]
print(df_old_car)

# import matplotlib.pyplot as plt
# Remove các dữ liệu trên 1 Million km
#df_old_car = df_old_car [df_old_car['km_number']<1000000]

# Plot histoframn
# ax = df_old_car[df_old_car['km_number']<=100000].hist(figsize=(20, 10),color='#86bf91', rwidth=0.9)
# plt.title('Số lượng xe bán theo khoảng công tơ mét')
# plt.show()

# Xử lý giá (price)
df_price = df.copy()
df_price_B = df_price[df_price['price'].str.contains('Tỷ')]
df_price_M = df_price[~df_price['price'].str.contains('Tỷ')]

df_price_M['tmp'] = df_price_M['price'].str.replace(' Triệu','')
df_price_M['price_number'] = df_price_M['tmp'].astype(float)

df_price_M.drop (columns=['tmp'],inplace=True)



df_price_B['tmp'] =  df_price_B['price'].str.replace(' Triệu','')
df_price_B['tmp_idx'] =  df_price_B['price'].str.find(' Tỷ')
df_price_B['price_number_b'] = df_price_B.apply(lambda x: x['tmp'][:x['tmp_idx']], 1)
df_price_B['price_number_m'] = df_price_B.apply(lambda x: x['tmp'][x['tmp_idx']+3:] if len(x['tmp'])>x['tmp_idx']+3 else 0, 1)
df_price_B['price_number_b'] = df_price_B['price_number_b'].astype(float)
df_price_B['price_number_m'] = df_price_B['price_number_m'].astype(float)

df_price_B['price_number'] = df_price_B['price_number_b']*1000 + df_price_B['price_number_m']
# print(df_price_B.tail(10))
df_price_B.drop (columns=['price_number_b','price_number_m','tmp','tmp_idx'],inplace=True)

df_price_final = pd.concat([df_price_M, df_price_B])
# print(df_price_final.tail(10))
# df_price_final['price_number'].hist(figsize=(20, 10),color='b', bins=25)
df_price_final[df_price_final['price_number']<2500].hist(figsize=(30, 10),color='r', bins=25)
plt.show()