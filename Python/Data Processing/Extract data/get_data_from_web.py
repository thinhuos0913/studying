import pandas as pd
url='https://www.theage.com.au/interactive/2020/coronavirus/data-feeder/covid-19-new-cases-json.json?v=3'
df = pd.read_json(url)
# df.to_csv('coronavirus.csv')
print(df)
# url='https://gw.vnexpress.net/th?types=gia_vang,box_ocb,ty_gia_vcb,data_shop_v2_home_vne_160,data_egift,data_shop_v2_home,rao_vat_v2,ewiki'
# df = pd.read_json(url)
# df = pd.DataFrame.from_dict(df)
# df.to_csv('gia_vang.csv')
# print(df)
# url='https://s1.vnecdn.net/vnexpress/restruct/j/v872/v3/production/vod.js'
# df = pd.read_json(url)
# df.to_csv('vod.csv')
# print(df)