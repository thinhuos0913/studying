from pandas_datareader import data
import datetime
from bokeh.plotting import figure, show, output_file

start_date = datetime.datetime(2019,1,1)
end_date = datetime.datetime(2022,11,24)
df = data.DataReader(name='NFLX',data_source='yahoo',start = start_date,end=end_date)
df.to_csv('Netflix.csv')
# print(df)
def inc_dec(c, o): # input là chỉ số close và open của từng ngày và output là kết quả so sánh hai giá trị: Increase, Decrease hay Equal
    if c > o:
        value="Increase"
    elif c < o:
        value="Decrease"
    else:
        value="Equal"
    return value

df["Status"]=[inc_dec(c,o) for c, o in zip(df.Close,df.Open)]
# print(df)
p = figure(x_axis_type='datetime', width=1000, height=500, sizing_mode="scale_width")
p.title.text = "Candlestick Chart"
hours_12 = 12*60*60*1000
df["Middle"] = (df.Open+df.Close)/2
df["Height"] = abs(df.Close-df.Open)
# print(df)
p.segment(df.index, df.High, df.index, df.Low, color="Black")
# For increase:
p.rect(df.index[df.Status=="Increase"],df.Middle[df.Status=="Increase"],
	hours_12,df.Height[df.Status=="Increase"],fill_color="blue", line_color="black")
# For decrease:
p.rect(df.index[df.Status=="Decrease"],df.Middle[df.Status=="Decrease"],
	hours_12, df.Height[df.Status=="Decrease"],fill_color="red", line_color="black")
# p.segment(df.index, df.High, df.index, df.Low, color="Black")
# output_file("chart.html")
show(p)