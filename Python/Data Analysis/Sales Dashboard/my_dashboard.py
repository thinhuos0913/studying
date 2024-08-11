import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ------ SET PAGE ------
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Super Market Sales Dashboard")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# ------ LOAD FILE ------
file = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if file is not None:
    filename = file.name
    st.write(filename)
    df = pd.read_excel(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"E:\Projects\Sales_Dashboard")
    df = pd.read_excel("supermarkt_sales.xlsx", 
    					engine="openpyxl",
    					sheet_name="Sales",
    					skiprows=3,
    					encoding = "ISO-8859-1")

# Add 'hour' column to dataframe
df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour

# ------ SELECT DATE ------
col1, col2 = st.columns((2))
df["Date"] = pd.to_datetime(df["Date"])

# Getting the min and max date 
startDate = pd.to_datetime(df["Date"]).min()
endDate = pd.to_datetime(df["Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()

# ------ SIDE BAR ------
st.sidebar.header("Fiter data: ")
# City
city = st.sidebar.multiselect("Choose the City", 
								options = df["City"].unique(), 
								default=df["City"].unique())
if not city:
    df2 = df.copy()
else:
    df2 = df[df["City"].isin(city)]

# Gender
gender = st.sidebar.multiselect("Choose Gender", 
								options=df["Gender"].unique(),
    							default=df["Gender"].unique())
if not gender:
    df3 = df2.copy()
else:
    df3 = df2[df2["Gender"].isin(gender)]

# Customer type
customer_type = st.sidebar.multiselect("Choose customer type",
										options=df["Customer_type"].unique(),
    									default=df["Customer_type"].unique(),)

df_selection = df.query(
    "City == @city & Customer_type == @customer_type & Gender == @gender"
)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

# ------ SUMMARY ------
# TOP KPI's
total_sales = int(df_selection["Total"].sum())
average_rating = round(df_selection["Rating"].mean(), 1)
star_rating = ":star:" * int(round(average_rating, 0))
average_sale_by_transaction = round(df_selection["Total"].mean(), 2)
# Create columns in layout
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")
with middle_column:
    st.subheader("Average Rating:")
    st.subheader(f"{average_rating} {star_rating}")
with right_column:
    st.subheader("Average Sales Per Transaction:")
    st.subheader(f"US $ {average_sale_by_transaction}")

st.markdown("""---""")

# ------ CREATE CHARTS ------
sales_by_product_line = df_selection.groupby(by=["Product line"])[["Total"]].sum().sort_values(by="Total")
with col1:
    st.subheader("Sales by Product line")
    fig = px.bar(sales_by_product_line, 
    	x = "Total", 
    	y = sales_by_product_line.index,
    	template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)
# fig_product_sales = px.bar(
#     sales_by_product_line,
#     x="Total",
#     y=sales_by_product_line.index,
#     orientation="h",
#     title="<b>Sales by Product Line</b>",
#     color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
#     template="plotly_white",
# )

# fig_product_sales.update_layout(
#     plot_bgcolor="rgba(0,0,0,0)",
#     xaxis=(dict(showgrid=False))
# )

# left_column, right_column = st.columns(2)
# left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
# right_column.plotly_chart(fig_product_sales, use_container_width=True)
sales_by_hour = df_selection.groupby(by=["hour"])[["Total"]].sum()
with col2:
    st.subheader("Sales by hour")
    fig = px.bar(sales_by_hour, 
    	x = sales_by_hour.index, 
    	y = "Total",color_discrete_sequence=["#0083B8"] * len(sales_by_hour))
    # fig.update_traces(text = filtered_df["Region"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

with col1:
	st.subheader("Sales by City")
	fig = px.pie(df_selection, values = "Total", names = "City", template = "gridon")
	st.plotly_chart(fig,use_container_width=True)

with col2:
	st.subheader("Sales by customer type")
	fig = px.pie(df_selection, values = "Total", names = "Customer_type", template = "gridon")
	st.plotly_chart(fig,use_container_width=True)

# df_selection["month_year"] = df_selection["Date"].dt.to_period("M")
# linechart = pd.DataFrame(df_selection.groupby(df_selection["month_year"].dt.strftime("%Y : %b"))["Total"].sum()).reset_index()
# with col1:
# 	st.subheader("Sales by month")
# 	fig = px.line(linechart, x = "month_year", y="Total", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
# 	st.plotly_chart(fig,use_container_width=True)

# # Create a scatter plot
# with col2:
# 	st.subheader("Relationship between Sales and Profits")
# 	fig = px.scatter(df_selection, x = "Total", y = "gross income", size = 'cogs')
# 	st.plotly_chart(fig,use_container_width=True)


# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

