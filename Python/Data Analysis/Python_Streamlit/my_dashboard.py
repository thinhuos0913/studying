import streamlit as st
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Superstore Sales Report")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
else:
    os.chdir(r"E:\Projects\PythonStreamlit-main")
    df = pd.read_csv("super_store.csv", encoding = "ISO-8859-1")

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Getting the min and max date 
startDate = pd.to_datetime(df["Order Date"]).min()
endDate = pd.to_datetime(df["Order Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Order Date"] >= date1) & (df["Order Date"] <= date2)].copy()

# ---- CREATE SIDEBAR ----
st.sidebar.header("Filter Here:")
region = st.sidebar.multiselect(
    "Select the Region:",
    options = df["Region"].unique(),
    default = df["Region"].unique(),
)

state = st.sidebar.multiselect(
    "Select the State:",
    options = df["State"].unique(),
    default = df["State"].unique(),
)

city = st.sidebar.multiselect(
    "Select the City:",
    options = df["City"].unique(),
    default = df["City"].unique(),
)

df_selection = df.query(
    "Region == @region & State == @state & City == @city"
)

# Check if the dataframe is empty:
if df_selection.empty:
    st.warning("No data available based on the current filter settings!")
    st.stop() # This will halt the app from further execution.

# TOP KPI's
total_sales = int(df_selection["Sales"].sum())
total_profit = int(df_selection["Profit"].sum())

left_column, right_column = st.columns(2)
with left_column:
    st.subheader("Total Sales:")
    st.subheader(f"US $ {total_sales:,}")

with right_column:
	st.subheader("Profit:")
	st.subheader(f"US $ {total_profit:,}")

st.markdown("""---""")

category_df = df_selection.groupby(by = ["Category"], as_index = False)["Sales"].sum()

col1,col2 = st.columns(2)
with col1:
	st.subheader("Sales by Category")
	fig1 = px.bar(category_df, x = "Category", y = "Sales", 
					text = ['${:,.2f}'.format(x) for x in category_df["Sales"]],
					# title = "<b>Sales by category</b>",
	                template = "seaborn")
# st.plotly_chart(fig1,use_container_width=True, height = 200)

with col2:
	st.subheader("Sales by Region")
	fig2 = px.pie(df_selection, values = "Sales", names = "Region", hole = 0.5,) 
					# title = "<b>Sales by Region</b>")
	fig2.update_traces(text = df_selection["Region"], textposition = "inside")
	# st.plotly_chart(fig2,use_container_width=True)

# left_column, right_column = st.columns(2)
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

df_selection["month_year"] = df_selection["Order Date"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(df_selection.groupby(df_selection["month_year"].dt.strftime("%Y : %b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y="Sales", labels = {"Sales": "Amount"},height=500, width = 1000,template="gridon")
st.plotly_chart(fig2,use_container_width=True)

with st.expander("View Data of TimeSeries:"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode("utf-8")
    st.download_button('Download Data', data = csv, file_name = "TimeSeries.csv", mime ='text/csv')