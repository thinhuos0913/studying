import pypyodbc as odbc
import pandas as pd

# 1. Create a connection string
conn_string = ("Driver={ODBC Driver 11 for SQL Server};" # C:\Windows\System32\odbcad32.exe

            "Server=THINHTRAN\SQLEXPRESS;"

            "Database=QuizData;"

            "Trusted_Connection=yes;")

# If using username and password, create a connection string as per the below code.

# Some other example server values are

# server = 'localhost\sqlexpress' # for a named instance
# server = 'myserver,port' # to specify an alternate port
# server = 'tcp:myserver.database.windows.net' 
# database = 'mydb' 
# username = 'myusername' 
# password = 'mypassword' 
# ENCRYPT defaults to yes starting in ODBC Driver 18. It's good to always specify ENCRYPT=yes on the client side to avoid MITM attacks.
# conn = pyodbc.connect('DRIVER={ODBC Driver 18 for SQL Server};SERVER='+server+';DATABASE='+database+';ENCRYPT=yes;UID='+username+';PWD='+ password)

# 2. Connect SQL Server using pyodbc.connect
conn = odbc.connect(conn_string)
# print(conn)

# 3. Use pandas to execute and fetch the results from a SQL Query.
df = pd.read_sql("Select * from people where state_code = 'CA'", conn)

print(df)
# Or use a cursor to execute a SQL command

# cursor = conn.cursor()
# cursor.execute("Select * from people")
