from io import StringIO
import pandas as pd

# xml = '''<?xml version='1.0' encoding='utf-8'?>
# <data xmlns="http://example.com">
#  <row>
#    <shape>square</shape>
#    <degrees>360</degrees>
#    <sides>4.0</sides>
#  </row>
#  <row>
#    <shape>circle</shape>
#    <degrees>360</degrees>
#    <sides/>
#  </row>
#  <row>
#    <shape>triangle</shape>
#    <degrees>180</degrees>
#    <sides>3.0</sides>
#  </row>
# </data>'''

# print(xml)

# df = pd.read_xml(StringIO(xml))
# print(df.head())

# Create a dataframe
df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                  index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])

print('Dataframe\n', df)

# Save json format
json = df.to_json(orient='split')
print('JSON\n', json)

# Encoding/decoding a Dataframe using 'split' formatted JSON:
json_df = pd.read_json(StringIO(json), orient='split')
print('Splitted\n', json_df)

# Encoding/decoding a Dataframe using 'index' formatted JSON:
json_df = df.to_json(orient='index')
print('Indexed\n',json_df)

# Encoding/decoding a Dataframe using 'records' formatted JSON. Note that index labels are not preserved with this encoding.
json_df = df.to_json(orient='records')
print('Records\n',json_df)

# Encoding with Table Schema
json_df = df.to_json(orient='table')
print('Table\n',json_df)

# data = '''{"index": {"0": 0, "1": 1},
#        "a": {"0": 1, "1": null},
#        "b": {"0": 2.5, "1": 4.5},
#        "c": {"0": true, "1": false},
#        "d": {"0": "a", "1": "b"},
#        "e": {"0": 1577.2, "1": 1577.1}}'''

# print(data)

# json_data = pd.read_json(StringIO(data), dtype_backend="numpy_nullable")
# print(json_data)

# Normalize json
data = [
    {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
    {"name": {"given": "Mark", "family": "Regner"}},
    {"id": 2, "name": "Faye Raker"},
]

print(data)

norm_data = pd.json_normalize(data)

print(norm_data)

data = [
    {
        "id": 1,
        "name": "Cole Volk",
        "fitness": {"height": 130, "weight": 60},
    },
    {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
    {
        "id": 2,
        "name": "Faye Raker",
        "fitness": {"height": 130, "weight": 60},
    },
]

norm_data = pd.json_normalize(data, max_level=0)
print(norm_data)

# Normalizes nested data up to level 1

norm_data = pd.json_normalize(data, max_level=1)
print(norm_data)

data = [
    {
        "state": "Florida",
        "shortname": "FL",
        "info": {"governor": "Rick Scott"},
        "counties": [
            {"name": "Dade", "population": 12345},
            {"name": "Broward", "population": 40000},
            {"name": "Palm Beach", "population": 60000},
        ],
    },
    {
        "state": "Ohio",
        "shortname": "OH",
        "info": {"governor": "John Kasich"},
        "counties": [
            {"name": "Summit", "population": 1234},
            {"name": "Cuyahoga", "population": 1337},
        ],
    },
]

result = pd.json_normalize(
    data, "counties", ["state", "shortname", ["info", "governor"]]
)

print('Result\n', result)

data = {"A": [1, 2]}
result = pd.json_normalize(data, "A", record_prefix="Prefix.")

print(result)

