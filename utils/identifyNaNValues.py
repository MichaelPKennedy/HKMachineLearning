import pandas as pd

df = pd.read_pickle('AllDataFrames.pkl')

nan_rows = df[df.isnull().any(axis=1)]

for index, row in nan_rows.iterrows():
    nan_columns = row[row.isnull()].index.tolist()
    print(f"Row {index} has NaN values in columns: {nan_columns}")
