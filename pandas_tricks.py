import pandas as pd

df = pd.read_csv("automated_output_TVP.csv")

# pandas built function
df.describe()
print(df.isna().sum())
print(df.dropna())
print(df.nunique())

# df[~df[].isin(df2[])]

# print(df)

# print columns
# print(df.columns)

# return rows 2 to 5
# print(df.iloc[1:, 1])

# return columns 2 to 5

