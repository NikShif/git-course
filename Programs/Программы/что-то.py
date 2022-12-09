import seaborn as sns
import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
df = sns.load_dataset('iris')

print(f'{df.info()} \n\n {df.head(10)}')
print(df.describe())
# print(df.columns)
# a = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# for i in a:
#     print(df[i].max(), df[i].min())
# print(df['petal_length'])

# for i in range(150):
#     df.iloc[i: i+1,1] = (df.iloc[i: i+1,1] - 2.0)/(4.4 - 2.0)
# print(df['sepal_width'])



    
