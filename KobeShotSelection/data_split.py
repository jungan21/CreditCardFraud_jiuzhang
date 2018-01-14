import pandas as pd

filename = "./data/raw_data.csv"
df = pd.read_csv(filename)
train = df[pd.notnull(df['shot_made_flag'])]
test = df[pd.isnull(df['shot_made_flag'])]

path = "./data/"
filename_train = 'train.csv'
train.to_csv(path + filename_train, index=False)
filename_test = 'test.csv'
test.to_csv(path + filename_test, index=False)

print('Finish raw data split.')