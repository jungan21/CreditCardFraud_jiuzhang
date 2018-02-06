import pandas as pd
import numpy as np

data = pd.read_csv('./inputs/train.csv')
new_id_np = np.array(range(1, 991, 1))
df = pd.DataFrame(new_id_np)
data.loc[:, 'id'] = df.iloc[:, 0]
# print(data.head())
correct_column = data.species.unique()
# print(data['species'].value_counts())
# print(correct_column)

train = data.iloc[:850, :]
test = data.iloc[850:, :]
test.to_csv('./outputs/test0.csv', index=False)

test_output = test.drop(test.columns[[1]], axis=1, inplace=False)
# print(test.head())

correct_id = test.iloc[:, 0]
# print correct_id
correct_submission = pd.DataFrame(0, index=correct_id-1, columns=correct_column)
print(correct_submission.head())
correct_submission.insert(loc=0, column='id', value=correct_id)
# print(correct_submission.head())
# print(correct_submission.shape)
for idx, row in correct_submission.iterrows():
    # print row
    #print idx
    column = test.loc[idx, 'species']
    row.loc[column] = 1
    # print row.loc[column]
    # print('finish')

output_path = './outputs/'
train.to_csv(output_path + 'train.csv', index=False)
test_output.to_csv(output_path + 'test.csv', index=False)
correct_submission.to_csv(output_path + 'correct_submission.csv', index=False)
