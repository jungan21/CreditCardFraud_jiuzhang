import numpy as np
import pandas as pd

# read training set from csv
data = pd.read_csv('./original_data/train.csv')
print(data.shape)

# split train test
train = data.iloc[:90000, :]
test = data.iloc[90000:, :]
print(test.shape)
print(test.shape)

# replace previous id
train_id = np.array(range(1, 90001, 1))
df_train_id = pd.DataFrame(train_id)
test_id = np.array(range(90001, 113845, 1))
df_test_id = pd.DataFrame(test_id)

train.loc[:, 'Id'] = df_train_id.iloc[:, 0]
test['Id'] = test.index + 1

correct_column = ['Id', 'Last']
correct = pd.DataFrame(index=test.index, columns=correct_column)
correct['Id'] = correct.index + 1

label = []
for idx, row in test.iterrows():
    row = test.loc[idx, 'Sequence']
    # print(row)
    seq = row.split(',')
    # print(seq)
    label.append(seq[-1])
    # print(label)
    seq = seq[:-1]
    new_row = ','.join(seq)
    test.loc[idx, 'Sequence'] = new_row
    # print(test_new.loc[idx])
    if idx % 100 == 0:
        epoch = idx//100
        print("epoch %d finished." % epoch)

se = pd.Series(label)
correct['Last'] = se.values
# print(correct.head)

output_path = './outputs/'
y_test = pd.read_csv(output_path + "correct_submission.csv").sort_values('Id')

y_correct = pd.merge(y_test, correct, on=['Id', 'Last'], how='inner')
print(y_correct)


train.to_csv(output_path + 'train.csv', index=False)
print('train finished')
test.to_csv(output_path + 'test.csv', index=False)
print('test finished')
correct.to_csv(output_path + 'correct_submission.csv', index=False, float_format='{:f}'.format, encoding='utf-8')

# print(train.head)
# print(test.head)
# print(correct.head)



