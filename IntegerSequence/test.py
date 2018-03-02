import pandas as pd

output_path = './outputs/'
y_test = pd.read_csv(output_path + "correct_submission.csv").sort_values('Id')
y_pred = pd.read_csv(output_path + "prediction_test.csv").sort_values('Id')

counts = (y_test.Last == y_pred.Last).value_counts()

accuracy = counts.iloc[0] / y_test.shape[0]
# accuracy = correct / y_test.shape[0]
# print(accuracy)

