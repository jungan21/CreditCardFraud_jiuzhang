import pandas as pd
import numpy as np

filename = "./input/raw_train.csv"
df = pd.read_csv(filename)
#print(df.shape)

train = df.sample(frac=0.7, random_state=200)
test = df.drop(train.index)
#print(train.shape)
#print(test.shape)

output_path = "./data/"
filename_train = "train.csv"
train.to_csv(output_path + filename_train, index=False)

cols_test = ['id', 'vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']
filename_test = "test.csv"
test[cols_test].to_csv(output_path + filename_test, index=False)

filename_correct = "correct_submission.csv"
cols_correct =['id', 'trip_duration']
test[cols_correct].to_csv(output_path + filename_correct, index=False)

print('Finish')


