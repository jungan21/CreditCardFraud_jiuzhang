import pandas as pd
import shutil
import os

data = pd.read_csv('./inputs/train.csv')

train = data.iloc[:850, :]
test = data.iloc[850:, :]

def copy_rename(dst_dir, id, count):
    src_dir = './images/'
    src_file_name = '%s.jpg' % id
    shutil.copy(src_dir + src_file_name, dst_dir) # copy to destination with old name

    dst_old_file_name = os.path.join(dst_dir, src_file_name)
    dst_file_name = '%s.jpg' % count
    dst_new_file_name = os.path.join(dst_dir, dst_file_name)
    os.rename(dst_old_file_name, dst_new_file_name)

count = 1
train_items = train['id']
for item in train_items:
    copy_rename('./train_images', item, count)
    count += 1
    #print('Print finish, id = %s', item)

test_items = test['id']
for item in test_items:
    copy_rename('./test_images', item, count)
    count += 1




