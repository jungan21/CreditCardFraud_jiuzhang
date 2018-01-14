import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as sp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import time

# load training and testing dataset
input_path = "./data/"
output_path = "./output/"

input_filename = input_path + "train.csv"
train = pd.read_csv(input_filename)

test_filename = input_path + "test.csv"
test = pd.read_csv(test_filename)

frames = [train, test]
raw = pd.concat(frames)

# loc_x, loc_y, lat and lon
alpha = 0.02
plt.figure(figsize=(10,10))

# loc_x and loc_y
plt.subplot(121)
plt.scatter(raw.loc_x, raw.loc_y, color='blue', alpha=alpha)
plt.title('loc_x and loc_y')

# lat and lon
plt.subplot(122)
plt.scatter(raw.lon, raw.lat, color='green', alpha=alpha)
plt.title('lat and lon')
plt.savefig(output_path + 'courts.jpg')

# coodinator transform
raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)

loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2

# time remaining
raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

print(raw.action_type.unique())
print(raw.combined_shot_type.unique())
print(raw.shot_type.unique())

# season
raw['season'].unique()
raw['season'] = raw['season'].apply(lambda x: int(x.split('-')[1]))
raw['season'].unique()

# team
print(raw['team_id'].unique())
print(raw['team_name'].unique())

# opponent, matchup
pd.DataFrame({'matchup': raw.matchup, 'opponent': raw.opponent})

# shot distance
plt.figure(figsize=(5,5))

plt.scatter(raw.dist, raw.shot_distance, color='blue')
plt.title('dist and shot_distance')
plt.savefig(output_path + 'shot_distance.jpg')

# shot zone
plt.figure(figsize=(20,10))

def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = raw.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)

# shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')
plt.savefig(output_path + 'shot_zone.jpg')

# drop unnecessary variables
drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)

# make dummy variables
# turn categorical variables into dummy variables
categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)

# seperate data
df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)

train = df.drop('shot_made_flag', 1)
train_y = df['shot_made_flag']
#print(train.shape)
#print(train_y.shape)

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0, 2, num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    k_fold = KFold(n_splits=10, shuffle=True)
    for train_k, test_k in k_fold.split(train, train_y):
        #print(train.iloc[train_k].shape)
        #print(train_y.iloc[train_k].shape)
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
print(best_n, min_score)

# find best max_depth for RandomForestClassifier
print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0, 2, num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    k_fold = KFold(n_splits=10, shuffle=True)
    for train_k, test_k in k_fold.split(train, train_y):
        #print(train.iloc[train_k].shape)
        #print(train_y.iloc[train_k].shape)
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))
print(best_m, min_score)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')
plt.savefig('Ntrees.jpg')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')
plt.savefig('max_depth.jpg')


model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train, train_y)
pred = model.predict_proba(submission)

sub = pd.read_csv(input_path + "sample_submission.csv")
sub['shot_made_flag'] = pred
sub.to_csv(output_path + "real_submission.csv", index=False)