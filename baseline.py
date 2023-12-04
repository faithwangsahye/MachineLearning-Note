import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
d = pd.concat([train,test])

cat_columns = d.select_dtypes(include='O').columns
cat_columns = list(cat_columns)
cat_columns.remove("subscribe")
for c in cat_columns:
    le = LabelEncoder()
    d[c] = le.fit_transform(d[c])
subscribe2int = {"no":0,"yes":1}
d['subscribe'] = d['subscribe'].apply(lambda x:subscribe2int.get(x,x))

train = d[~d['subscribe'].isnull()]
test = d[d['subscribe'].isnull()]
features = list(train.columns)
features.remove("id")
features.remove("subscribe")
y = train['subscribe'].values.astype(int)
train = train[features].values


kfold = KFold(n_splits = 5,shuffle = True,random_state = 2023)
for trn_idx,val_idx in kfold.split(train):
    trn,y_trn = train[trn_idx],y[trn_idx]
    val,y_val = train[val_idx],y[val_idx]
    break


def model_lgb(trn,y_trn):
    model = lgb.LGBMClassifier(
            objective='binary',
            max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2023,
            n_estimators=5000, subsample=1, colsample_bytree=1,silent = False
    )
    model.fit(trn,y_trn)
    return model

model = model_lgb(trn,y_trn)
preds = model.predict(val)
print("验证集准确率:%.5f,共预测出%d个1"%(accuracy_score(y_val,preds),preds.sum()))

preds = model.predict(test[features])
test['subscribe'] = preds
int2subscribe = {subscribe2int[x]:x for x in subscribe2int}
test['subscribe'] = test['subscribe'].apply(lambda x:int2subscribe[x])
test[['id','subscribe']].to_csv("submission.csv",index = None)


f_imp = model.feature_importances_
f_imp_argsort = f_imp.argsort()[::-1]
features = np.array(features)
features_imp = features[f_imp_argsort]

cat_columns_imp = [x for x in features_imp if x in cat_columns]

