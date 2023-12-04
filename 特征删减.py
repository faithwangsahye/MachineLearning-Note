#!/usr/bin/env python
# coding: utf-8
#FaithHuangSihui’ Learning Project

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np
train = pd.read_csv(r"D:\train.csv")
test = pd.read_csv(r"D:\test.csv")
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
            max_depth=-1, learning_rate=0.002, min_child_samples=3, random_state=2023,
            n_estimators=6500, subsample=1, colsample_bytree=1,silent = False
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


# In[17]:


train_[cat_columns_imp].mode(axis=0)


# In[7]:


test_= d[d['subscribe'].isnull()]


# In[18]:


test_[cat_columns_imp].mode(axis=0)


# In[20]:


train_[cat_columns_].describe()



# In[70]:


col_columns=d[features_imp].iloc[:,0:15].columns



dd=pd.concat([d[col_columns],d['subscribe']],axis=1)


# In[56]:


train = dd[~dd['subscribe'].isnull()]
test = dd[dd['subscribe'].isnull()]
features = list(train.columns)
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
            n_estimators=6000, subsample=1, colsample_bytree=1,silent = False
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
test[['subscribe']].to_csv("submission.csv",index = None)


# In[57]:


col_col=[x for  x in col_columns if x in cat_columns_imp]


dl=d[cat_columns_].iloc[:,0:5]



from sklearn.preprocessing import KBinsDiscretizer
kbd=KBinsDiscretizer(encode="ordinal")
dl_transformed=kbd.fit_transform(dl)
print(dl_transformed)


# In[76]:


dl=pd.DataFrame(dl_transformed,columns=["duration","pdays","lending_rate3m","age","cons_price_index"])


# In[77]:


features=d[cat_columns_imp]
features_=d[col_col].reset_index().iloc[:,1:11]
dl_=dl.reset_index().iloc[:,1:6]
features__=pd.concat([dl_,features_],axis=1)
colnames=features__.columns


# In[78]:


from sklearn import preprocessing
preprocessing.OneHotEncoder()


# In[79]:


def cate_colName(Transformer,category_cols,drop=None): 
    cate_cols_new=[]
    col_value=Transformer.categories_
    for i,j in enumerate(category_cols):
        if(drop==None)&(len(col_value[i])==2):
            cate_cols_new(j)
        else:
            for f in col_value[i]:
                feature_name=j+"_"+f
                cate_cols_new.append(feature_name)
    return(cate_cols_new)


# In[80]:


def binary_cross_combination(colnames,features__,OneHot=True):
    colnames_new_l=[]
    features_new_l=[]
    features=features__[colnames]
    for col_index,col_name in enumerate(colnames):
        for col_sub_index in range(col_index+1,len(colnames)):
            newnames=col_name+"&"+colnames[col_sub_index]
            colnames_new_l.append(newnames)
            newDF=pd.Series(features[col_name].astype("str")+"&"+features[colnames[col_sub_index]].astype("str"))
            features_new_l.append(newDF)
    features_new=pd.concat(features_new_l,axis=1)
    features_new.columns=colnames_new_l
    colnames_new=colnames_new_l
    if OneHot==True:
        enc=preprocessing.OneHotEncoder()
        enc.fit_transform(features_new)
        colnames_new=cate_colName(enc,colnames_new_l,drop=None)
        features_new=pd.DataFrame(enc.fit_transform(features_new).toarray(),columns=cate_colName(enc,colnames_new_l,drop=None))
    return features_new,colnames_new


# In[81]:


features_new,colnames_new=binary_cross_combination(colnames,features__)


# In[82]:


dd=dd.reset_index()
features_new=features_new.reset_index()


# In[83]:


d_new=pd.concat([dd,features_new],axis=1)


# In[84]:


train = d_new[~d_new['subscribe'].isnull()]
test = d_new[d_new['subscribe'].isnull()]
features = list(train.columns)
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
            max_depth=-1, learning_rate=0.0005, min_child_samples=3, random_state=2023,
            n_estimators=7500, subsample=1, colsample_bytree=1,silent = False
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
test[['subscribe']].to_csv("submission.csv",index = None)


# In[85]:


pd.Series(preds).to_excel(r"D:\q.xlsx")

