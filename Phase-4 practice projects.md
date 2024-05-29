# Restaurant Food Cost


```python
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```


```python
train = pd.read_excel('Data_Train (1).xlsx')
test = pd.read_excel('Data_Test.xlsx')
```


```python
train.shape
```




    (12690, 9)




```python
test.shape
```




    (4231, 8)




```python
train.duplicated().sum(), test.duplicated().sum()
```




    (25, 1)




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TITLE</th>
      <th>RESTAURANT_ID</th>
      <th>CUISINES</th>
      <th>TIME</th>
      <th>CITY</th>
      <th>LOCALITY</th>
      <th>RATING</th>
      <th>VOTES</th>
      <th>COST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASUAL DINING</td>
      <td>9438</td>
      <td>Malwani, Goan, North Indian</td>
      <td>11am – 4pm, 7:30pm – 11:30pm (Mon-Sun)</td>
      <td>Thane</td>
      <td>Dombivali East</td>
      <td>3.6</td>
      <td>49 votes</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASUAL DINING,BAR</td>
      <td>13198</td>
      <td>Asian, Modern Indian, Japanese</td>
      <td>6pm – 11pm (Mon-Sun)</td>
      <td>Chennai</td>
      <td>Ramapuram</td>
      <td>4.2</td>
      <td>30 votes</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASUAL DINING</td>
      <td>10915</td>
      <td>North Indian, Chinese, Biryani, Hyderabadi</td>
      <td>11am – 3:30pm, 7pm – 11pm (Mon-Sun)</td>
      <td>Chennai</td>
      <td>Saligramam</td>
      <td>3.8</td>
      <td>221 votes</td>
      <td>800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>QUICK BITES</td>
      <td>6346</td>
      <td>Tibetan, Chinese</td>
      <td>11:30am – 1am (Mon-Sun)</td>
      <td>Mumbai</td>
      <td>Bandra West</td>
      <td>4.1</td>
      <td>24 votes</td>
      <td>800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DESSERT PARLOR</td>
      <td>15387</td>
      <td>Desserts</td>
      <td>11am – 1am (Mon-Sun)</td>
      <td>Mumbai</td>
      <td>Lower Parel</td>
      <td>3.8</td>
      <td>165 votes</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 12690 entries, 0 to 12689
    Data columns (total 9 columns):
     #   Column         Non-Null Count  Dtype 
    ---  ------         --------------  ----- 
     0   TITLE          12690 non-null  object
     1   RESTAURANT_ID  12690 non-null  int64 
     2   CUISINES       12690 non-null  object
     3   TIME           12690 non-null  object
     4   CITY           12578 non-null  object
     5   LOCALITY       12592 non-null  object
     6   RATING         12688 non-null  object
     7   VOTES          11486 non-null  object
     8   COST           12690 non-null  int64 
    dtypes: int64(2), object(7)
    memory usage: 892.4+ KB
    


```python
for i in train.columns:
    print("Unique values in", i, train[i].nunique())
```

    Unique values in TITLE 113
    Unique values in RESTAURANT_ID 11892
    Unique values in CUISINES 4155
    Unique values in TIME 2689
    Unique values in CITY 359
    Unique values in LOCALITY 1416
    Unique values in RATING 32
    Unique values in VOTES 1847
    Unique values in COST 86
    


```python
df = train.append(test,ignore_index=True)
```


```python
df = df[['TITLE', 'CUISINES', 'TIME', 'CITY', 'LOCALITY', 'RATING', 'VOTES', 'COST']]
```


```python
def extract_closed(time):
    a = re.findall('Closed \(.*?\)', time)
    if a != []:
        return a[0]
    else:
        return 'NA'

df['CLOSED'] = df['TIME'].apply(extract_closed)
```


```python
df['TIME'] = df['TIME'].str.replace(r'Closed \(.*?\)','')
```


```python
df['RATING'] = df['RATING'].str.replace('NEW', '1')
df['RATING'] = df['RATING'].str.replace('-', '1').astype(float)
```


```python
df['VOTES'] = df['VOTES'].str.replace(' votes', '').astype(float)
```


```python
df['CITY'].fillna('Missing', inplace=True)  
df['LOCALITY'].fillna('Missing', inplace=True)  
df['RATING'].fillna(3.8, inplace=True)  
df['VOTES'].fillna(0.0, inplace=True) 
```


```python
df['COST'] = df['COST'].astype(float)
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TITLE</th>
      <th>CUISINES</th>
      <th>TIME</th>
      <th>CITY</th>
      <th>LOCALITY</th>
      <th>RATING</th>
      <th>VOTES</th>
      <th>COST</th>
      <th>CLOSED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASUAL DINING</td>
      <td>Malwani, Goan, North Indian</td>
      <td>11am – 4pm, 7:30pm – 11:30pm (Mon-Sun)</td>
      <td>Thane</td>
      <td>Dombivali East</td>
      <td>3.6</td>
      <td>49.0</td>
      <td>1200.0</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASUAL DINING,BAR</td>
      <td>Asian, Modern Indian, Japanese</td>
      <td>6pm – 11pm (Mon-Sun)</td>
      <td>Chennai</td>
      <td>Ramapuram</td>
      <td>4.2</td>
      <td>30.0</td>
      <td>1500.0</td>
      <td>NA</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['TITLE'].nunique(), df['CUISINES'].nunique()
```




    (123, 5183)




```python
calc_mean = df.groupby(['CITY'], axis=0).agg({'RATING': 'mean'}).reset_index()
calc_mean.columns = ['CITY','CITY_MEAN_RATING']
df = df.merge(calc_mean, on=['CITY'],how='left')

calc_mean = df.groupby(['LOCALITY'], axis=0).agg({'RATING': 'mean'}).reset_index()
calc_mean.columns = ['LOCALITY','LOCALITY_MEAN_RATING']
df = df.merge(calc_mean, on=['LOCALITY'],how='left')
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TITLE</th>
      <th>CUISINES</th>
      <th>TIME</th>
      <th>CITY</th>
      <th>LOCALITY</th>
      <th>RATING</th>
      <th>VOTES</th>
      <th>COST</th>
      <th>CLOSED</th>
      <th>CITY_MEAN_RATING</th>
      <th>LOCALITY_MEAN_RATING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASUAL DINING</td>
      <td>Malwani, Goan, North Indian</td>
      <td>11am – 4pm, 7:30pm – 11:30pm (Mon-Sun)</td>
      <td>Thane</td>
      <td>Dombivali East</td>
      <td>3.6</td>
      <td>49.0</td>
      <td>1200.0</td>
      <td>NA</td>
      <td>3.376271</td>
      <td>3.388889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASUAL DINING,BAR</td>
      <td>Asian, Modern Indian, Japanese</td>
      <td>6pm – 11pm (Mon-Sun)</td>
      <td>Chennai</td>
      <td>Ramapuram</td>
      <td>4.2</td>
      <td>30.0</td>
      <td>1500.0</td>
      <td>NA</td>
      <td>3.584588</td>
      <td>3.472222</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Define TF-IDF vectorizers
tf1 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
tf2 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
tf3 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
tf4 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)
tf5 = TfidfVectorizer(ngram_range=(1, 1), lowercase=True)

# Fit and transform each text column
df_title = tf1.fit_transform(df['TITLE'])
df_cuisines = tf2.fit_transform(df['CUISINES'])
df_city = tf3.fit_transform(df['CITY'])
df_locality = tf4.fit_transform(df['LOCALITY'])
df_time = tf5.fit_transform(df['TIME'])

# Create DataFrames from the transformed data
df_title = pd.DataFrame(data=df_title.toarray(), columns=tf1.get_feature_names_out())
df_cuisines = pd.DataFrame(data=df_cuisines.toarray(), columns=tf2.get_feature_names_out())
df_city = pd.DataFrame(data=df_city.toarray(), columns=tf3.get_feature_names_out())
df_locality = pd.DataFrame(data=df_locality.toarray(), columns=tf4.get_feature_names_out())
df_time = pd.DataFrame(data=df_time.toarray(), columns=tf5.get_feature_names_out())
```


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TITLE</th>
      <th>CUISINES</th>
      <th>TIME</th>
      <th>CITY</th>
      <th>LOCALITY</th>
      <th>RATING</th>
      <th>VOTES</th>
      <th>COST</th>
      <th>CLOSED</th>
      <th>CITY_MEAN_RATING</th>
      <th>LOCALITY_MEAN_RATING</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CASUAL DINING</td>
      <td>Malwani, Goan, North Indian</td>
      <td>11am – 4pm, 7:30pm – 11:30pm (Mon-Sun)</td>
      <td>Thane</td>
      <td>Dombivali East</td>
      <td>3.6</td>
      <td>49.0</td>
      <td>1200.0</td>
      <td>NA</td>
      <td>3.376271</td>
      <td>3.388889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CASUAL DINING,BAR</td>
      <td>Asian, Modern Indian, Japanese</td>
      <td>6pm – 11pm (Mon-Sun)</td>
      <td>Chennai</td>
      <td>Ramapuram</td>
      <td>4.2</td>
      <td>30.0</td>
      <td>1500.0</td>
      <td>NA</td>
      <td>3.584588</td>
      <td>3.472222</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = pd.concat([df, df_title, df_cuisines, df_city, df_locality, df_time], axis=1) 
df.drop(['TITLE', 'CUISINES', 'CITY', 'LOCALITY', 'TIME'], axis=1, inplace=True)
```


```python
df = pd.get_dummies(df, columns=['CLOSED'], drop_first=True)
```


```python
df.shape
```




    (16921, 2285)




```python
train_df = df[df['COST'].isnull()!=True]
test_df = df[df['COST'].isnull()==True]
test_df.drop('COST', axis=1, inplace=True)
```


```python
train_df.shape, test_df.shape
```




    ((12690, 2285), (4231, 2284))




```python
train_df['COST'] = np.log1p(train_df['COST'])
```
Train test split

```python
X = train_df.drop(labels=['COST'], axis=1)
y = train_df['COST'].values

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=1)
```


```python
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
```




    ((9517, 2284), (9517,), (3173, 2284), (3173,))


Build the model

```python
from math import sqrt 
from sklearn.metrics import mean_squared_log_error
```


```python
import warnings
warnings.filterwarnings('ignore')
!pip install lightgbm
```

    Collecting lightgbm
      Downloading lightgbm-4.3.0-py3-none-win_amd64.whl (1.3 MB)
                                                  0.0/1.3 MB ? eta -:--:--
                                                  0.0/1.3 MB ? eta -:--:--
                                                  0.0/1.3 MB ? eta -:--:--
                                                  0.0/1.3 MB ? eta -:--:--
                                                  0.0/1.3 MB ? eta -:--:--
                                                  0.0/1.3 MB ? eta -:--:--
         -                                        0.0/1.3 MB 393.8 kB/s eta 0:00:04
         -                                        0.1/1.3 MB 465.5 kB/s eta 0:00:03
         ---                                      0.1/1.3 MB 590.8 kB/s eta 0:00:03
         -----                                    0.2/1.3 MB 697.2 kB/s eta 0:00:02
         ---------                                0.3/1.3 MB 1.1 MB/s eta 0:00:01
         ------------                             0.4/1.3 MB 1.3 MB/s eta 0:00:01
         --------------                           0.5/1.3 MB 1.3 MB/s eta 0:00:01
         ------------------------                 0.8/1.3 MB 1.9 MB/s eta 0:00:01
         --------------------------------         1.1/1.3 MB 2.3 MB/s eta 0:00:01
         ---------------------------------------  1.3/1.3 MB 2.6 MB/s eta 0:00:01
         ---------------------------------------- 1.3/1.3 MB 2.4 MB/s eta 0:00:00
    Requirement already satisfied: numpy in c:\users\alsha mohammed\anaconda3\lib\site-packages (from lightgbm) (1.24.3)
    Requirement already satisfied: scipy in c:\users\alsha mohammed\anaconda3\lib\site-packages (from lightgbm) (1.10.1)
    Installing collected packages: lightgbm
    Successfully installed lightgbm-4.3.0
    


```python
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_log_error
from math import sqrt

# Assuming X_train, y_train, X_cv, y_cv are already defined and preprocessed

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_cv, label=y_cv)

params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',  
    'metric': 'rmse',  # Correct metric name for Root Mean Squared Error
    'learning_rate': 0.05, 
    'num_iterations': 350,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 15,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'feature_fraction': 0.55,
    'verbose': 1  # Set verbosity level here
}

lgbm = lgb.train(
    params=params,
    train_set=train_data,
    valid_sets=[test_data]
)

# Predicting on the cross-validation set
y_pred_lgbm = lgbm.predict(X_cv)

# Ensure y_cv and y_pred_lgbm are positive if using RMSLE
# If y values were log-transformed before, use np.exp for both y_cv and y_pred_lgbm
# Here assuming y_cv and y_pred_lgbm are in the original scale
y_cv = np.maximum(y_cv, 0)
y_pred_lgbm = np.maximum(y_pred_lgbm, 0)

rmsle = sqrt(mean_squared_log_error(y_cv, y_pred_lgbm))
print('RMSLE:', rmsle)

```


    ---------------------------------------------------------------------------

    LightGBMError                             Traceback (most recent call last)

    Cell In[65], line 26
          9 test_data = lgb.Dataset(X_cv, label=y_cv)
         11 params = {
         12     'objective': 'regression',
         13     'boosting_type': 'gbdt',  
       (...)
         23     'verbose': 1  # Set verbosity level here
         24 }
    ---> 26 lgbm = lgb.train(
         27     params=params,
         28     train_set=train_data,
         29     valid_sets=[test_data]
         30 )
         32 # Predicting on the cross-validation set
         33 y_pred_lgbm = lgbm.predict(X_cv)
    

    File ~\anaconda3\Lib\site-packages\lightgbm\engine.py:255, in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, feature_name, categorical_feature, keep_training_booster, callbacks)
        253 # construct booster
        254 try:
    --> 255     booster = Booster(params=params, train_set=train_set)
        256     if is_valid_contain_train:
        257         booster.set_train_data_name(train_data_name)
    

    File ~\anaconda3\Lib\site-packages\lightgbm\basic.py:3433, in Booster.__init__(self, params, train_set, model_file, model_str)
       3426     self.set_network(
       3427         machines=machines,
       3428         local_listen_port=params["local_listen_port"],
       3429         listen_time_out=params.get("time_out", 120),
       3430         num_machines=params["num_machines"]
       3431     )
       3432 # construct booster object
    -> 3433 train_set.construct()
       3434 # copy the parameters from train_set
       3435 params.update(train_set.get_params())
    

    File ~\anaconda3\Lib\site-packages\lightgbm\basic.py:2462, in Dataset.construct(self)
       2455             self._set_init_score_by_predictor(
       2456                 predictor=self._predictor,
       2457                 data=self.data,
       2458                 used_indices=used_indices
       2459             )
       2460 else:
       2461     # create train
    -> 2462     self._lazy_init(data=self.data, label=self.label, reference=None,
       2463                     weight=self.weight, group=self.group,
       2464                     init_score=self.init_score, predictor=self._predictor,
       2465                     feature_name=self.feature_name, categorical_feature=self.categorical_feature,
       2466                     params=self.params, position=self.position)
       2467 if self.free_raw_data:
       2468     self.data = None
    

    File ~\anaconda3\Lib\site-packages\lightgbm\basic.py:2123, in Dataset._lazy_init(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)
       2121     raise TypeError(f'Wrong predictor type {type(predictor).__name__}')
       2122 # set feature names
    -> 2123 return self.set_feature_name(feature_name)
    

    File ~\anaconda3\Lib\site-packages\lightgbm\basic.py:2863, in Dataset.set_feature_name(self, feature_name)
       2861         raise ValueError(f"Length of feature_name({len(feature_name)}) and num_feature({self.num_feature()}) don't match")
       2862     c_feature_name = [_c_str(name) for name in feature_name]
    -> 2863     _safe_call(_LIB.LGBM_DatasetSetFeatureNames(
       2864         self._handle,
       2865         _c_array(ctypes.c_char_p, c_feature_name),
       2866         ctypes.c_int(len(feature_name))))
       2867 return self
    

    File ~\anaconda3\Lib\site-packages\lightgbm\basic.py:263, in _safe_call(ret)
        255 """Check the return value from C API call.
        256 
        257 Parameters
       (...)
        260     The return value from C API calls.
        261 """
        262 if ret != 0:
    --> 263     raise LightGBMError(_LIB.LGBM_GetLastError().decode('utf-8'))
    

    LightGBMError: Feature (bakery) appears more than one time.



```python
from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=0.9, max_features=1.0, bootstrap=True, 
                      bootstrap_features=True, oob_score=True, warm_start=False, n_jobs=1, random_state=42, verbose=1)
br.fit(X_train, y_train)
y_pred_br = br.predict(X_cv)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred_br))))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[46], line 2
          1 from sklearn.ensemble import BaggingRegressor
    ----> 2 br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=0.9, max_features=1.0, bootstrap=True, 
          3                       bootstrap_features=True, oob_score=True, warm_start=False, n_jobs=1, random_state=42, verbose=1)
          4 br.fit(X_train, y_train)
          5 y_pred_br = br.predict(X_cv)
    

    TypeError: BaggingRegressor.__init__() got an unexpected keyword argument 'base_estimator'



```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                           min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                           random_state=42, verbose=1, warm_start=False)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_cv)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred_rf))))
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[47], line 2
          1 from sklearn.ensemble import RandomForestRegressor
    ----> 2 rf = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
          3                            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
          4                            min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
          5                            random_state=42, verbose=1, warm_start=False)
          6 rf.fit(X_train, y_train)
          7 y_pred_rf = rf.predict(X_cv)
    

    TypeError: RandomForestRegressor.__init__() got an unexpected keyword argument 'min_impurity_split'



```python
y_pred = y_pred_lgbm*0.70 + y_pred_br*0.15 +  y_pred_rf*0.15
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred))))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[48], line 1
    ----> 1 y_pred = y_pred_lgbm*0.70 + y_pred_br*0.15 +  y_pred_rf*0.15
          2 print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_cv), np.exp(y_pred))))
    

    NameError: name 'y_pred_lgbm' is not defined


# Predict on test set


```python
Xtest = test_df
```


```python
from sklearn.model_selection import KFold, RepeatedKFold
from lightgbm import LGBMRegressor

errlgb = []
y_pred_totlgb = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    param = {'objective': 'regression',
             'boosting': 'gbdt',
             'metric': 'l2_root',
             'learning_rate': 0.05,
             'num_iterations': 350,
             'num_leaves': 31,
             'max_depth': -1,
             'min_data_in_leaf': 15,
             'bagging_fraction': 0.85,
             'bagging_freq': 1,
             'feature_fraction': 0.55
             }

    lgbm = LGBMRegressor(**param)
    lgbm.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=0,
             early_stopping_rounds=100
             )

    y_pred_lgbm = lgbm.predict(X_test)
    print("RMSE LGBM: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))

    errlgb.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))
    p = lgbm.predict(Xtest)
    y_pred_totlgb.append(p)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[50], line 27
         13 param = {'objective': 'regression',
         14          'boosting': 'gbdt',
         15          'metric': 'l2_root',
       (...)
         23          'feature_fraction': 0.55
         24          }
         26 lgbm = LGBMRegressor(**param)
    ---> 27 lgbm.fit(X_train, y_train,
         28          eval_set=[(X_test, y_test)],
         29          verbose=0,
         30          early_stopping_rounds=100
         31          )
         33 y_pred_lgbm = lgbm.predict(X_test)
         34 print("RMSE LGBM: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))
    

    TypeError: LGBMRegressor.fit() got an unexpected keyword argument 'verbose'



```python
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor

err_br = []
y_pred_totbr = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=1.0, max_features=1.0, bootstrap=True,
                          bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=42, verbose=0)
    
    br.fit(X_train, y_train)
    y_pred_br = br.predict(X_test)

    print("RMSE BR:", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_br))))

    err_br.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_br))))
    p = br.predict(Xtest)
    y_pred_totbr.append(p)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[51], line 13
         10 X_train, X_test = X.loc[train_index], X.loc[test_index]
         11 y_train, y_test = y[train_index], y[test_index]
    ---> 13 br = BaggingRegressor(base_estimator=None, n_estimators=30, max_samples=1.0, max_features=1.0, bootstrap=True,
         14                       bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=42, verbose=0)
         16 br.fit(X_train, y_train)
         17 y_pred_br = br.predict(X_test)
    

    TypeError: BaggingRegressor.__init__() got an unexpected keyword argument 'base_estimator'



```python
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

err_rf = []
y_pred_totrf = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rf = RandomForestRegressor(n_estimators=40, criterion='mse', max_depth=None, min_samples_split=4, min_samples_leaf=1, 
                           min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                           min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, 
                           random_state=42, verbose=0, warm_start=False)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("RMSE RF: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))

    err_rf.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_rf))))
    p = rf.predict(Xtest)
    y_pred_totrf.append(p)
```


```python
np.mean(errlgb,0), np.mean(err_br,0), np.mean(err_rf,0)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[52], line 1
    ----> 1 np.mean(errlgb,0), np.mean(err_br,0), np.mean(err_rf,0)
    

    NameError: name 'err_rf' is not defined



```python
lgbm_final = np.exp(np.mean(y_pred_totlgb,0))
br_final = np.exp(np.mean(y_pred_totbr,0))
rf_final = np.exp(np.mean(y_pred_totrf,0))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[53], line 3
          1 lgbm_final = np.exp(np.mean(y_pred_totlgb,0))
          2 br_final = np.exp(np.mean(y_pred_totbr,0))
    ----> 3 rf_final = np.exp(np.mean(y_pred_totrf,0))
    

    NameError: name 'y_pred_totrf' is not defined



```python
y_pred = (lgbm_final*0.70 + br_final*0.215 + rf_final*.15) 
y_pred
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[54], line 1
    ----> 1 y_pred = (lgbm_final*0.70 + br_final*0.215 + rf_final*.15) 
          2 y_pred
    

    NameError: name 'rf_final' is not defined



```python
df_sub = pd.DataFrame(data=y_pred, columns=['COST'])
writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')
df_sub.to_excel(writer,sheet_name='Sheet1', index=False)
writer.save()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[55], line 1
    ----> 1 df_sub = pd.DataFrame(data=y_pred, columns=['COST'])
          2 writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')
          3 df_sub.to_excel(writer,sheet_name='Sheet1', index=False)
    

    NameError: name 'y_pred' is not defined



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
