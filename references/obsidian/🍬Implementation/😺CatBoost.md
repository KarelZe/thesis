Ideas to try out in CatBoost
- look into search space, where did I choose the search space poorly
- use custom weighting factor
- make relatative distance to quote categorical / cut-off at threshold
- Try out `lossguided` growing strategy. Add it to objective.
- Try out `XGBoost` or `lightgbm`
- Think about indicator variables
- Think about interaction features, which e. g.,which features could be summed / multiplied etc.?
- Perform an error analysis. For which classes does CatBoost do so poorly? See some ideas here. https://elitedatascience.com/feature-engineering-best-practices
- Check time consistency (found idea here: https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600/notebook)
> We added 28 new feature above. We have already removed 219 V Columns from correlation analysis done [here](https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id). So we currently have 242 features now. We will now check each of our 242 for "time consistency". We will build 242 models. Each model will be trained on the first month of the training data and will only use one feature. We will then predict the last month of the training data. We want both training AUC and validation AUC to be above `AUC = 0.5`. It turns out that 19 features fail this test so we will remove them. Additionally we will remove 7 D columns that are mostly NAN. More techniques for feature selection are listed [here](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308)
> 

- Could try featuretools. Not sure, if it actually has some advantage. The thing is features are not to fancy here. (Check out this kernel https://www.kaggle.com/code/vbmokin/titanic-featuretools-automatic-fe-fs/notebook?scriptVersionId=43519589)

- Go for even greater ensembles, if stopped out early e. g., 20000. See kaggle notebooks e. g.,:
```python
# https://www.kaggle.com/code/kyakovlev/ieee-fe-for-local-test/notebook
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':80000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                }
```
- add statistics /  group features like min, max, std dev etc.
```python
# found at: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575
temp = df.groupby('card1')['TransactionAmt'].agg(['mean'])   
    .rename({'mean':'TransactionAmt_card1_mean'},axis=1)
df = pd.merge(df,temp,on='card1',how='left')```
```
- add frequency encoding
```python
# found at: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575
temp = df['card1'].value_counts().to_dict()
df['card1_counts'] = df['card1'].map(temp)
```
- Fill missing values with something obvious e. g., `train_df.fillna(-999,inplace=True)`

```python
# found at: https://www.kaggle.com/code/kooaslansefat/tips-tricks-catboost-version
# create new features from newly created categorical variables
age_group_feat_eng = True
bmi_group_feat_eng = True

if age_group_feat_eng:
    train_X, test_X = _group_feature_eng(combined_df, n_train, 'age_group', num_feats)

if bmi_group_feat_eng:
    train_X, test_X = _group_feature_eng(combined_df, n_train, 'bmi_group', num_feats)

def _group_feature_eng(combined_df, n_train, group_var, num_feats):
    
    """
    combined_df: the combined train & test datasets.
    n_train: number of training observations
    group_var: the variable we'd like to group by
    num_feat: numerical features
    
    This function loops through all numerical features, 
    group by the variable and compute new statistics of the numerical features.
    """
    
    grouped = combined_df.groupby(group_var)

    for nf in num_feats:

        combined_df[group_var + '_' + nf + '_max'] = grouped[nf].transform('max')
        combined_df[group_var + '_' + nf + '_min'] = grouped[nf].transform('min')
        combined_df[group_var + '_' + nf + '_mean'] = grouped[nf].transform('mean')
        combined_df[group_var + '_' + nf + '_skew'] = grouped[nf].transform('skew')
        combined_df[group_var + '_' + nf + '_std'] = grouped[nf].transform('std')

    train_X = combined_df.iloc[:n_train]
    test_X = combined_df.iloc[n_train:]
    
    return train_X, test_X
```

Study feature interactions


```python
feature_interaction = [[X.columns[interaction[0]], X.columns[interaction[1]], interaction[2]] for i,interaction in interactions.iterrows()]
feature_interaction_df = pd.DataFrame(feature_interaction, columns=['feature1', 'feature2', 'interaction_strength'])
feature_interaction_df.head(10)
```

Restructure TOC conforming to CRISPDM:

```
1.  Project Scoping / Data Collection
2.  Exploratory Analysis
3.  Data Cleaning
4.  **Feature Engineering**
5.  Model Training (including cross-validation to tune hyper-parameters)
6.  Project Delivery / Insights
```