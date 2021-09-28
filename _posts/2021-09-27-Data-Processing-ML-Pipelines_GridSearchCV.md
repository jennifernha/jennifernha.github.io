---
title: "Data Preprocessing, ML Pipelines, and GridSearchCV"
date: 2021-09-27 07:49:28 -0400
categories: Data Preprocessing, GridSearchCV
---

During my third project at FIS (Flatiron School), I worked on identifying the best performing machine learning model to determine whether an Airbnb property is valuable or not. After the [data cleaning and preparation process](https://jennifernha.github.io/data/cleaning/preparation/Data-Cleaning-and-Preparation/), I prerocessed the dataset, built machine learning pipelines for different models, and used GridSearchCV for parameter tuning. 

## Preprocessing
After train-test splitting the dataset, I had to preprocess the data in order to transform some columns so that they can be used for different machine learning models. Below code from my instructor allows us to categorize all columns depending on the value type. The code can be modified based on the value types and you can also modify the max frequency. For example, I used 6 as it made sense to assign which columns to run one hot encoder transformer and frequency transformer.
```ruby
# Below clode is from Lindsey that creates empty lists for different column types
ohe_cols = [] 
freq_cols = []
num_cols = [] 

# Loop through the columns and append col to proper list
for col in X_train.columns:
    # Numeric columns
    if X_train[col].dtype in ['float64', 'int64']:
        num_cols.append(col)
        
    # Columns with fewer than 5 unique values
    elif len(X_train[col].unique()) < 6:
        ohe_cols.append(col)
            
    # Columns with more than 5 unique values
    else:
        freq_cols.append(col)
```
Below is the different column transformers I used for my project. There are obviously more transformers out there but I will share the three I used as an example.
```ruby
num_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

ohe_transformer = Pipeline(steps=[
    ('ohe_imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('ohencoder', OneHotEncoder(handle_unknown='ignore'))])

freq_transformer = Pipeline(steps=[
    ('freq_imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('freq_enc', ce.count.CountEncoder(normalize=True, 
                                       handle_unknown=0,
                                       min_group_size=0.001,
                                       min_group_name='Other'))])
```
Finally, the below preprocessor allows us to preprocess the columns in my dataset based on how I categorized them in the empty list in the very first step.
```ruby
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('ohe', ohe_transformer, ohe_cols),
        ('freq', freq_transformer, freq_cols)])
```        

## Pipeline 
Building pipelines for machine learning models is rather simple than one can imagine. Below are examples for different models from my project.
```ruby
clf_logreg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))])
                      
clf_dt = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', DecisionTreeClassifier(class_weight='balanced'))]) 

clf_rf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(random_state=42,class_weight='balanced'))])

clf_abt = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', AdaBoostClassifier())])
                         
clf_gbm = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', GradientBoostingClassifier())])
```

## GridSearchCV
As the last step of my project, I worked on identifying the parameters that maximizes accuracy of each model. There are other scoring metrics besides accuracy, and they can be used for model evaluation by all means. 
```ruby
# Code borrowed and modified from https://medium.com/analytics-vidhya/ml-pipelines-using-scikit-learn-and-gridsearchcv-fe605a7f9e05
# Code borrowed from https://medium.com/swlh/the-hyperparameter-cheat-sheet-770f1fed32ff

# Set grid search params
param_range = [9, 10]
param_range_fl = [1.0, 0.5]

grid_params_lr = [{'classifier__penalty': ['l1', 'l2'],
        'classifier__C': param_range_fl,
        'classifier__solver': ['liblinear']}] 

grid_params_rf = [{'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': param_range,
        'classifier__min_samples_split': param_range[1:]}]

grid_params_abt = [{'classifier__n_estimators': [100, 200], 
        'classifier__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5]}]


grid_params_gbm = [{'classifier__n_estimators': [5, 50, 100, 200], 
        'classifier__max_depth': [1, 3, 5, 7, 9],
        'classifier__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5]}]

# Construct grid searches
jobs = -1

LR = GridSearchCV(estimator=clf_logreg,
            param_grid=grid_params_lr,
            scoring='accuracy',
            cv=5) 

RF = GridSearchCV(estimator=clf_rf,
            param_grid=grid_params_rf,
            scoring='accuracy',
            cv=5, 
            n_jobs=jobs)

AB = GridSearchCV(estimator=clf_abt,
            param_grid=grid_params_abt,
            scoring='accuracy',
            cv=5, 
            n_jobs=jobs)

GBM = GridSearchCV(estimator=clf_gbm,
            param_grid=grid_params_gbm,
            scoring='accuracy',
            cv=5, 
            n_jobs=jobs)

# List of pipelines for iterating through each of them
grids = [LR, RF, ABT, GBM]

# Creating a dict for our reference
grid_dict = {0: 'Logistic Regression', 
        1: 'Random Forest',
        2: 'AdaBoost',
        3: 'Gradient Boosting'}

# Fit the grid search objects
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    gs.fit(X_train, y_train)
    print('Best params are : %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
```
The code above will not only let you know the best parameters for each model along with test/train but also the best performing classifier. The result would look something like this:
```ruby
```
Performing model optimizations...

Estimator: Logistic Regression
Best params are : {'classifier__C': 1.0, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'}
Best training accuracy: 0.682
Test set accuracy score for best params: 0.683 

Estimator: Random Forest
Best params are : {'classifier__criterion': 'entropy', 'classifier__max_depth': 10, 'classifier__min_samples_split': 10}
Best training accuracy: 0.744
Test set accuracy score for best params: 0.745 

Estimator: AdaBoost
Best params are : {'classifier__learning_rate': 0.1, 'classifier__n_estimators': 100}
Best training accuracy: 0.627
Test set accuracy score for best params: 0.773 

Estimator: Gradient Boosting
Best params are : {'classifier__learning_rate': 0.1, 'classifier__max_depth': 7, 'classifier__n_estimators': 200}
Best training accuracy: 0.795
Test set accuracy score for best params: 0.801 

Classifier with best test set accuracy: Gradient Boosting
```
```
