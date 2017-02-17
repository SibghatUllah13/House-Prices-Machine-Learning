import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def read_files(file_names): #loads train and test data in RAM
    train = pd.read_csv(file_names[0], index_col=0)
    test = pd.read_csv(file_names[1], index_col=0) 
    return train,test

def drop_unnecessary_columns(train): # Drop columns having too many NaNs
    train.columns[train.isnull().sum() > 500] 
    train.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature','FireplaceQu'], axis=1, inplace=True)
    train.fillna(train.median(axis=0), inplace=True) #Fill remaining NANs with Median
    return train

def convert_categorical_to_numerical(train,test):
    
    # First, do that on training dataset
    columns = train.select_dtypes([object]).columns
    for column in columns:
        train[column]=train[column].astype('category')
    cat_columns = train.select_dtypes(['category']).columns
    train[cat_columns] = train[cat_columns].apply(lambda x: x.cat.codes)
    
    # Now,do that on test dataset
    columns = test.select_dtypes([object]).columns
    for column in columns:
        test[column]=test[column].astype('category')
    cat_columns = test.select_dtypes(['category']).columns
    test[cat_columns] = test[cat_columns].apply(lambda x: x.cat.codes)
    return train,test


def log_linear_model(train,test): # fit a linear model to the data
    from sklearn import linear_model
    lr = linear_model.LinearRegression(normalize= True,fit_intercept=True)
    X, y = train.values[:,:-1], train.values[:,-1]
    ylog = np.log(y)
    lr.fit(X, ylog)
    test = test.loc[:, train.columns[:-1]] # select the variables that are in our training model
    test = test.fillna(test.median(axis=0), inplace=True) # fill NaNs in test data
    preds = pd.DataFrame({"SalePrice":lr.predict(test)}, index=test.index)
    preds.SalePrice = np.exp(preds.SalePrice)
    preds.to_csv("pred.csv") 
    return lr.score(X, ylog),lr.coef_ 

def cross_validation(train): #Making Sure my Model doesn't allow overfitting
    X, y = train.values[:,:-1], train.values[:,-1]
    from sklearn import cross_validation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    from sklearn import linear_model
    lr = linear_model.LinearRegression(normalize= True,fit_intercept=True)
    lr.fit(X_train, np.log(y_train))
    return lr.score(X_train, np.log(y_train)),lr.score(X_test, np.log(y_test))

def find_outliers(files): #Ploting Outliers
    tr,te= read_files(files)
    train_num = tr.select_dtypes([np.number])
    plt.ion()
    train_num_n = (train_num - train_num.mean())/train_num.std()
    train_num_n.boxplot(vert=False, return_type='axes' )
    plt.title("Outliers")
    plt.savefig("1772576.png")
