import numpy as np
import pandas as pd
import sys
lib=__import__('lib1772576')

files = sys.argv[1:]
train,test= lib.read_files(files)
train= lib.drop_unnecessary_columns(train)
train,test =lib.convert_categorical_to_numerical(train,test)
R2score, theta = lib.log_linear_model(train,test) 
train_score, test_score = lib.cross_validation(train)
print ('R2:')
print (R2score)
print ('Cross Validation train and test scores respectively')
print (train_score)
print (test_score)
lib.find_outliers(files)
