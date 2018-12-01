# -*- coding: utf-8 -*-



# Examples of cleaning data methods in Python and some introduction into preprocessing

# Libraries
import numpy as np
import pandas as pd
# And also sklearn


# Reading data to DataFrame ===================================================
dataFrame = pd.read_csv("MyData")


# Detecting missing data ======================================================

pd.isnull(dataFrame)
# Replacing specific data into new one
dataFrame.replace(to_replace="New_value", value="Old_value")


# Removing all missing data
dataFrame.dropna()
# Removing missing data from specific columns
dataFrame.dropna(subset=['Column_1'])


# Interpolating data by placing mean values
from sklearn.preprocessing import Imputer
imput = Imputer(missing_values='NaN', strategy='mean', axis=0)
imput = imput.fit(dataFrame)
imputedData = imput.transform(dataFrame.values)


# Dealing with outliers =======================================================

# Get the 98th and 2nd percentile as the limits of our outliers
upperBoundary = np.percentile(dataFrame.values, 98) 
lowerBoundary = np.percentile(dataFrame.values, 2) 
# Filter the outliers from the dataframe
AnotherDataFrame["ColName"].loc[dataFrame["ColName"]>upperBoundary] = upperBoundary 
AnotherDataFrame["ColName"].loc[dataFrame["ColName"]<lowerBoundary] = lowerBoundary

# Handling with categorical data ==============================================

# Unificate names of categorical data

# Whole string lower case
[i.lower() for i in dataFrame["ColName"]]
# First letter capitalised
[i.Capitalize() for i in dataFrame["ColName"]]


# Convert categorical data into integers
from sklearn.preprocessing import LabelEncoder
target_feature = 'Some feature name'
# Using encoder and transform
encoder = LabelEncoder()
enc_Values = encoder.fit_transform(dataFrame[target_feature].values)
dataFrame[target_feature] = pd.Series(enc_Values, index=dataFrame.index)

# Convert categorical data into integers
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
dataFrame = oneHotEncoder.fit_transform(dataFrame).toarray()

# Creating dummy features
dataFrame = pd.get_dummies(dataFrame)

# Scaling features ============================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_dataFrame= sc.fit_transform(train_dataFrame)
test_dataFrame= sc.transform(test_dataFrame)


































