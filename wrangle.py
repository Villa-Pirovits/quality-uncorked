

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

################################### acquire zillow data ###################################




def acquire_wine():
    '''
    this code brings in both the red and white wine datasets
    '''
    # Brings in the red wine data set
    red = pd.read_csv('https://query.data.world/s/b5kzevnjtzdhjkca2cki3z27jgfl47?dws=00000')

    # Brings in the white wine data set
    white = pd.read_csv('https://query.data.world/s/34hlwmdrmsimbmrzz2ojgtq2wgkvdm?dws=00000')

    # adds white type column
    white['type'] = 'white'

    # adds white type column
    red['type'] = 'red'

    # concats the two dataframes
    df = pd.concat([red, white])

    return df





################################### cleaning the data ###################################



def clean_wine():

    # Creates the df from the function to bring the dataframe
    df = acquire_wine()

    # Search for duplicates in the entire DataFrame (based on all columns)
    duplicates = df.duplicated()

    # Drops the duplicates with the mask 
    df = df[duplicates==False]

    # Drops any values that are null
    df = df[df.isnull()==False]
 
    # Resets the index after dropping the diupicate rows
    df = df.reset_index(drop=True)


    return df


    


def split_wine():
    '''
    this function takes in a dataframe and splits it into 3 samples,
    a test, which is 20% of the entire dataframe,
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe.
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    '''
    # create df
    df = clean_wine()

    # create target
    # target = df.quality

    

    df_dummies = pd.get_dummies(df.type, drop_first=True)
    # Concatenate the original DataFrame and the dummy variables DataFrame
    df = pd.concat([df, df_dummies], axis=1)
    
    df = df.drop(columns=['type'])
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    
    return train, validate , test




    
def X_y_split():

    
    train, validate , test = split_wine()
    
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=['quality'])
    y_train = train['quality']
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=['quality'])
    y_validate = validate['quality']
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=['quality'])
    y_test = test['quality']
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test





def scale_data():

    X_train, y_train, X_validate, y_validate, X_test, y_test =  X_y_split()

    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    #Turn array back into dataframe and rename to original columns
    columns = X_train.columns #List of Columns
    numbers = list(range(0, len(X_train.columns))) #List of numbers for the scaled np array I'm converting into a dataframe
    zipped= dict(zip(numbers, columns))
    X_train_scaled = pd.DataFrame(X_train_scaled).rename(columns=zipped)
    X_validate_scaled = pd.DataFrame(X_validate_scaled).rename(columns=zipped)
    X_test_scaled = pd.DataFrame(X_test_scaled).rename(columns=zipped)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled







##################################### create clusters #####################################



def create_cluster_models():

    X_train, y_train, X_validate, y_validate, X_test, y_test =  X_y_split()

    X_train_scaled, X_validate_scaled, X_test_scaled = scale_data()

    X_train_2_features = X_train_scaled[['density', 'alcohol']]
    X_validate_2_features = X_validate_scaled[['density', 'alcohol']]


    kmeans2 = KMeans(n_clusters=4)
    kmeans2.fit(X_train_2_features)

    X_train['2_cluster'] = kmeans2.predict(X_train_2_features)
    X_validate['2_cluster'] = kmeans2.predict(X_validate_2_features)

    X_test_2_features = X_test_scaled[['density', 'alcohol']]
    X_test['2_cluster'] = kmeans2.predict(X_test_2_features)


    X_train_3_features = X_train_scaled[['residual sugar', 'total sulfur dioxide', 'alcohol']]
    X_train_3_features
    X_validate_3_features = X_validate_scaled[['residual sugar', 'total sulfur dioxide', 'alcohol']]
    X_validate_3_features

    kmeans3 = KMeans(n_clusters=4)
    kmeans3.fit(X_train_3_features)

    X_train['3_cluster'] = kmeans3.predict(X_train_3_features)
    X_validate['3_cluster'] = kmeans3.predict(X_validate_3_features)


    X_train_4_features = X_train_scaled[['volatile acidity', 'chlorides', 'density', 'alcohol']]
    X_train_4_features
    X_validate_4_features = X_validate_scaled[['volatile acidity', 'chlorides', 'density', 'alcohol']]
    X_validate_4_features

    kmeans4 = KMeans(n_clusters=4)
    kmeans4.fit(X_train_4_features)

    X_train['4_cluster'] = kmeans4.predict(X_train_4_features)
    X_validate['4_cluster'] = kmeans4.predict(X_validate_4_features)
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    


################################### dummy clusters ###################################

def cluster_to_dummy(X):  
    
    feat_to_dummy = ['2_cluster', '3_cluster', '4_cluster']
    cluster_dummy_list = []

    for feat in feat_to_dummy:
        # creating a dummy column for the current feature
        df_dummies = pd.get_dummies(X[feat], drop_first=True)

        # Concatenate the original DataFrame and the dummy variables DataFrame
        df = pd.concat([X, df_dummies], axis=1)

        # dropping the original feature
        df.drop(columns=feat_to_dummy, inplace=True)

        cluster_dummy_list.append(df)
        
    return cluster_dummy_list





def baseline():

    X_train, y_train, X_validate, y_validate, X_test, y_test =  train_validate_test_dummy()

    # turn series into dataframes to append new columns with predicted values
    y_train_mvp = pd.DataFrame(y_train)
    y_validate_mvp = pd.DataFrame(y_validate)
    y_test_mvp = pd.DataFrame(y_test)

    # 1. Predict based on mean
    quality_pred_mean = y_train_mvp['quality'].mean()
    y_train_mvp['quality_pred_mean'] = quality_pred_mean
    y_validate_mvp['quality_pred_mean'] = quality_pred_mean

    # 2. Do same for median
    quality_pred_median_mvp = y_train_mvp['quality'].median()
    y_train_mvp['quality_pred_median'] = quality_pred_median_mvp
    y_validate_mvp['quality_pred_median'] = quality_pred_median_mvp

    # 3.  RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train_mvp.quality, y_train_mvp.quality_pred_mean) ** (1/2)
    rmse_validate = mean_squared_error(y_validate_mvp.quality, y_validate_mvp.quality_pred_mean) ** (1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2)) 

    # 4.  RMSE of tax_value_pred_median
    rmse_train = mean_squared_error(y_train_mvp.quality, y_train_mvp.quality_pred_median) ** (1/2)
    rmse_validate = mean_squared_error(y_validate_mvp.quality, y_validate_mvp.quality_pred_median) ** (1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    



