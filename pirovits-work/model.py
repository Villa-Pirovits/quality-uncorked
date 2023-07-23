import acquire
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def train_validate_test(df, target):
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
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target, 'type'])
    y_train = train[target]
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target, 'type'])
    y_validate = validate[target]
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target, 'type'])
    y_test = test[target]
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def train_validate_test_dummy(df, target):
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
    df_dummies = pd.get_dummies(df.type, drop_first=True)
    # Concatenate the original DataFrame and the dummy variables DataFrame
    df = pd.concat([df, df_dummies], axis=1)
    
    df = df.drop(columns=['type'])
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    


def scale_data(X_train, X_validate, X_test):
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    
    #Turn array back into dataframe and rename to original columns
    columns = X_train.columns #List of Columns
    numbers = [0,1,2,3,4,5,6,7,8,9,10,11] #List of numbers for the scaled np array I'm converting into a dataframe
    zipped= dict(zip(numbers, columns))
    X_train_scaled = pd.DataFrame(X_train_scaled).rename(columns=zipped)
    X_validate_scaled = pd.DataFrame(X_validate_scaled).rename(columns=zipped)
    X_test_scaled = pd.DataFrame(X_test_scaled).rename(columns=zipped)
    
    return X_train_scaled, X_validate_scaled, X_test_scaled

def baseline(y_train, y_validate, y_test):
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
    
#Create dataframes with the clusters
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


 # Perform clustering on two features   
def linear_regression_two_features(X_train_2, X_validate_2, y_train, y_validate, y_test):
    # turn series into dataframes to append new columns with predicted values
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    X_train_2 = X_train_2.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    X_validate_2 = X_validate_2.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    #. Create the model object
    lm = LinearRegression()

    #. Fit to training and specify column in y_train since it is now a series
    lm.fit(X_train_2, y_train.quality)

    # predict
    y_train['quality_pred_lm'] = lm.predict(X_train_2)

    # RMSE
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lm) ** (1/2)

    # predict validate
    y_validate['quality_pred_lm'] = lm.predict(X_validate_2)

    #Validate RMSE 
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lm) ** (1/2)

    print('RMSE for OLS using LinearRegression\nTraining/In-Sample: ', rmse_train,
         '\nValidation/Out-of-Sample: ', rmse_validate)
    
 #Perform clustering on the three features   

def linear_regression_three_features(X_train_3, X_validate_3, y_train, y_validate, y_test):
    
    # turn series into dataframes to append new columns with predicted values
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    X_train_3 = X_train_3.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    X_validate_3 = X_validate_3.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    #. Create the model object
    lm = LinearRegression()

    #. Fit to training and specify column in y_train since it is now a series
    lm.fit(X_train_3, y_train.quality)

    # predict
    y_train['quality_pred_lm'] = lm.predict(X_train_3)

    # RMSE
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lm) ** (1/2)

    # predict validate
    y_validate['quality_pred_lm'] = lm.predict(X_validate_3)

    #Validate RMSE 
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lm) ** (1/2)

    print('RMSE for OLS using LinearRegression\nTraining/In-Sample: ', rmse_train,
         '\nValidation/Out-of-Sample: ', rmse_validate)
    

#Perform clustering on the 4 features  
def linear_regression_four_features(X_train_4, X_validate_4, y_train, y_validate, y_test):

    # turn series into dataframes to append new columns with predicted values
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    X_train_4 = X_train_4.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    X_validate_4 = X_validate_4.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    #. Create the model object
    lm = LinearRegression()

    #. Fit to training and specify column in y_train since it is now a series
    lm.fit(X_train_4, y_train.quality)

    # predict
    y_train['quality_pred_lm'] = lm.predict(X_train_4)

    # RMSE
    rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_lm) ** (1/2)

    # predict validate
    y_validate['quality_pred_lm'] = lm.predict(X_validate_4)

    #Validate RMSE 
    rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_lm) ** (1/2)

    print('RMSE for OLS using LinearRegression\nTraining/In-Sample: ', rmse_train,
         '\nValidation/Out-of-Sample: ', rmse_validate)
    

#Quadratic Model
def quadratic_model(X_train_2, X_validate_2, y_train, y_validate, y_test):
    X_train_2 = X_train_2.rename(columns={1:'cluster1', 2:'cluster2', 3:'cluster3'})
    X_validate_2 = X_validate_2.rename(columns={1:'cluster1', 2:'cluster2', 3:'cluster3'})

    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)


    for i in range(2,3):
        # make the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=i)

        # fit and transform X_train_scaled
        X_train_degree2 = pf.fit_transform(X_train_2)

        # transform X_validate_scaled & X_test_scaled
        X_validate_degree2 = pf.transform(X_validate_2)
        #X_test_degree2_mvp = pf.transform(X_test_mvp)

        # create the model object
        lm2 = LinearRegression()

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lm2.fit(X_train_2, y_train.quality)

        # predict train
        y_train['quality_pred_poly'] = lm2.predict(X_train_2)

        # evaluate: rmse
        rmse_train = mean_squared_error(y_train.quality, y_train.quality_pred_poly)**(1/2)

        # predict validate
        y_validate['quality_pred_poly'] = lm2.predict(X_validate_2)

        # evaluate: rmse
        rmse_validate = mean_squared_error(y_validate.quality, y_validate.quality_pred_poly)**(1/2)

        print("RMSE for Polynomial Model, degrees=", i, "\nTraining/In-Sample: ", rmse_train, 
              "\nValidation/Out-of-Sample: ", rmse_validate)

#Run the test     
def test(X_train_2, X_test_scaled, X_test, y_train, y_validate, y_test):
    #Turn Series into dataframes
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)

    #Create the two feature dataframe scaled for the kmeans
    #X_test_2_features = X_test_scaled[['density', 'alcohol']]
    #X_test['2_cluster'] = kmeans2.predict(X_test_2_features) Left out for scope issues
    #Create a dummy dataframe for the test set needed for the linear regression
    df_dummies = pd.get_dummies(X_test['2_cluster'], drop_first=True)
    X_test = pd.concat([X_test, df_dummies], axis=1)
    X_test.drop(columns='2_cluster', inplace=True)
    X_train_2 = X_train_2.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})

    # Perform clustering on two features   
    #def linear_regression_final(X_test_2 y_test):
    # turn series into dataframes to append new columns with predicted values
    y_test = pd.DataFrame(y_test)

    X_test = X_test.rename(columns={1: 'cluster_1', 2: 'cluster_2', 3: 'cluster_3'})
    #. Create the model object
    lm = LinearRegression()

    #. Fit to training and specify column in y_train since it is now a series
    lm.fit(X_train_2, y_train.quality)

    # predict
    y_test['quality_pred_lm'] = lm.predict(X_test)

    # RMSE
    rmse_test = mean_squared_error(y_test.quality, y_test.quality_pred_lm) ** (1/2)

    print('RMSE for OLS using LinearRegression\nTest: ', rmse_test)


#Clusters on 2 three and four features and adds those results to the X_train, X_validate, and X_test 
def create_cluster_models(X_train, X_validate, X_test, X_train_scaled, X_validate_scaled, X_test_scaled):
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
    
    return X_train, X_validate, X_test
    