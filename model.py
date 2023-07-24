from wrangle import X_y_split, cluster_to_dummy, create_cluster_models
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
from scipy import stats

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

###################################  ###################################


def linear_regression_two_features():


    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()

    X_train_2, X_train_3, X_train_4 = cluster_to_dummy(X_train)
    X_validate_2, X_validate_3, X_validate_4 = cluster_to_dummy(X_validate)

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





def linear_regression_three_features():


    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()

    X_train_2, X_train_3, X_train_4 = cluster_to_dummy(X_train)
    X_validate_2, X_validate_3, X_validate_4 = cluster_to_dummy(X_validate)

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
def linear_regression_four_features():



    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()

    X_train_2, X_train_3, X_train_4 = cluster_to_dummy(X_train)
    X_validate_2, X_validate_3, X_validate_4 = cluster_to_dummy(X_validate)

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
def quadratic_model():


    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()

    X_train_2, X_train_3, X_train_4 = cluster_to_dummy(X_train)
    X_validate_2, X_validate_3, X_validate_4 = cluster_to_dummy(X_validate)

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
def test():
    
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()

    X_train_2, X_train_3, X_train_4 = cluster_to_dummy(X_train)
    X_validate_2, X_validate_3, X_validate_4 = cluster_to_dummy(X_validate)
    
    
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


    
    
    
def anova_test_2_feat():
    
    alpha = 0.05
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    
    
    x_2 = recombined_train[recombined_train['2_cluster']==0]
    y_2 = recombined_train[recombined_train['2_cluster']==1]
    z_2 = recombined_train[recombined_train['2_cluster']==2]
    w_2 = recombined_train[recombined_train['2_cluster']==3]

    x_3 = recombined_train[recombined_train['3_cluster']==0]
    y_3 = recombined_train[recombined_train['3_cluster']==1]
    z_3 = recombined_train[recombined_train['3_cluster']==2]
    w_3 = recombined_train[recombined_train['3_cluster']==3]

    x_4 = recombined_train[recombined_train['4_cluster']==0]
    y_4 = recombined_train[recombined_train['4_cluster']==1]
    z_4 = recombined_train[recombined_train['4_cluster']==2]
    w_4 = recombined_train[recombined_train['4_cluster']==3]
    
    f2, p2 = stats.f_oneway(x_2.quality,y_2.quality,z_2.quality,w_2.quality)
    
    f3, p3 = stats.f_oneway(x_3.quality,y_3.quality,z_3.quality,w_3.quality)
    
    f4, p4 = stats.f_oneway(x_4.quality,y_4.quality,z_4.quality,w_4.quality)

    return print(f'''{p2} < {alpha} \
        We can reject the Null Hypothesis. The four clusters types in the feature clustersed are different from each other.''')







def anova_test_3_feat():
    
    alpha = 0.05
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    
    
    x_2 = recombined_train[recombined_train['2_cluster']==0]
    y_2 = recombined_train[recombined_train['2_cluster']==1]
    z_2 = recombined_train[recombined_train['2_cluster']==2]
    w_2 = recombined_train[recombined_train['2_cluster']==3]

    x_3 = recombined_train[recombined_train['3_cluster']==0]
    y_3 = recombined_train[recombined_train['3_cluster']==1]
    z_3 = recombined_train[recombined_train['3_cluster']==2]
    w_3 = recombined_train[recombined_train['3_cluster']==3]

    x_4 = recombined_train[recombined_train['4_cluster']==0]
    y_4 = recombined_train[recombined_train['4_cluster']==1]
    z_4 = recombined_train[recombined_train['4_cluster']==2]
    w_4 = recombined_train[recombined_train['4_cluster']==3]
    
    f2, p2 = stats.f_oneway(x_2.quality,y_2.quality,z_2.quality,w_2.quality)
    
    f3, p3 = stats.f_oneway(x_3.quality,y_3.quality,z_3.quality,w_3.quality)
    
    f4, p4 = stats.f_oneway(x_4.quality,y_4.quality,z_4.quality,w_4.quality)

    return print(f'''{p3} < {alpha} \
        We can reject the Null Hypothesis. The four clusters types in the feature clustersed are different from each other.''')







def anova_test_4_feat():
    
    alpha = 0.05
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    
    
    x_2 = recombined_train[recombined_train['2_cluster']==0]
    y_2 = recombined_train[recombined_train['2_cluster']==1]
    z_2 = recombined_train[recombined_train['2_cluster']==2]
    w_2 = recombined_train[recombined_train['2_cluster']==3]

    x_3 = recombined_train[recombined_train['3_cluster']==0]
    y_3 = recombined_train[recombined_train['3_cluster']==1]
    z_3 = recombined_train[recombined_train['3_cluster']==2]
    w_3 = recombined_train[recombined_train['3_cluster']==3]

    x_4 = recombined_train[recombined_train['4_cluster']==0]
    y_4 = recombined_train[recombined_train['4_cluster']==1]
    z_4 = recombined_train[recombined_train['4_cluster']==2]
    w_4 = recombined_train[recombined_train['4_cluster']==3]
    
    f2, p2 = stats.f_oneway(x_2.quality,y_2.quality,z_2.quality,w_2.quality)
    
    f3, p3 = stats.f_oneway(x_3.quality,y_3.quality,z_3.quality,w_3.quality)
    
    f4, p4 = stats.f_oneway(x_4.quality,y_4.quality,z_4.quality,w_4.quality)

    return print(f'''{p4} < {alpha} \
        We can reject the Null Hypothesis. The four clusters types in the feature clustersed are different from each other.''')



def scatter_2_cluster():
     
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    cluster_2 = pd.crosstab(recombined_train.quality, recombined_train['2_cluster'])
    display(cluster_2)
    sns.scatterplot(cluster_2)
    
    
    
def scatter_3_cluster():
     
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    cluster_3 = pd.crosstab(recombined_train.quality, recombined_train['3_cluster'])
    display(cluster_3)
    sns.scatterplot(cluster_3)
    
    
    
def scatter_4_cluster():
     
    X_train, y_train, X_validate, y_validate, X_test, y_test = create_cluster_models()
    
    
    pd.DataFrame(y_train)
    recombined_train = pd.concat([X_train, y_train], axis=1)
    cluster_4 = pd.crosstab(recombined_train.quality, recombined_train['4_cluster'])
    display(cluster_4)
    sns.scatterplot(cluster_4)