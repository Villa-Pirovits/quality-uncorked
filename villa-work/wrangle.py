
import os
import pandas as pd
import numpy as np


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



################################### nulls by column ###################################



def nulls_by_col():

    df = acquire_wine()


    num_missing = df.isnull().sum() 
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing




################################### nulls by row ###################################



def nulls_by_row():

    # Creates the df from the function to bring the dataframe
    df = acquire_wine()
    # creates mask
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing



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


    df = df.drop(columns=['quality'], inplace=True)

    return df


