#standard ds imports
import pandas as pd
import numpy as np
import os

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

from wrangle import split_wine

############################## density vs residual sugar ##############################

def print_den_v_sug():

    train, validate, test = split_wine()

    print(f'density vs residual sugar')
    # plots                                         ## adjust the fig size to preference and fit
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))   ###### cols change when i un comment the plots #########

    #sns.lineplot(data= train, x='density' , y='residual sugar', ax= ax[0])
    sns.scatterplot(data= train, x='density' , y='residual sugar', ax= ax, hue='white') ### change the ax when you adjust the col row
    #sns.kdeplot(data= train, x='density' , y='residual sugar', ax= ax[2])

    plt.tight_layout()

    # save visual to file path
    # explore_.save_visuals(fig=fig, viz_name=f"{col[0]}_vs_{col[1]}", folder_name= 2)

    plt.show()




    ############################## density vs alcohol ##############################


def print_den_v_alc():

    train, validate, test = split_wine()

    print(f'density vs alcohol')
    # plots                                         ## adjust the fig size to preference and fit
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,5))   ###### cols change when i un comment the plots #########

    #sns.lineplot(data= train, x='density' , y='residual sugar', ax= ax[0])
    sns.scatterplot(data= train, x='density' , y='alcohol', ax= ax, hue='white') ### change the ax when you adjust the col row
    #sns.kdeplot(data= train, x='density' , y='residual sugar', ax= ax[2])

    plt.tight_layout()

    # save visual to file path
    # explore_.save_visuals(fig=fig, viz_name=f"{col[0]}_vs_{col[1]}", folder_name= 2)

    plt.show()



       ############################## density vs alcohol ##############################



def ph_level():

    train, validate, test = split_wine()

    #pllotting the ph level 
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
    # sns.barplot(data= train, x=train.white , y=train.pH, ax= ax[0])
    sns.boxplot(data= train, x=train.white , y=train.pH, ax= ax)
    # sns.stripplot(data= train, x=train.white , y=train.pH, ax= ax[2])
    plt.tight_layout()
    
    
    

    
 ############################## density vs chlorides ##############################
    
def print_den_v_chlo():
    
    # splitting data
    train, validate, test = split_wine()
    
    # setting red wine df
    red = train[train['white']==0]
    
    # settig white wine df
    white = train[train['white']==1]
    
    print(f'density vs chlorides')
    # plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,5))   

    sns.scatterplot(data=red, x='density' , y='chlorides', ax= ax[0], hue='white')
    ax[0].set_title('Red Wines')

    sns.scatterplot(data=white, x='density' , y='chlorides', ax= ax[1], hue='white')
    ax[1].set_title('White Wines')
    # sns.kdeplot(data= train, x='density' , y='chlorides', ax= ax[2])

    plt.tight_layout()

    # save visual to file path
    # explore_.save_visuals(fig=fig, viz_name=f"{col[0]}_vs_{col[1]}", folder_name= 2)

    plt.show()
