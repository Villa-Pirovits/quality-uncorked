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