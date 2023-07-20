import pandas as pd

def get_wine():
    #Pull red and white wines from data world
    red_wine = pd.read_csv('https://query.data.world/s/ahz66k3pnpfiuknaaql4n4m4jtnliw?dws=00000')
    white_wine = pd.read_csv('https://query.data.world/s/y7kdf76jp3jjg56g45jdeew6726zfw?dws=00000')
    
    #Add a column that shows red or white
    red_wine['type'] = 'red'
    white_wine['type'] = 'white'
    
    #Create and return a dataframe
    df = pd.concat([red_wine, white_wine])
    return df