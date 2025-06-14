import pandas as pd
import numpy as np

# latitude, longitude : Location of fire
# brightness : Fire temperature ( in Kelvin ) ( I don't think so )
# confidence : fire detection confidence ( 0 - 100 )
# frp : Fire Radiative Power ( MV ) - Energy Output - higher = bigger fire 
# type : 0 - Normal fire, 2 - high confidence ( very likely )

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
 
dataset = pd.read_csv("./dataset/2023/modis_2023_Iran.csv")
dataset = dataset.dropna()
dataset = pd.DataFrame(dataset.drop(columns=["instrument", "version", "satellite",
                                             "brightness", "bright_t31", "track", "scan"]))

sorted_by_date = pd.DataFrame(dataset.groupby("acq_time"))

for index in range(len(sorted_by_date[0])):
    print(sorted_by_date[0][index])
    break

for index in range(len(sorted_by_date[1])):
    print(sorted_by_date[1][index])
    
    if index == 10:
        break
