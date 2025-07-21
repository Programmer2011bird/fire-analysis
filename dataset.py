from iran_extents import iran_provinces_extent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# latitude, longitude : Location of fire
# confidence : fire detection confidence ( 0 - 100 )
# frp : Fire Radiative Power ( MV ) - Energy Output - higher = bigger fire 
# type : 0 - Normal fire, 2 - high confidence ( very likely )
# TODO: Make it so you can import the provinces from a python file and make the preprocess pipeline more flexible for other countries

pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  

def filter_low_confidence(dataframe: pd.DataFrame, provinces_extent: None | dict[str, tuple]) -> pd.DataFrame:
    high_confidence = []

    for index in range(len(dataframe[1])):
        date = dataframe[0][index]
        latitude = np.average(dataframe[1][index]["latitude"])
        longitude = np.average(dataframe[1][index]["longitude"])
        frp = np.round(np.average(dataframe[1][index]["frp"]))
        fire_type = np.median(dataframe[1][index]["type"])
        confidence = np.round(np.average(dataframe[1][index]["confidence"]))
        
        province = ""
        if provinces_extent != None:
            for key, value in provinces_extent.items():
                if (latitude <= value[1] and latitude >= value[0]) and (longitude <= value[3] and longitude >= value[2]):
                    province = key
    
        if confidence > 50 :
            high_confidence.append({
                "date": date,
                "latitude": latitude,
                "longitude": longitude,
                "frp": frp, 
                "province": province
            })

    return pd.DataFrame(high_confidence)

def preprocess_data(country: str, provinces_extent: None | dict[str, tuple]) -> pd.DataFrame:
    country = country.replace(" ", "_").capitalize()
    dataset = pd.read_csv(f"./dataset/2023/modis_2023_{country}.csv")
    dataset = dataset.dropna()
    dataset = pd.DataFrame(dataset.drop(columns=["instrument", "version", "satellite",
                                                 "brightness", "bright_t31", "track", "scan"]))
    
    sorted_by_date = pd.DataFrame(dataset.groupby("acq_time"))
    high_confidence = filter_low_confidence(sorted_by_date, provinces_extent)

    return high_confidence

def analyze_data(country: str, provinces_extend: dict[str, tuple] | None):
    df = preprocess_data(country, provinces_extend)
    if provinces_extend != None:
        province = pd.DataFrame(df.groupby("province").count()["date"]).to_dict()["date"]
    
        plt.subplot(1, 2, 1)
        plt.scatter(x=df["latitude"], y=df["longitude"], 
                s=df["frp"]/2, c=df["frp"])
    
        plt.subplot(1, 2, 2)
        plt.xticks(rotation="vertical")
        plt.bar(x=np.array(list(province.keys())), height=province.values())
        plt.show()

    else:
        plt.scatter(x=df["latitude"], y=df["longitude"], 
                s=df["frp"]/2, c=df["frp"])
        plt.show()

if __name__ == "__main__":
    analyze_data("Iran", iran_provinces_extent)
