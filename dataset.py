import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# latitude, longitude : Location of fire
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
high_confidence = []
iran_provinces_extent = {
    "Alborz":                     (35.7, 36.1, 50.8, 51.8),
    "Ardabil":                    (37.7, 39.7, 47.5, 48.9),
    "East Azerbaijan":            (36.7, 39.8, 45.6, 47.8),
    "West Azerbaijan":            (35.0, 39.1, 44.6, 46.9),
    "Bushehr":                    (27.0, 30.0, 50.5, 53.1),
    "Chaharmahal & Bakhtiari":    (30.4, 32.3, 49.8, 51.8),
    "Fars":                       (27.0, 31.0, 50.5, 55.5),
    "Gilan":                      (36.5, 38.5, 48.9, 50.5),
    "Golestan":                   (36.6, 38.2, 54.1, 55.8),
    "Hamadan":                    (33.5, 35.2, 47.8, 49.4),
    "Hormozgan":                  (25.5, 28.8, 53.4, 59.3),
    "Ilam":                       (32.6, 34.2, 45.8, 47.8),
    "Isfahan":                    (31.5, 34.0, 49.9, 55.0),
    "Kerman":                     (26.9, 31.6, 54.5, 59.5),
    "Kermanshah":                 (33.8, 35.2, 45.6, 47.7),
    "Khuzestan":                  (30.8, 32.8, 47.4, 50.8),
    "Kohgiluyeh & Boyer-Ahmad":   (30.6, 31.9, 50.1, 51.9),
    "Kurdistan":                  (34.7, 36.5, 45.5, 47.6),
    "Lorestan":                   (32.6, 34.5, 47.5, 49.9),
    "Markazi":                    (33.1, 35.0, 49.4, 50.6),
    "Mazandaran":                 (35.7, 36.8, 50.5, 54.0),
    "North Khorasan":             (36.5, 38.1, 55.8, 58.5),
    "Razavi Khorasan":            (34.9, 37.8, 56.5, 61.5),
    "South Khorasan":             (30.6, 34.4, 57.7, 61.5),
    "Qazvin":                     (35.5, 36.5, 49.0, 50.5),
    "Qom":                        (34.0, 35.0, 50.8, 51.8),
    "Semnan":                     (34.0, 36.5, 52.0, 56.5),
    "Sistan & Baluchestan":       (25.1, 31.0, 59.1, 63.4),
    "Tehran":                     (35.4, 36.2, 50.7, 51.8),
    "Yazd":                       (30.9, 32.2, 53.5, 56.5),
    "Zanjan":                     (35.8, 36.7, 47.2, 49.0),
}

for index in range(len(sorted_by_date[1])):
    date = sorted_by_date[0][index]
    latitude = np.average(sorted_by_date[1][index]["latitude"])
    longitude = np.average(sorted_by_date[1][index]["longitude"])
    frp = np.round(np.average(sorted_by_date[1][index]["frp"]))
    fire_type = np.median(sorted_by_date[1][index]["type"])
    confidence = np.round(np.average(sorted_by_date[1][index]["confidence"]))
    
    province = ""

    for key, value in iran_provinces_extent.items():
        if (latitude <= value[1] and latitude >= value[0]) and (longitude <= value[3] and longitude >= value[2]):
            province = key

    if confidence > 50 and fire_type == 2:
        high_confidence.append({
            "date": date,
            "latitude": latitude,
            "longitude": longitude,
            "frp": frp, 
            "province": province
        })

high_confidence = pd.DataFrame(high_confidence)
province = pd.DataFrame(high_confidence.groupby("province").count()["date"]).to_dict()["date"]

plt.subplot(1, 2, 1)
plt.scatter(x=high_confidence["latitude"], y=high_confidence["longitude"], 
            s=high_confidence["frp"]/2, c=high_confidence["frp"])

plt.subplot(1, 2, 2)
plt.xticks(rotation="vertical")
plt.bar(x=np.array(list(province.keys())), height=province.values())
plt.show()
