import numpy as np 
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __model 
    global __locations
    global __data_columns

    __data_columns = None
    __locations = None
    __model = None

    with open('./Artifacts/Data_columns.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    if (__model is None):
        with open('./Artifacts/Home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)

    print("loading saved artifacts...done")


def get_location_name():
    return __locations

def get_data_columns():
    return __data_columns

def get_estimated_price(location, total_sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_name())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) 
   