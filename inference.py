import numpy as np
import pandas as pd
import tensorflow.keras as keras
import pickle
import json
import matplotlib.pyplot as plt


def plotit(sample):
    plt.imshow(sample, cmap='gray', interpolation='nearest', aspect='auto')
    plt.show()


def transform_set(scalar, data, fit=False):
    original_shape = data.shape 
    reshaped = data.reshape(-1, data.shape[-1]) 

    if fit:
        reshaped = scalar.fit_transform(reshaped)
    else:
        reshaped = scalar.transform(reshaped)
        
    reshaped = reshaped.reshape(original_shape)

    return reshaped


# dataset_dir = 'dataset/original_data'
dataset_dir = 'dataset/our_data'
# filename = '10sec.csv'
# filename = 'subjectd-concentrating-2.csv'
# filename = 'subjecta-neutral-1' # in original_data
filename = 'concentrating_johnpaul1'


window_size = 256 #samples


sub_df = pd.read_csv(f"{dataset_dir}/{filename}.csv") 

# NOTE: Split the data after reading, THEN perform sliding window

# INFERENCE DATA
windows = []
cur_window = []
sample_count = 0

row=0
while row < sub_df.shape[0]:
    sample_count+=1
    cur_window.append(sub_df.iloc[row])

    if sample_count == window_size:
        windows.append(cur_window)
        cur_window = []
        sample_count = 0
        row-=window_size//2

    row+=1


windows = np.asarray(windows, dtype='object')
windows = windows[:, :, 1:5]

# print(windows[0].shape)

X_scalar = None

# To load the data back
with open('models/X_scalar.pckl', 'rb') as file:
    X_scalar = pickle.load(file)

windows = transform_set(X_scalar, np.array(windows), fit=False)


model = keras.models.load_model('models/jp.keras', custom_objects=None, compile=True, safe_mode=True)
preds = model.predict(windows)

with open(f'results/{filename}.json', 'w') as f:
    obj = {'class': (preds*100).reshape((preds.shape[0],)).tolist()}
    # print(obj)
    json.dump(obj, f)

score = 0

for i in range(preds.shape[0]):
    is_equal = (np.round(preds[i]) == 0)
    if is_equal:

        # plotit(windows[i])
        score+=1

print(preds)
print(f"{score}/{preds.shape[0]}")