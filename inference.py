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
filename = 'JonnyUnderstanding'


window_size = 256 #samples


sub_df = pd.read_csv(f"{dataset_dir}/{filename}.csv") 
cutoff = int(sub_df.shape[0]*0.85)
sub_df = sub_df.iloc[cutoff:]

# NOTE: Split the data after reading, THEN perform sliding window

# INFERENCE DATA
windows = []
cur_window = []
sample_count = 0

row=0
while row+window_size < sub_df.shape[0]:
    windows.append(sub_df.iloc[row:row+window_size])
    row+=int(window_size*0.1)



windows = np.asarray(windows, dtype='object')
windows = windows[:, :, 1:5]

# print(windows[0].shape)

X_scalar = None

# To load the data back
with open('models/X_scalar.pckl', 'rb') as file:
    X_scalar = pickle.load(file)

windows = transform_set(X_scalar, np.array(windows), fit=False)


model = keras.models.load_model("models/3_output_model_jonny_2.keras", custom_objects=None, compile=True, safe_mode=True)
preds = model.predict(windows)

'''with open(f'results/{filename}.json', 'w') as f:
    obj = {'class': (preds*100).reshape((preds.shape[0],)).tolist()}
    # print(obj)
    json.dump(obj, f)
'''
score = 0

# Confused, Control, Understanding
for i in range(0, preds.shape[0], 3):

    is_equal = (np.round(preds[i+1]) == np.array([0,0,1]))
    
    if is_equal.all():
        score+=1
        # print(preds[i])
        # plotit(X_test_scaled[i])


for p in np.round(preds):
    print(p)

print(f"{score}/{preds.shape[0]/3} = {score/(preds.shape[0]/3)*100}%")