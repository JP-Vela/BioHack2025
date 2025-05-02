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
filename = 'JonnyControl2'
filename2 = 'JonnyControl'


window_size = 256 #samples


sub_df = pd.read_csv(f"{dataset_dir}/{filename}.csv") 
cutoff = int(sub_df.shape[0]*0.85)
sub_df = sub_df.iloc[cutoff:]

sub_df2 = pd.read_csv(f"{dataset_dir}/{filename2}.csv") 
cutoff = int(sub_df2.shape[0]*0.85)
sub_df2 = sub_df2.iloc[cutoff:]

pd.concat((sub_df, sub_df2))

# NOTE: Split the data after reading, THEN perform sliding window

# INFERENCE DATA
windows = []
cur_window = []
sample_count = 0

row=0
while row+window_size < sub_df.shape[0]:
    windows.append(sub_df.iloc[row:row+window_size])
    row+=int(window_size*1)



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
num_samples = 0
for i in range(0, preds.shape[0]):

    is_equal = (np.round(preds[i]) == np.array([1,0,0]))
    num_samples+=1
    if is_equal.all():
        score+=1


print(f"{score}/{num_samples} = {score/num_samples*100}%")


score = 0
num_samples = 0
# Confused, Control, Understanding

for i in range(3, preds.shape[0], 3):

    added = np.round(  (preds[i]+preds[i-1]+preds[i-2])/3 )
    # print(added)
    is_equal = (np.round(added) == np.array([0,1,0]))
    num_samples+=1
    if is_equal.all():
        score+=1

print(f"{score}/{num_samples} = {score/num_samples*100}%")