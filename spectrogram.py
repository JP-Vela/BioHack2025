# Importing libraries using import keyword.
import math
import numpy as np
import matplotlib.pyplot as plt
 
# Set the time difference to take picture of
# the the generated signal.
Time_difference = 0.0001

# Example EEG data (excluding timestamps)
data = np.array([
    [0.57012211, 0.52551873, 0.51916968, 0.63666119],
    [0.55691002, 0.50988991, 0.51282063, 0.63666119],
    [0.57012211, 0.49328436, 0.51013447, 0.63829859],
    [0.53861748, 0.46715499, 0.5159949 , 0.63175236],
    [0.55691002, 0.49474972, 0.52161177, 0.63829859],
    [0.56300823, 0.51990236, 0.52674002, 0.64811792]
])

 
# Generating an array of values
Time_Array = np.linspace(0, 5, math.ceil(5 / Time_difference))
 
# Actual data array which needs to be plot
Data = 20*(np.sin(3 * np.pi * Time_Array))
 
# Matplotlib.pyplot.specgram() function to
# generate spectrogram
plt.specgram(data[:,1], Fs=6, cmap="rainbow")
 
# Set the title of the plot, xlabel and ylabel
# and display using show() function
plt.title('Spectrogram Using matplotlib.pyplot.specgram() Method')
plt.xlabel("DATA")
plt.ylabel("TIME")
plt.savefig('spec.png')