import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout

df = pd.read_csv('/kaggle/input/visual-taxonomy/train.csv') #change directory as per your convenience
df.head()

df['Category'].unique()

# Define the categories
categories = ['Men Tshirts', 'Sarees', 'Kurtis', 'Women Tshirts', 'Women Tops & Tunics']

# Loop through each category, filter rows, and save to a new CSV file
for category in categories:
    filtered_df = df[df['Category'] == category]
    filename = category.replace(" ", "_").replace("&", "and").lower() + '.csv'  # Format filename
    filtered_df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(filtered_df)} rows.")
