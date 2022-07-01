import os
import pandas as pd
import random
import cv2
from matplotlib import pyplot as plt

# Variables
description_file = "train_solution_bounding_boxes.csv"

# Load data
data_description = pd.read_csv(description_file)
valid_list = data_description['image'].values
full_list = os.listdir("./train")
invalid_list = list(set(full_list).difference(valid_list))
valid_list = list(set(valid_list))
n_val = int(len(valid_list) * 0.2)

random.shuffle(valid_list)

val_data = valid_list[:n_val]
train_data = valid_list[n_val:]

# # Delete val_data from train
# for val_file in val_data:
#     os.remove(os.path.join("train", val_file))

# # Delete train_data from val
# for train_file in train_data:
#     os.remove(os.path.join("valid", train_file))