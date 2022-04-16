from dataloader.dataloader import Dataloader
from model.model import create_model
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pdb

# Variables
train_dataset_path = "/home/yemika/Mikael/Code/datasets/car_object_detection/training_images"
bboxes_dir = "/home/yemika/Mikael/Code/datasets/car_object_detection/train_solution_bounding_boxes.csv"

# Load the data
train_data = Dataloader(train_dataset_path, bboxes_dir)

# Create the model
model = create_model()
pdb.set_trace()