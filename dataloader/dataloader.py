import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pdb

class Dataloader:
    def __init__(self, directory, bboxes_dir) -> None:
        self.dataset_dir = directory
        self.image_labels = os.listdir(directory)
        self.bboxes = self.load_bboxes(bboxes_dir, self.image_labels)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        # Load the image
        label = self.image_labels[idx]
        img_path = os.path.join(self.dataset_dir, label)
        image = cv2.imread(img_path)
        image = image[...,::-1] # Convert BGR to RGB
        
        # Load the bounding box
        bboxes = self.bboxes[label]
        return image, bboxes

    def load_bboxes(self, bboxes_dir, image_labels):
        df = pd.read_csv(bboxes_dir)
        df = df.values.tolist()
        bboxes = {label: list() for label in image_labels}
        for box in df:
            bboxes[box[0]].append(box[1:])
        return bboxes
    
    def plot_image_and_bboxes(self, idx):
        image, bboxes = self.__getitem__(idx)
        image = image.astype(np.int16)
        for bbox in bboxes:
            # print(bbox)
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1]),int(bbox[2])-int(bbox[0]),int(bbox[3])-int(bbox[1])), (255,0,0), 2)
        plt.imshow(image)
        plt.show()