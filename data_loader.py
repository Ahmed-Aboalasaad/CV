import os
import cv2

class DataLoader:
    def __init__(self, train_path='./dataset/train/', val_path='./dataset/val/'):
        '''Loads training and validation data supposing it's organized as follows:
        The train_path and val_path should have 3 folders (images, masks, detections)
        All of the 3 folders should have subfolders one for each patient.
        Images and masks should be corresponding to each other and named the same.
        Detections of each patient should include text files named with the scan number
        and each scan files includes a line or more of 4 numbers each (xMin, yMin, xMax, yMax)'''
        self.train_path = train_path
        self.val_path = val_path
        
        print('[ Loading Training Data ]  ', end='')
        self.train_data = self.load_data(self.train_path)
        print('Done')
        print('[ Loading Validation Data ]  ', end='')
        self.val_data = self.load_data(self.val_path)
        print('Done')
    
    def load_data(self, data_path):
        data = {}
        for patient in os.listdir(os.path.join(data_path, 'images/')):
            for image_name in os.listdir(os.path.join(data_path, 'images/', patient)):
                image_path = os.path.join(data_path, 'images/', patient, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                scan_ID, extension = os.path.splitext(image_name)
                scan_ID = patient + '_' + scan_ID
                data[scan_ID] = [img]

            for mask_name in os.listdir(os.path.join(data_path, 'masks/', patient)):
                mask_path = os.path.join(data_path, 'masks/', patient, mask_name)
                
                scan_ID, extension = os.path.splitext(mask_name)
                scan_ID = patient + '_' + scan_ID
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                data[scan_ID].append(mask)

            for detection_name in os.listdir(os.path.join(data_path, 'detections/', patient)):
                detection_path = os.path.join(data_path, 'detections/', patient, detection_name)
                detections = self.read_detections(detection_path)

                scan_ID, extension = os.path.splitext(detection_name)
                scan_ID = patient + '_' + scan_ID
                data[scan_ID].append(detections)
        
        for scan in data.values():
            if len(scan) == 2:
                scan.append(None)
        return data
    
    def read_detections(self, path):
        detections = []
        with open(path, 'r') as file:
            for line in file:
                coordinates = [int(num.strip()) for num in line.split(',')]
                detections.append(tuple(coordinates))
        return tuple(detections)
