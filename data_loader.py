import os
import cv2
import shutil

class DataLoader:
    def __init__(self, train_path='dataset/train/', val_path='dataset/val/'):
        self.train_path = train_path
        self.val_path = val_path

    def load_data(self):
        '''Reads the training & validation data from the train_path & val_path.
        Returns 2 dictionaries.
        Each one has string keys as the scanID (ex: "Subject0_0).
        And each value is a list of [Image, Mask, detections_list]
        where detections list may include 0 or more lists
        of 4 numbers eachlocating the bounding box'''
        self.image_height, self.image_width = cv2.imread("dataset/train/images/Subject_0/0.png", cv2.IMREAD_UNCHANGED).shape
        print(f"The dataset consists of {self.image_height} x {self.image_width} grayscale images")
        
        print('[ Loading Training Data   ]  ', end='')
        train_data = self.__load(self.train_path)
        train_cancer_num, train_healthy_num = self.__count_cancer(train_data)
        print(f'Loaded [{len(train_data)}] Scans ({train_cancer_num} Cancer + {train_healthy_num} Healthy)')
        
        print('[ Loading Validation Data ]  ', end='')
        val_data = self.__load(self.val_path)
        val_cancer_num,val_healthy_num = self.__count_cancer(val_data)
        print(f'Loaded [{len(val_data)}] Scans ({val_cancer_num} Cancer + {val_healthy_num} Healthy)')
        return train_data, val_data

    def __load(self, data_path):
        '''Loads the training / validation data using the given path'''
        
        images_and_detections_path = f'{data_path}/images_and_detections/'
        images_and_masks_path = f'{data_path}/images_and_masks/'
        if os.path.exists(images_and_detections_path):
            shutil.rmtree(images_and_detections_path)
        if os.path.exists(images_and_masks_path):
            shutil.rmtree(images_and_masks_path)
        os.makedirs(images_and_detections_path)
        os.makedirs(images_and_masks_path)

        data = {}
        for patient in os.listdir(os.path.join(data_path, 'images/')):
            for image_name in os.listdir(os.path.join(data_path, 'images/', patient)):
                image_path = os.path.join(data_path, 'images/', patient, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                scan_ID, extension = os.path.splitext(image_name)
                scan_ID = patient + '_' + scan_ID
                data[scan_ID] = [img]

                copied_image_path = os.path.join(f'{data_path}/images_and_detections/', f'{scan_ID}.png')
                shutil.copy(image_path, copied_image_path)
                copied_image_path = os.path.join(f'{data_path}/images_and_masks/', f'image_{scan_ID}.png')
                shutil.copy(image_path, copied_image_path)                

            for mask_name in os.listdir(os.path.join(data_path, 'masks/', patient)):
                mask_path = os.path.join(data_path, 'masks/', patient, mask_name)
                scan_ID, extension = os.path.splitext(mask_name)
                scan_ID = patient + '_' + scan_ID
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                data[scan_ID].append(mask)

                copied_mask_path = os.path.join(f'{data_path}/images_and_masks/', f'mask_{scan_ID}.png')
                shutil.copy(mask_path, copied_mask_path)    

            for detection_name in os.listdir(os.path.join(data_path, 'detections/', patient)):
                detection_path = os.path.join(data_path, 'detections/', patient, detection_name)
                detections = self.__read_detections(detection_path)

                scan_ID, extension = os.path.splitext(detection_name)
                scan_ID = patient + '_' + scan_ID
                data[scan_ID].append(detections)

                detections = [f'1,{det[0]},{det[1]},{det[2]},{det[3]}' for det in detections]
                copied_detection_path = os.path.join(f'{data_path}/images_and_detections/', f'{scan_ID}.txt')
                with open(copied_detection_path, 'w') as file:
                    file.write('\n'.join(detections))

        for scan in data.values():
            if len(scan) == 2:
                scan.append([])
        return data

    def __read_detections(self, path):
        '''Reads cancer detections from a text file in path'''
        detections = []
        if not os.path.exists(path):
            print(f'Error: Reading a detection from a wrong path')
            return ()
        with open(path, 'r') as file:
            for line in file:
                xmin, ymin, xmax, ymax = [int(num.strip()) for num in line.split(',')]
                bbox_x_center = (xmin + xmax) / 2 / self.image_width
                bbox_y_center = (ymin + ymax) / 2 / self.image_height
                bbox_width = (xmax - xmin) / self.image_width
                bbox_height = (ymax - ymin) / self.image_height
                coordinates = [1, bbox_x_center, bbox_y_center, bbox_width, bbox_height]
                detections.append(tuple(coordinates))
        return tuple(detections)

    def __count_cancer(self, data):
        '''Counts the number of scans that do/doesn't have
        cancer detections in the dictionary "data"'''
        cancer_sum = 0
        healthy_sum = 0
        for image, scan, detections in data.values():
            if len(detections) == 0:
                healthy_sum += 1
            else:
                cancer_sum += 1
        return cancer_sum, healthy_sum
