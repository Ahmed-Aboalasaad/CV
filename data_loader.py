import os
import cv2
import shutil
from zipfile import ZipFile

class DataLoader:
    def __init__(self, dataset_path='LungTumorDetectionAndSegmentation.zip', loading_destination='dataset/'):
        self.dataset_path = dataset_path
        self.loading_destination = loading_destination        
        self.train_path = os.path.join(loading_destination, 'train')
        self.val_path = os.path.join(loading_destination, 'val')

    def load_data(self):
        '''Reads the training & validation data from the train_path & val_path.
        Returns 2 dictionaries.
        Each one has string keys as the scanID (ex: "Subject0_0).
        And each value is a list of [Image, Mask, detections_list]
        where detections list may include 0 or more lists
        of 4 numbers eachlocating the bounding box'''
        # Extract the original dataset directory
        if os.path.exists(self.loading_destination):
            shutil.rmtree(self.loading_destination)
        os.makedirs(self.loading_destination)
        with ZipFile(self.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.loading_destination)
        
        self.image_height, self.image_width = cv2.imread("dataset/train/images/Subject_0/0.png", cv2.IMREAD_UNCHANGED).shape
        print(f"The dataset consists of {self.image_height} x {self.image_width} grayscale images")

        print('[Loading Training Data  ]  ', end='')
        self.__organize(self.train_path)
        self.__report_loading(self.train_path)
        self.__delete_old_directories(self.train_path)

        print('[Loading Validation Data]  ', end='')
        self.__organize(self.val_path)
        self.__report_loading(self.val_path)
        self.__delete_old_directories(self.val_path)

    def __organize(self, data_path):
        '''Organizes the data in data_path'''
        all_images_path = f'{data_path}/all_images/'
        all_masks_path = f'{data_path}/all_masks/'
        images_and_detections_path = f'{data_path}/images_and_detections/'
        for path in [all_images_path, all_masks_path, images_and_detections_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        for patient in os.listdir(os.path.join(data_path, 'images/')):
            for image_name in os.listdir(os.path.join(data_path, 'images/', patient)):
                image_path = os.path.join(data_path, 'images/', patient, image_name)
                scan_ID, extension = os.path.splitext(image_name)
                scan_ID = patient + '_' + scan_ID

                image_copy_path = os.path.join(all_images_path, f'{scan_ID}.png')
                shutil.copy(image_path, image_copy_path)
                image_copy_path = os.path.join(images_and_detections_path, f'{scan_ID}.png')
                shutil.copy(image_path, image_copy_path)

            for mask_name in os.listdir(os.path.join(data_path, 'masks/', patient)):
                mask_path = os.path.join(data_path, 'masks/', patient, mask_name)
                scan_ID, extension = os.path.splitext(mask_name)
                scan_ID = patient + '_' + scan_ID

                mask_copy_path = os.path.join(all_masks_path, f'{scan_ID}.png')
                shutil.copy(mask_path, mask_copy_path)

            for det_name in os.listdir(os.path.join(data_path, 'detections/', patient)):
                detection_path = os.path.join(data_path, 'detections/', patient, det_name)
                self.__normalize_detections(detection_path)

                scan_ID, extension = os.path.splitext(det_name)
                scan_ID = patient + '_' + scan_ID
                detection_copy_path = os.path.join(images_and_detections_path, f'{scan_ID}.txt')
                shutil.copy(detection_path, detection_copy_path) 

    def __normalize_detections(self, path):
        '''Normalizes detections written in the file placed in path'''
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
                coordinates = f'0 {bbox_x_center} {bbox_y_center} {bbox_width} {bbox_height}'
                detections.append(coordinates)
            with open(path, 'w') as file:
                file.write('\n'.join(detections))

    def __report_loading(self, data_path):
        '''Counts the number of scans that do/doesn't have
        cancer detections data_path"'''
        file_names = os.listdir(f'{data_path}/images_and_detections')
        image_names = [os.path.splitext(name)[0] for name in file_names if name.endswith('png')]
        detections_names = [os.path.splitext(name)[0] for name in file_names if name.endswith('txt')]
        healthy_num = len(image_names) - len(detections_names)
        cancer_num = len(image_names) - healthy_num
        print(f'Loaded [{len(image_names)}] Scans ({cancer_num} Cancer + {healthy_num} Healthy)')
        
    def __delete_old_directories(self, data_path):
        '''Removes the original dataset folders that are not needed anymore'''
        shutil.rmtree(os.path.join(data_path, 'images'))
        shutil.rmtree(os.path.join(data_path, 'masks'))
        shutil.rmtree(os.path.join(data_path, 'detections'))
        os.rename(os.path.join(data_path, 'all_images/'), os.path.join(data_path, 'images'))
        os.rename(os.path.join(data_path, 'all_masks/'), os.path.join(data_path, 'masks'))
