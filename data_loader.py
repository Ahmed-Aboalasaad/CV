import os
import cv2
import shutil
import random
import numpy as np
from zipfile import ZipFile

class DataLoader:
    def __init__(self, dataset_path='dataset.zip', loading_destination='dataset/'):
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
        if os.path.normpath(self.loading_destination) != '.':
            if os.path.exists(self.loading_destination):
                shutil.rmtree(self.loading_destination)
            os.makedirs(self.loading_destination)
        with ZipFile(self.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.loading_destination)
        
        self.image_height, self.image_width = cv2.imread(f"{self.train_path}/images/Subject_0/0.png", cv2.IMREAD_UNCHANGED).shape
        print(f"The dataset consists of {self.image_height} x {self.image_width} grayscale images")

        def report_loading(data_path):
            '''Counts the number of scans that do/doesn't have
            cancer detections data_path"'''
            file_names = os.listdir(f'{data_path}/images_and_detections')
            image_names = [os.path.splitext(name)[0] for name in file_names if name.endswith('png')]
            detections_names = [os.path.splitext(name)[0] for name in file_names if name.endswith('txt')]
            healthy_num = len(image_names) - len(detections_names)
            cancer_num = len(image_names) - healthy_num
            print(f'Loaded [{len(image_names)}] Scans ({cancer_num} Cancer + {healthy_num} Healthy)')

        def delete_old_directories(data_path):
            '''Removes the original dataset folders that are not needed anymore'''
            shutil.rmtree(os.path.join(data_path, 'images'))
            shutil.rmtree(os.path.join(data_path, 'masks'))
            shutil.rmtree(os.path.join(data_path, 'detections'))
            os.rename(os.path.join(data_path, 'all_images/'), os.path.join(data_path, 'images'))
            os.rename(os.path.join(data_path, 'all_masks/'), os.path.join(data_path, 'masks'))
            os.rename(os.path.join(data_path, 'all_detections/'), os.path.join(data_path, 'detections'))

        print('[Loading Training Data  ]  ', end='')
        self.__organize(self.train_path)
        report_loading(self.train_path)
        delete_old_directories(self.train_path)

        print('[Loading Validation Data]  ', end='')
        self.__organize(self.val_path)
        report_loading(self.val_path)
        delete_old_directories(self.val_path)

    def __organize(self, data_path):
        '''Organizes the data in data_path (train / val) into 3 folders:
        all_images/ all_masks/ images_and_detections/
        Placing all of them under data_path'''
        all_images_path = f'{data_path}/all_images/'
        all_masks_path = f'{data_path}/all_masks/'
        all_detections_path = f'{data_path}/all_detections/'
        images_and_detections_path = f'{data_path}/images_and_detections/'
        for path in [all_images_path, all_masks_path, all_detections_path, images_and_detections_path]:
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
                detection_copy_path = os.path.join(all_detections_path, f'{scan_ID}.txt')
                shutil.copy(detection_path, detection_copy_path)

    def __normalize_detections(self, path):
        '''Normalizes detections written in the file placed in path for yolo
        to be on this format: [classID, bbox_x_center, bbox_y_center, bbox_width, bbox_height]'''
        detections = []
        if not os.path.exists(path):
            print(f'Error: Reading a detection from a non-existent path')
            return ()
        with open(path, 'r') as file:
            for line in file:
                xmin, ymin, xmax, ymax = [int(num.strip()) for num in line.split(',')]
                bbox_x_center = (xmin + xmax) / 2 / self.image_width
                bbox_y_center = (ymin + ymax) / 2 / self.image_height
                bbox_width = (xmax - xmin) / self.image_width
                bbox_height = (ymax - ymin) / self.image_height
                coordinates = f'0 {bbox_x_center} {bbox_y_center} {bbox_width} {bbox_height}'
                detections.append(coordinates)
            with open(path, 'w') as file:
                file.write('\n'.join(detections))

    def random_scans(self, num_scans):
        '''Retrurns a list of length num_scans of tuples
        of images and their corresponding detections.
        A detection will be returned as [xmin, ymin, xmax, ymax]'''
        images_path = os.path.join(self.train_path, 'images/')
        image_names = random.sample(os.listdir(images_path), num_scans)
        random_images = []
        for name in image_names:
            img = cv2.imread(os.path.join(images_path, name), cv2.IMREAD_GRAYSCALE)
            random_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        detectinos_path = os.path.join(self.train_path, 'images_and_detections/')
        random_detections = []
        for name in image_names:
            scanID = os.path.splitext(name)[0]
            det_path = os.path.join(detectinos_path, f'{scanID}.txt')
            random_detections.append(self.__read_detections(det_path))
        
        return [(img, dets) for img, dets in zip(random_images, random_detections)]

    def __read_detections(self, dets_path, yolo_normalization=False):
        '''Reads the detections of one image whose detections file is placed dets_path.
        This text file should include 5 whitespace-separated numbers normalized for yolo:
        [class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height]
        To convert them back to [xmin, ymin, xmax, ymax], raise the unnormalize flag'''
        if not os.path.exists(dets_path):
            return []
        detections = []
        file = open(dets_path, 'r')
        for line in file:
            det = [float(num.strip()) for num in line.split(' ')]
            if yolo_normalization:
                detections.append(det[1:])
                continue
            class_id, bbox_x_center, bbox_y_center, bbox_width, bbox_height = det
            xmin = int(bbox_x_center * self.image_width - bbox_width * self.image_width/2)
            ymin = int(bbox_y_center * self.image_height - bbox_height * self.image_height/2) 
            xmax = int(bbox_x_center * self.image_width + bbox_width * self.image_width/2)
            ymax = int(bbox_y_center * self.image_height + bbox_height * self.image_height/2)
            detections.append([xmin, ymin, xmax, ymax])
        file.close()
        return detections

    def generate_cropped_data(self, destination=''):
        self.__generate_cropped_data(self.train_path)
        self.__generate_cropped_data(self.val_path)

    def __generate_cropped_data(self, data_path):
        '''Crops the images & masks from images/ & masks/ located under dat_path,
        which also has images_and_detections/ that is used to crop.'''
        cropped_images_path = os.path.join(data_path, 'cropped_images/')
        cropped_masks_path = os.path.join(data_path, 'cropped_masks/')
        for path in [cropped_images_path, cropped_masks_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        
        for image_name in os.listdir(os.path.join(data_path, 'images/')):
            scanID, extension = os.path.splitext(image_name)
            det_path = os.path.join(data_path, 'images_and_detections/', f'{scanID}.txt')
            if not os.path.exists(det_path):
                continue
            
            image_path = os.path.join(data_path, 'images/', image_name)
            mask_path = os.path.join(data_path, 'masks/', image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            detections = self.__read_detections(det_path)
            for i, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det
                cropped_image = image[ymin:ymax, xmin:xmax]
                cropped_mask = mask[ymin:ymax, xmin:xmax]
                
                cropped_image = cv2.resize(cropped_image, (self.image_width, self.image_height))
                cropped_mask = cv2.resize(cropped_mask, (self.image_width, self.image_height))
                
                cropped_img_dst = os.path.join(cropped_images_path, f"{scanID}_tumor{i+1}.png")
                cropped_mask_dst = os.path.join(cropped_masks_path, f"{scanID}_tumor{i+1}.png")
                cv2.imwrite(cropped_img_dst, cropped_image)
                cv2.imwrite(cropped_mask_dst, cropped_mask)

    def rebuild_masks(self, cropped_masks_path, detections_path, destination):
        def rebuild_mask(cropped_mask_path, detection, canvas):
            '''Reconstrcuts the original mask from the zoomed-in cropped mask located at "cropped_mask_path".
            Detection is a list of 4 number normalized for yolo : [box_xcenter, box_ycenter, box_width, box_height]
            Using "detection", it resizees this cropped mask to the original size
            and places it on the given canvas. If no canvas was given, a blank one is created by defualt.'''
            big_box = cv2.imread(cropped_mask_path, cv2.IMREAD_GRAYSCALE)

            # These 4 numbers are related to the small original box
            box_xcenter, box_ycenter, box_width, box_height = detection
            box_xcenter = int(box_xcenter * self.image_width)
            box_ycenter = int(box_ycenter * self.image_height)
            box_width = int(box_width * self.image_width)
            box_height = int(box_height * self.image_height)

            small_box = cv2.resize(big_box, (box_width, box_height))
            xmin = box_xcenter - box_width // 2
            ymin = box_ycenter - box_height // 2

            canvas[ymin:ymin+box_height, xmin:xmin+box_width] = small_box
            return canvas
        
        if not os.path.exists(cropped_masks_path):
            print(f'[Error] non-existent mask reconstruction destination')
        if os.path.exists(destination):
            shutil.rmtree(destination)
        os.makedirs(destination)
        
        for dets_name in os.listdir(detections_path):
            if not dets_name.endswith('.txt'):
                continue
            scanID, extension = dets_name.split('.')
            dets_path = os.path.join(detections_path, dets_name)
            dets = self.__read_detections(dets_path, yolo_normalization=True)
            canvas = np.zeros((self.image_height, self.image_width))
            for i, det in enumerate(dets):
                cropped_mask_name = f'{scanID}_tumor{i+1}.png'
                cropped_mask_path = os.path.join(cropped_masks_path, cropped_mask_name)
                rebuilt_mask = rebuild_mask(cropped_mask_path, det, canvas)
            cv2.imwrite(os.path.join(destination, f'{scanID}.png'), rebuilt_mask)
