import os
import cv2
import shutil
import random
import numpy as np
from zipfile import ZipFile

class DataWizard:
    def __init__(self, dataset_path='dataset.zip', dataset_loading_destination='datasets/'):
        self.dataset_path = dataset_path
        self.loading_destination = dataset_loading_destination        
        self.train_path = os.path.join(dataset_loading_destination, 'train')
        self.val_path = os.path.join(dataset_loading_destination, 'val')
        
        # Extract the original dataset directory
        if os.path.normpath(self.loading_destination) != '.':
            if os.path.exists(self.loading_destination):
                shutil.rmtree(self.loading_destination)
            os.makedirs(self.loading_destination)
        with ZipFile(self.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.loading_destination)
        first_image_path = f"{self.train_path}/images/Subject_0/0.png"
        self.image_height, self.image_width = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED).shape
        print(f"The dataset consists of {self.image_height} x {self.image_width} grayscale images\n")

    def generate_dataset_files(self):
        '''Reads the training & validation data from the train_path & val_path.
        Returns 2 dictionaries.
        Each one has string keys as the scanID (ex: "Subject0_0).
        And each value is a list of [Image, Mask, detections_list]
        where detections list may include 0 or more lists
        of 4 numbers eachlocating the bounding box'''
        def delete_old_directories(data_path):
            '''Removes the original dataset folders that are not needed anymore'''
            shutil.rmtree(os.path.join(data_path, 'images'))
            shutil.rmtree(os.path.join(data_path, 'masks'))
            shutil.rmtree(os.path.join(data_path, 'detections'))
            os.rename(os.path.join(data_path, 'all_images/'), os.path.join(data_path, 'images'))
            os.rename(os.path.join(data_path, 'all_masks/'), os.path.join(data_path, 'masks'))
            os.rename(os.path.join(data_path, 'all_detections/'), os.path.join(data_path, 'detections'))

        def report_loading(data_path):
            '''Counts the number of scans that do/doesn't have cancer detections data_path"'''
            images_count = len(os.listdir(f'{data_path}/images'))
            detections_count = len(os.listdir(f'{data_path}/detections'))
            healthy_count = images_count - detections_count
            print(f'Loaded [{images_count}] Scans ({detections_count} Cancer + {healthy_count} Healthy)')
            
        print('[Ordering Training Data  ]  ', end='')
        self.__organize(self.train_path)
        delete_old_directories(self.train_path)
        report_loading(self.train_path)

        print('[Ordering Validation Data]  ', end='')
        self.__organize(self.val_path)
        delete_old_directories(self.val_path)
        report_loading(self.val_path)

        self.crop_dataset(self.train_path)
        self.crop_dataset(self.val_path)
        print('\n[INFO]\nReorganized File Structure to:\n-images/\n-masks/\n-detections/\n' +
              '-images_and_detections/\n-cropped_images/\n-cropped_masks/')

    def load_dataset(self, data_path):
        '''Reads the images and masks in the given data_path (train/val)
        into 2 lists of normalized numpy arrays'''
        def load_images(src):
            images = []
            for image_name in os.listdir(src):
                image = cv2.imread(os.path.join(image_name), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.image_width, self.image_height))
                # Generally, normalized images perform well with pretrained models
                images.append(image / 255.0)
            return np.array(images)

        print(f'[Loading {os.path.basename(data_path)} Dataset]', end='')
        print(' Done')
        return [load_images(os.path.join(data_path, 'images')),
                load_images(os.path.join(data_path, 'masks'))]

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

    def __organize(self, data_path):
        '''Organizes the data in data_path (train / val) into 4 folders:
        all_images/ all_masks/ all_detections/ images_and_detections/
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

    def crop_image(self, image_path, saving_path, detections_path):
        image_name = os.path.basename(image_path)
        scanID, extension = os.path.splitext(image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        dets = self.__read_detections(detections_path, yolo_normalization=False)
        for i, det in enumerate(dets):
            xmin, ymin, xmax, ymax = det
            cropped_image = image[ymin:ymax, xmin:xmax]            
            cropped_image = cv2.resize(cropped_image, (self.image_width, self.image_height))
            cropped_img_path = os.path.join(saving_path, f"{scanID}_tumor{i+1}.png")
            cv2.imwrite(cropped_img_path, cropped_image)

    def crop_dataset(self, destination=''):
        def crop(source_paths, dest_paths):
            '''Crops the images & masks from images/ & masks/ located under dat_path (train/val),
            which also has images_and_detections/ that is used to crop.'''
            src_images_path, src_masks_path, src_detections_path = source_paths
            dst_images_path, dst_masks_path = dest_paths
            for path in dest_paths:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)

            for dets_name in os.listdir(src_detections_path):
                dets_path = os.path.join(src_detections_path, dets_name)
                scanID, extension = os.path.splitext(dets_name)
                image_name = f'{scanID}.png'
                image_path = os.path.join(src_images_path, image_name)
                mask_path = os.path.join(src_masks_path, image_name)
                self.crop_image(image_path, dst_images_path, dets_path)
                self.crop_image(mask_path, dst_masks_path, dets_path)
        
        dataset_dirs = os.listdir(self.train_path)
        if not all(x in dataset_dirs for x in ['images', 'masks', 'detections', 'images_and_detections']):
            print('The dataset is not loaded yet.\nPlease call .load_data() first')
            return
       
        for data_path in [self.train_path, self.val_path]:
            src_paths = [os.path.join(data_path, 'images/'),
                         os.path.join(data_path, 'masks/'),
                         os.path.join(data_path, 'detections/')]
            
            # Default Destination is in train/ and val/ next to sources
            if destination == '':
                dst_paths = [os.path.join(data_path, 'cropped_images'),
                              os.path.join(data_path, 'cropped_masks/')]
            else:
                dst_paths = [os.path.join(destination, data_path, 'cropped_images'),
                             os.path.join(destination, data_path, 'cropped_masks')]
            crop(src_paths, dst_paths)

    def rebuild_masks(self, cropped_masks_path, detections_path, destination):
        '''Recreates full masks using cropped ones in "cropped_masks_path" & detections in "detections_path"
        The results are saveed to "destination"'''
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
