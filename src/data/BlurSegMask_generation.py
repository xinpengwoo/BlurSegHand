import cv2
import os
import numpy as np
import os.path as osp
import torch
import concurrent.futures

from tqdm import tqdm
from pycocotools.coco import COCO
from utils.options import parse_options, copy_opt_file


def main():    
    # load opt and args from yaml
    opt, args = parse_options()
    
    # dynamic dataset import
    opt_data = opt['dataset']
    dataset_name = opt_data['name']
    exec(f'from data.{dataset_name} import {dataset_name}')
    globals().update(locals())
    # train and test dataset
    tester = Tester(opt, opt_data, "train")
    data = tester.datalist

    process_with_tqdm(28, data)
    # for image_paths in tqdm(data):
    #     process_each_dict(image_paths)

def process_with_tqdm(max_threads, data_dicts):
    with tqdm(total=len(data_dicts), position=0, leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            # Function to process a single dictionary with progress update
            def process_single(data_dict):
                process_dict(data_dict)
                pbar.update(1)

            # Submit the dictionaries for processing concurrently
            futures = [executor.submit(process_single, data_dict) for data_dict in data_dicts]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

def process_dict(image_paths):
    blurry_image_path = image_paths.pop('blurry_image_path')
    sharp_images_path = [value for key, value in image_paths.items()]
    # get variance path
    blurry_image_path_parts = blurry_image_path.split(".")
    blurry_image_path_parts[-2] += '-var'
    variance_path = ".".join(blurry_image_path_parts)
    cacuclate_variance(sharp_images_path, variance_path)
    #print(variance_path + " is successful")

def cacuclate_variance(data,var_path):
    # Read the first image to get the image size
    image_path = data[0]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, _ = image.shape

    # Create an empty array to store pixel values for all images
    pixel_values = np.zeros((height, width, 3, len(data)-1), dtype=np.float32)

    # Read and process each image
    for i in range(len(data)-1):
        image_path = data[i]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        pixel_values[:, :, :, i] = image.astype(np.float32) #/ 255.0  # Normalize pixel values to [0, 1]
        
    variance = np.var(pixel_values, axis=3)

    # ostu's thresholding
    gray_image = cv2.cvtColor(variance, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # mathematical morphology - closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(var_path, np.uint8(closing))


class Tester(torch.utils.data.Dataset):
    def __init__(self, opt, opt_data, data_split):
        self.opt = opt
        self.opt_params = opt['task_parameters']
        self.data_split = 'train' if data_split == 'train' else 'test'

        # path for images and annotations
        self.img_path = opt_data['img_path']
        self.annot_path = opt_data['annot_path']

        # path for sharp images path
        self.sharp_img_path = "./dataset/InterHand3.6M/InterHand2.6M_30fps_batch1/images"

        # data loader and construct batch generator
        opt_data = self.opt['dataset']
        
        dataset_name = opt_data['name']
        self.datalist = self.load_data()

    def load_data(self):
        db = COCO(osp.join(self.annot_path, self.data_split, f'BlurHand_{self.data_split}_data.json'))
        num = 0
        datalist = []
        for aid in [*db.anns.keys()]:
            ann = db.anns[aid]
            if not ann['is_middle']:
                continue
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_path, self.data_split, img['file_name'])
            # repeat the process for past(1st) image
            data = {}
            data['blurry_image_path'] = img_path
            sharp_aid_list = ann['aid_list'][:]
            for i in range(len(sharp_aid_list)):
                aid = sharp_aid_list[i]
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                # sharp_img_path, png -> jpg
                file_name_parts = img['file_name'].split(".")
                file_name_parts[-1] = "jpg"
                img_path = osp.join(self.sharp_img_path, self.data_split, ".".join(file_name_parts))
                data[f'{i + 1}_sharp_image_path'] = img_path
            datalist.append(data)
        print("Total number of sample in BlurHand: {}".format(len(datalist)))
        return datalist
    
    
if __name__ == "__main__":
    main()
