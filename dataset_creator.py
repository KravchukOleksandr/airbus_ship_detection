import os
import pandas as pd
import numpy as np
import cv2
import shutil


# Create catalog system for storing datasets
def create_catalog_system(path_to_dataset):
    os.mkdir(os.path.join(path_to_dataset, 'train_img'))
    os.mkdir(os.path.join(path_to_dataset, 'test_img'))
    os.mkdir(os.path.join(path_to_dataset, 'train_mask'))
    os.mkdir(os.path.join(path_to_dataset, 'test_mask'))
    os.mkdir(os.path.join(path_to_dataset, 'train_img/img'))
    os.mkdir(os.path.join(path_to_dataset, 'test_img/img'))
    os.mkdir(os.path.join(path_to_dataset, 'train_mask/img'))
    os.mkdir(os.path.join(path_to_dataset, 'test_mask/img'))


# Creating a subdataset
def copy_img_by_list(source_folder, destination_folder, img_names):
    for img in img_names:
        source_path = os.path.join(source_folder, img)
        destination_path = os.path.join(destination_folder, img)
        shutil.copy(source_path, destination_path)


# Extracting masks from csv
def mask_csv_to_image(img_names, mask_folder_path, df, img_shape):
    for img in img_names:
        bin_mask = np.zeros(img_shape[0] * img_shape[1])
        filtered_df = df[df['ImageId'] == img][['EncodedPixels']]
        filtered_df.fillna('0 0', inplace=True)
        ships_pix = filtered_df.values.flatten()
        ships_pix = np.array(' '.join(ships_pix).split(' ')).astype(int)
        for j in range(0, len(ships_pix), 2):
            pix, pix_num = ships_pix[j], ships_pix[j + 1]
            bin_mask[pix: pix + pix_num] = 255
        bin_mask = bin_mask.reshape(img_shape[0], img_shape[1]).T
        cv2.imwrite(mask_folder_path + img, bin_mask)


def get_img_names(path_to_original, sz_train, sz_test):
    csv_file_path = path_to_original + 'train_ship_segmentations_v2.csv'
    df = pd.read_csv(csv_file_path)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    img_names = df['ImageId'].unique()
    img_names_train = img_names[:sz_train]
    img_names_test = img_names[sz_train: sz_train + sz_test]
    return img_names_train, img_names_test, df


def my_interface():
    # jupyt_not_vs/data/ship_dataset
    # jupyt_not_vs/data/airbus-ship-detection
    print('Enter a path to the direcoty, where dataset will be located') 
    path_to_dataset = input() + '/'
    print('Enter a path to the original airbus-ship-detection directory') 
    path_to_original = input() + '/'
    print('Enter train sizes')
    sz_train = int(input())
    print('Enter test sizes')
    sz_test = int(input())
    return path_to_dataset, path_to_original, sz_train, sz_test


def main():
    path_to_dataset, path_to_original, sz_train, sz_test = my_interface()
    create_catalog_system(path_to_dataset)

    img_names_train, img_names_test, df = get_img_names(path_to_original, sz_train, sz_test)
    source_folder_train = path_to_original + 'train_v2/'
    destination_folder_train = path_to_dataset + 'train_img/img/'
    source_folder_test = path_to_original + 'train_v2/'
    destination_folder_test = path_to_dataset + 'test_img/img/'
    copy_img_by_list(source_folder_train, destination_folder_train, img_names_train)
    copy_img_by_list(source_folder_test, destination_folder_test, img_names_test)

    img_shape = (768, 768, 3)
    mask_folder_path_train = path_to_dataset + 'train_mask/img/'
    mask_folder_path_test = path_to_dataset + 'test_mask/img/'
    mask_csv_to_image(img_names_train, mask_folder_path_train, df, img_shape)
    mask_csv_to_image(img_names_test, mask_folder_path_test, df, img_shape)
    print('Process is finished')


if __name__ == "__main__":
    main()
