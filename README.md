# airbus_ship_detection
Detecting ships in satellite's photos 

## Model performance

In OPEN_ME.ipynb you can see the Dice Score for 512 images and several predicted binary masks in diffrent situations (big and small ships, with and without clouds, near land and in the open sea).

## Files' description

1) OPEN_ME.ipynb shows model's Dice Score and a few photos for which model predicted binary masks. 
2) source/dataset_creator.py is used to create catalog system, copy images and extract masks from csv for model training and evaluation
3) source/train_model.py is used to train model.
4) source/predict_model.py is used to automaticly predict binary masks for all images in specific folder, show them if necessary, save them to another specific folder.
5) model_weights is a folder with splitted (because of its size) model weights archive
6) requirments.txt is a list od necessary libraries
7) test_photos.zip is an archive of a few photos shown in OPEN_ME.ipynb

## Quick run

1) Download all archives from model_weights folder, and unarchive the ALL AT ONCE
2) Download train_model.py and put it in the same folder with OPEN_ME.ipynb
3) Download and open OPEN_ME.ipynb, Insert your pathes in the In[2] cell
4) Run all cells EXCEPT In[5] and In[6] (because you don't have big dataset to evaluate)

## Model architecture

It ois a Unet with 6 levels in encoder and decoder. The loss function = dice_loss + α * binary_crossentropy
