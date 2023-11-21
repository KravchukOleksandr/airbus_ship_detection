# airbus_ship_detection
Detecting ships in satellite's photos 

## Files' description

1) OPEN_ME.ipynb shows model's Dice Score and a few photos for which model predicted binary masks. 
2) source/dataset_creator.py is used to create catalog system for model training and evaluation
3) source/train_model.py is used to train model.
4) source/predict_model.py is used to automaticly predict binary masks for all images in specific folder, show them if necessary, save them to another specific folder.
5) model_weights is a folder with splitted (because of its size) model weights archive
6) requirments.txt is a list od necessary libraries
7) test_photos.zip is an archive of a few photos shown in OPEN_ME.ipynb
