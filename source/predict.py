import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from train_model import build_model
from train_model import dice_bc_loss
from train_model import dice_score


def predict_image(model, predict_data_path, show_results):
    img = cv2.imread(predict_data_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img_tensor = np.expand_dims(img, axis=0)
    mask = model.predict(img_tensor)[0, :, :, 0]
    mask = np.where(255 * mask > 1, 255, 0).astype('uint8')
    if show_results:
        show_img_mask(img, mask)
    return img, mask


def predict_folder(model, predict_data_path, save_results_path, show_results):
    imgs_pathes = os.listdir(predict_data_path)
    for img_path in imgs_pathes:
        full_img_path = os.path.join(predict_data_path, img_path)
        _, mask = predict_image(model, full_img_path, show_results=show_results)
        if save_results_path != None:
            cv2.imwrite(os.path.join(save_results_path, img_path), mask)
    

def show_img_mask(img, mask):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()


def main():
    img_shape = (768, 768, 3)

    # Enter your values here
    save_weights_path = 'jupyt_not_vs/data/checkpoint/model_dc01bc256/my_model'
    to_predict_data_path = 'jupyt_not_vs/data/ship_dataset/test_img/img'
    save_results_path = 'jupyt_not_vs/data/save' # None, if don't want to save
    show_results = False


    model = build_model(img_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model.compile(optimizer=optimizer, loss=dice_bc_loss, metrics=[dice_score])
    model.load_weights(save_weights_path).expect_partial()

    predict_folder(model, to_predict_data_path, save_results_path, show_results)


if __name__ == "__main__":
    main()
