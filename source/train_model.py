import os
import tensorflow as tf


# Unet Architecture
def build_model(img_shape):
    input_layer = tf.keras.layers.Input(shape = img_shape)

    # ENCODER

    # (768, 768, 3)
    conv_layer0 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(input_layer)
    conv_layer0 = tf.keras.layers.Conv2D(8, (3,3), padding='same', activation='relu')(conv_layer0)
    maxpool_layer0 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer0)

    # (384, 384, 8)
    conv_layer1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(maxpool_layer0)
    conv_layer1 = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(conv_layer1)
    maxpool_layer1 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer1)

    # (192, 192, 16)
    conv_layer2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(maxpool_layer1)
    conv_layer2 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(conv_layer2)
    maxpool_layer2 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer2)

    # (96, 96, 32)
    conv_layer3 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(maxpool_layer2)
    conv_layer3 = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(conv_layer3)
    maxpool_layer3 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer3)

    # (48, 48, 64)
    conv_layer4 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(maxpool_layer3)
    conv_layer4 = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(conv_layer4)
    maxpool_layer4 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer4)

    # (24, 24, 128)
    conv_layer5 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(maxpool_layer4)
    conv_layer5 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu')(conv_layer5)
    maxpool_layer5 = tf.keras.layers.MaxPooling2D(2,2)(conv_layer5)

    # (12, 12, 256)
    conv_layer6 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu')(maxpool_layer5)
    conv_layer6 = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu')(conv_layer6)
    
    # DECODER

    # (12, 12, 512)
    upconv_layer5 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(conv_layer6)
    concated5 =  tf.keras.layers.concatenate([conv_layer5, upconv_layer5])
    upconv_layer5 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(concated5)
    upconv_layer5 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(upconv_layer5)
    
    # (24, 24, 256)
    upconv_layer4 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(upconv_layer5)
    concated4 =  tf.keras.layers.concatenate([conv_layer4, upconv_layer4])
    upconv_layer4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(concated4)
    upconv_layer4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(upconv_layer4)
    
    # (48, 48, 128)
    upconv_layer3 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(upconv_layer4)
    concated3 =  tf.keras.layers.concatenate([conv_layer3, upconv_layer3])
    upconv_layer3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(concated3)
    upconv_layer3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(upconv_layer3)

    # (96, 96, 64)
    upconv_layer2 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(upconv_layer3)
    concated2 =  tf.keras.layers.concatenate([conv_layer2, upconv_layer2])
    upconv_layer2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(concated2)
    upconv_layer2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(upconv_layer2)
    
    # (192, 192, 32)
    upconv_layer1 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(upconv_layer2)
    concated1 =  tf.keras.layers.concatenate([conv_layer1, upconv_layer1])
    upconv_layer1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(concated1)
    upconv_layer1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu')(upconv_layer1)
    
    # (384, 384, 16)
    upconv_layer0 = tf.keras.layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), 
                                                    padding='same', activation='relu')(upconv_layer1)
    concated0 =  tf.keras.layers.concatenate([conv_layer0, upconv_layer0])
    upconv_layer0 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(concated0)
    upconv_layer0 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu')(upconv_layer0)
    
    # (768, 768, 8)

    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(upconv_layer0)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
    return model


# Prepare trainnig dataset for fitting the model
def train_generator(img_folder_path, mask_folder_path, batch_size):
    seed = 909
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    image_generator = image_datagen.flow_from_directory(img_folder_path, batch_size=batch_size, 
                                                        class_mode=None, seed=seed)
    mask_generator = mask_datagen.flow_from_directory(mask_folder_path, batch_size=batch_size, 
                                                      class_mode=None, seed=seed)
    return zip(image_generator, mask_generator)


# Prepare validation dataset for measuring performance during training
def validation_generator(val_img_folder_path, val_mask_folder_path, batch_size):
    seed = 909
    val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_image_generator = val_image_datagen.flow_from_directory(val_img_folder_path, batch_size=batch_size, 
                                                        class_mode=None, seed=seed)
    val_mask_generator = val_mask_datagen.flow_from_directory(val_mask_folder_path, batch_size=batch_size, 
                                                      class_mode=None, seed=seed)
    return zip(val_image_generator, val_mask_generator)


# Getting train/validation data and steps_per_epoch/validation_steps parameter
def fit_parameters(img_folder_path, mask_folder_path,
                   val_img_folder_path, val_mask_folder_path, 
                   batch_size):
    train_data = train_generator(img_folder_path, mask_folder_path, batch_size)
    val_data = validation_generator(val_img_folder_path, val_mask_folder_path, batch_size)
    steps_per_epoch = int(len(os.listdir(img_folder_path + '/img')) / batch_size)
    validation_steps = int(len(os.listdir(val_img_folder_path + '/img')) / batch_size)
    return train_data, val_data, steps_per_epoch, validation_steps


# Performance measure, dice score on binary images
def dice_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.where(y_pred > 1/2550.0, 1.0, 0.0)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / denominator


# Dice coefficient with no binarization (used to calculate loss)
def dice_sm_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred) + 0.0001
    denominator = tf.reduce_sum(y_true + y_pred) + 0.0001
    return numerator / denominator


# Dice loss
def dice_sm_loss(y_true, y_pred):
    return 1 - dice_sm_metric(y_true, y_pred)


# Combo loss (used for training)
def dice_bc_loss(y_true, y_pred):
    return  dice_sm_loss(y_true, y_pred) + 0.1 * tf.keras.losses.binary_crossentropy(y_true, y_pred)


def main():
    img_shape = (768, 768, 3)
    batch_size = 256

    # Enter your values here
    img_folder_path = 'jupyt_not_vs/data/ship_dataset/train_img'
    mask_folder_path = 'jupyt_not_vs/data/ship_dataset/train_mask'
    val_img_folder_path = 'jupyt_not_vs/data/ship_dataset/test_img'
    val_mask_folder_path = 'jupyt_not_vs/data/ship_dataset/test_mask'
    load_weights_path = 'jupyt_not_vs/data/checkpoint/model_dc01bc256/my_model'
    save_weights_path = 'jupyt_not_vs/data/checkpoint/model_dc01bc256/my_model'
    save_last = 'jupyt_not_vs/data/checkpoint/last/my_model'

    model = build_model(img_shape)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025)
    model.compile(optimizer=optimizer, loss=dice_bc_loss, metrics=[dice_score])

    print('Load weights? (Y/n)')
    load_model = input()
    if load_model == 'Y':
        model.load_weights(load_weights_path)

    train_data, val_data, steps_per_epoch, validation_steps = fit_parameters(img_folder_path, mask_folder_path, 
                                                                             val_img_folder_path, val_mask_folder_path, 
                                                                             batch_size)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(save_weights_path, save_weights_only=True,
                                                       save_best_only=True, monitor='val_dice_score', mode='max')
    
    model.fit(train_data, validation_data=val_data, epochs=10, 
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
              callbacks=[checkpoint_cb])
    model.save_weights(save_last)

    print('Success!')


if __name__ == "__main__":
    main()
