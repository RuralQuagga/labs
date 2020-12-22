__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
from keras.layers import Conv2D, InputLayer, Conv2DTranspose
from tensorflow.keras import layers
from keras.models import Sequential
import tensorflow_io as tfio

SHUFFLE_BUFFER = 4
BATCH_SIZE = 256
NUM_CLASSES = 6
PARALLEL_CALLS = 4
RESIZE_TO = 224
TRAINSET_SIZE = 14034
VALSET_SIZE = 3000

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def visualize_images(epoch, model, dataset, writer):
    item = iter(dataset).next()

    l_channel = item[:, :, :, 0]

    target_ab = item[:, :, :, 1:]
    target_image = np.zeros(item.shape)
    target_image[:, :, :, 0] = l_channel
    target_image[:, :, :, 1:] = target_ab

    predicted_ab = model(np.reshape(l_channel, (-1, 224, 224, 1)))
    predicted_image = np.zeros(item.shape)
    predicted_image[:, :, :, 0] = l_channel
    predicted_image[:, :, :, 1:] = predicted_ab

    target_rgb = tfio.experimental.color.lab_to_rgb(target_image)
    predicted_rgb = tfio.experimental.color.lab_to_rgb(predicted_image)

    with writer.as_default():
        tf.summary.image('Target Lab', np.reshape(target_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Result Lab', np.reshape(predicted_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Target RGB', target_rgb, step=epoch)
        tf.summary.image('Result RGB', predicted_rgb, step=epoch)


def visualize_images_augmented(epoch, dataset, writer):
    item = dataset[0]

    with writer.as_default():
        tf.summary.image('Augmented', np.reshape(item, (-1, 224, 224, 3)), step=epoch)


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    example = tf.io.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image']


def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .batch(batch_size)\
        .prefetch(batch_size)


def build_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(224, 224, 3)))

    model.add(Conv2D(1, (2, 2), activation='relu', padding='same'))

    model.add(Conv2D(32, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2D(64, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2D(128, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(2, (2, 2), strides=2, activation='relu', padding='same'))

    return model


def main():
    log_dir = "C:/Users/Alex/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    train_dir = Path(current_dir + "/../train_tfr")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    file_list_train = tf.random.shuffle(file_list_train)
    c = 0
    for fn in file_list_train:
        for record in tf.data.TFRecordDataset(fn):
            c += 1
    print(f'Count of train images: {c}')

    valid_dir = Path(current_dir + "/../validation_tfr")
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    file_list_valid = tf.random.shuffle(file_list_valid)
    v = 0
    for fn in file_list_valid:
        for record in tf.data.TFRecordDataset(fn):
            v += 1
    print(f'Count of validation images: {v}')

    train_dataset = create_dataset(file_list_train, BATCH_SIZE)
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(112, 112),
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.experimental.preprocessing.RandomRotation(factor=0.45),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal')
    ])
    augmented = [data_augmentation(i) for i in train_dataset]

    validation_dataset = create_dataset(file_list_valid, BATCH_SIZE)

    validation_y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in validation_dataset]

    x = [np.reshape(i[:, :, :, 0], (-1, 224, 224, 1)) for i in augmented]
    y = [np.reshape(i[:, :, :, 1:], (-1, 224, 224, 2)) for i in validation_dataset]

    model = build_model()

    model.compile(
         optimizer=tf.optimizers.SGD(lr=0.01, momentum=0.9),
         loss=tf.keras.losses.mean_squared_error
    )

    model.fit(
        x=x.pop(),
        y=y.pop(),
        epochs=100,
        validation_data=validation_y.pop().all(),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, validation_dataset, file_writer)
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images_augmented(epoch, augmented, file_writer)
            )
        ]
    )

    print(model.summary())


if __name__ == '__main__':
    main()
