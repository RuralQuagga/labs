__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""

import os
from datetime import datetime
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
import tensorflow_io as tfio
import numpy as np

SHUFFLE_BUFFER = 4
BATCH_SIZE = 128
NUM_CLASSES = 6
PARALLEL_CALLS = 4
RESIZE_TO = 224
TRAINSET_SIZE = 14034
VALSET_SIZE = 3000
TRAIN_FOLDER = 'train_tfr'
VALIDATION_FOLDER = 'validation_tfr'


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def generator_train():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(112, 112),
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.experimental.preprocessing.RandomRotation(factor=0.45),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal')
    ])

    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            dataset = data_augmentation(dataset)
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            yield (x, y)


def generator_valid():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            yield (x, y)


def visualize_images(epoch, model, dataset, writer):
    item = iter(dataset).next()

    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    predicted_ab = model(np.reshape(l_channel, (-1, 224, 224, 1)))
    predicted_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    predicted_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))
    predicted_image[:, :, :, 1:] = predicted_ab
    maxl = tf.math.reduce_max(target_image)
    minl = tf.math.reduce_min(target_image)
    meanl = tf.math.reduce_mean(target_image)
    print(f'Info about image in Lab format')
    print(f'Max in target: {maxl}, min: {minl}, mean: {meanl}')

    maxlp = tf.math.reduce_max(predicted_image)
    minlp = tf.math.reduce_min(predicted_image)
    meanlp = tf.math.reduce_mean(predicted_image)
    print(f'Max in predicted: {maxlp}, min: {minlp}, mean: {meanlp}')

    target_rgb = tfio.experimental.color.lab_to_rgb(target_image)
    target_rgb = tf.math.multiply(target_rgb, 256)
    predicted_rgb = tfio.experimental.color.lab_to_rgb(predicted_image) * 256

    max = tf.math.reduce_max(target_rgb)
    min = tf.math.reduce_min(target_rgb)
    mean = tf.math.reduce_mean(target_rgb)
    print(f'Info about image in RGB format')
    print(f'Max in target: {max}, min: {min}, mean: {mean}')

    maxp = tf.math.reduce_max(predicted_rgb)
    minp = tf.math.reduce_min(predicted_rgb)
    meanp = tf.math.reduce_mean(predicted_rgb)
    print(f'Max in predicted: {maxp}, min: {minp}, mean: {meanp}')

    with writer.as_default():
        tf.summary.image('Target Lab', np.reshape(target_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Result Lab', np.reshape(predicted_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Target RGB', target_rgb, step=epoch)
        tf.summary.image('Result RGB', predicted_rgb, step=epoch)


def visualize_images_augmented(epoch, dataset, writer):
    item = iter(dataset).next()
    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    with writer.as_default():
        tf.summary.image('Augmented', target_image, step=epoch)


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


def display_image_count():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    file_list_train = tf.random.shuffle(file_list_train)
    c = 0
    for fn in file_list_train:
        for record in tf.data.TFRecordDataset(fn):
            c += 1
    print(f'Count of train images: {c}')

    valid_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    file_list_valid = tf.random.shuffle(file_list_valid)
    v = 0
    for fn in file_list_valid:
        for record in tf.data.TFRecordDataset(fn):
            v += 1
    print(f'Count of validation images: {v}')


def main():
    log_dir = "C:/Users/Alex/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    display_image_count()

    train = tf.data.Dataset.from_generator(
        generator_train,
        (tf.float32, tf.float32))

    valid = tf.data.Dataset.from_generator(
        generator_valid,
        (tf.float32, tf.float32))

    IMG_SHAPE = (RESIZE_TO, RESIZE_TO, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(2)
    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 1))
    x = tf.image.grayscale_to_rgb(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=4, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(2, (2, 2), strides=2, activation='relu', padding='same')(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
         optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9),
         loss=tf.keras.losses.mean_squared_error
    )

    print(model.summary())

    model.fit(
        train,
        epochs=100,
        validation_data=valid,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, valid, file_writer)
            ),
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=lambda epoch, logs: visualize_images_augmented(epoch, train, file_writer)
            # )
        ]
    )


if __name__ == '__main__':
    main()
