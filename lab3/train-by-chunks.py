__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from datetime import datetime
import tensorflow as tf
from pathlib import Path
from keras.layers import Conv2D, InputLayer, Conv2DTranspose
from tensorflow.keras import layers
from keras.models import Sequential
import tensorflow_io as tfio
import numpy as np

SHUFFLE_BUFFER = 4
BATCH_SIZE = 256
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

    #current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path("G:/train_tf/")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            dataset = data_augmentation(dataset)
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            # print(f'\nIteration: {i}. Train')
            yield (x, y)


def generator_valid(args):
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path("G:/test_tfr/")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            # print(f'\nIteration {i}. Validation')
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

    target_rgb = tfio.experimental.color.lab_to_rgb(target_image)
    predicted_rgb = tfio.experimental.color.lab_to_rgb(predicted_image)

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


def display_image_count(args):
    #current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(args.train)
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    file_list_train = tf.random.shuffle(file_list_train)
    c = 0
    for fn in file_list_train:
        for record in tf.data.TFRecordDataset(fn):
            c += 1
    print(f'Count of train images: {c}')

    valid_dir = Path(args.test)
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    file_list_valid = tf.random.shuffle(file_list_valid)
    v = 0
    for fn in file_list_valid:
        for record in tf.data.TFRecordDataset(fn):
            v += 1
    print(f'Count of validation images: {v}')


def build_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(224, 224, 1)))

    model.add(Conv2D(1, (2, 2), activation='relu', padding='same'))

    model.add(Conv2D(32, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2D(64, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2D(128, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding='same'))

    model.add(Conv2DTranspose(2, (2, 2), strides=2, activation='relu', padding='same'))

    return model


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files')
    args.add_argument('--test', type=str, help='Glob pattern to collect test tfrecord files')
    args = args.parse_args()

    log_dir = "C:/Users/dimas/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    display_image_count(args)

    train = tf.data.Dataset.from_generator(
        generator_train,
        (tf.float32, tf.float32))

    valid = tf.data.Dataset.from_generator(
        generator_valid,
        (tf.float32, tf.float32))

    model = build_model()

    model.compile(
         optimizer=tf.optimizers.SGD(lr=0.01, momentum=0.9),
         loss=tf.keras.losses.mean_squared_error
    )

    model.fit(
        train,
        epochs=50,
        validation_data=valid,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, valid, file_writer)
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images_augmented(epoch, train, file_writer)
            )
        ]
    )

    print(model.summary())


if __name__ == '__main__':
    main()
    
