**Скрипт**
```
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
    log_dir = "C:/Users/dimas/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
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
```

**Логи**
```
C:\Python38\python.exe D:/nns/lab4/transfer-learn.py
2020-12-28 20:52:16.286076: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-28 20:52:18.115027: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-28 20:52:18.125175: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library nvcuda.dll
2020-12-28 20:52:18.195785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-28 20:52:18.195943: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-28 20:52:18.712787: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-28 20:52:18.712874: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-28 20:52:18.776443: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-28 20:52:18.809829: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-28 20:52:19.091001: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-28 20:52:19.316418: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-28 20:52:20.269140: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-28 20:52:20.269285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-28 20:52:21.531153: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-28 20:52:21.531246: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2020-12-28 20:52:21.531293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2020-12-28 20:52:21.532383: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4616 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:06:00.0, compute capability: 7.5)
2020-12-28 20:52:21.534484: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-28 20:52:21.539069: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-28 20:52:21.539184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-28 20:52:21.539315: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-28 20:52:21.539381: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-28 20:52:21.539451: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-28 20:52:21.539531: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-28 20:52:21.539603: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-28 20:52:21.539821: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-28 20:52:21.539881: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-28 20:52:21.539986: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-28 20:52:21.540086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-28 20:52:21.540509: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-28 20:52:21.540632: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-28 20:52:21.540771: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-28 20:52:21.540881: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-28 20:52:21.540978: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-28 20:52:21.541053: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-28 20:52:21.541119: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-28 20:52:21.541185: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-28 20:52:21.541251: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-28 20:52:21.541345: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-28 20:52:21.541441: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-28 20:52:21.541510: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2020-12-28 20:52:21.541555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2020-12-28 20:52:21.541673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4616 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:06:00.0, compute capability: 7.5)
2020-12-28 20:52:21.541966: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-28 20:52:21.616781: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Count of train images: 56536
Count of validation images: 910
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 1)]     0         
_________________________________________________________________
tf.image.grayscale_to_rgb (T (None, 224, 224, 3)       0         
_________________________________________________________________
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984   
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 128)       2621568   
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 56, 56, 128)       65664     
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 112, 112, 64)      32832     
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 224, 224, 2)       514       
_________________________________________________________________
dense (Dense)                (None, 224, 224, 2)       6         
=================================================================
Total params: 4,978,568
Trainable params: 2,720,584
Non-trainable params: 2,257,984
_________________________________________________________________
None
2020-12-28 20:52:25.891899: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2020-12-28 20:52:25.891962: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
2020-12-28 20:52:25.892459: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1365] Profiler found 1 GPUs
2020-12-28 20:52:25.921064: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cupti64_110.dll
2020-12-28 20:52:26.006803: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2020-12-28 20:52:26.007305: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed
Epoch 1/100
2020-12-28 20:52:28.789820: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-28 20:52:30.036694: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-28 20:52:30.486717: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-28 20:52:34.381683: E tensorflow/stream_executor/cuda/cuda_driver.cc:851] failed to alloc 4294967296 bytes on host: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2020-12-28 20:52:34.381774: W .\tensorflow/core/common_runtime/gpu/gpu_host_allocator.h:44] could not allocate pinned host memory of size: 4294967296
2020-12-28 20:52:34.381893: E tensorflow/stream_executor/cuda/cuda_driver.cc:851] failed to alloc 3865470464 bytes on host: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2020-12-28 20:52:34.381974: W .\tensorflow/core/common_runtime/gpu/gpu_host_allocator.h:44] could not allocate pinned host memory of size: 3865470464
2020-12-28 20:52:34.382056: E tensorflow/stream_executor/cuda/cuda_driver.cc:851] failed to alloc 3478923264 bytes on host: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2020-12-28 20:52:34.382130: W .\tensorflow/core/common_runtime/gpu/gpu_host_allocator.h:44] could not allocate pinned host memory of size: 3478923264
2020-12-28 20:52:44.240148: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2020-12-28 20:52:44.501619: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2020-12-28 20:52:46.145702: W tensorflow/core/common_runtime/bfc_allocator.cc:248] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-12-28 20:52:46.145895: W tensorflow/core/common_runtime/bfc_allocator.cc:248] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
      1/Unknown - 25s 25s/step - loss: 0.23452020-12-28 20:52:51.612914: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2020-12-28 20:52:51.612984: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
      2/Unknown - 27s 3s/step - loss: 0.2341 2020-12-28 20:52:53.561538: I tensorflow/core/profiler/lib/profiler_session.cc:71] Profiler session collecting data.
2020-12-28 20:52:53.565332: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed
2020-12-28 20:52:53.686252: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:228]  GpuTracer has collected 489 callback api events and 461 activity events. 
2020-12-28 20:52:53.706686: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2020-12-28 20:52:53.750178: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53
2020-12-28 20:52:53.762728: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.trace.json.gz
2020-12-28 20:52:53.822935: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53
2020-12-28 20:52:53.828967: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.memory_profile.json.gz
2020-12-28 20:52:53.842774: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53Dumped tool data for xplane.pb to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.xplane.pb
Dumped tool data for overview_page.pb to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.overview_page.pb
Dumped tool data for input_pipeline.pb to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201228-205221\train\plugins\profile\2020_12_28_17_52_53\DESKTOP-81L2T6G.kernel_stats.pb
...
Epoch 11/100
448/448 [==============================] - 178s 397ms/step - loss: 0.0586 - val_loss: 0.0578
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4351135606504577
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.6544329132580895, min: 0.0, mean: 1.9311017759246467
Epoch 12/100
448/448 [==============================] - 178s 398ms/step - loss: 0.0584 - val_loss: 0.0576
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4341582457565215
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.64483176403773, min: 0.0, mean: 1.9298113505263725
Epoch 13/100
448/448 [==============================] - 179s 399ms/step - loss: 0.0589 - val_loss: 0.0576
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43383708232430745
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.640766510963823, min: 0.0, mean: 1.9296884576874629
Epoch 14/100
448/448 [==============================] - 179s 399ms/step - loss: 0.0590 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4331611236362581
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.633315430696341, min: 0.0, mean: 1.9286307166875043
Epoch 15/100
448/448 [==============================] - 183s 408ms/step - loss: 0.0594 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4330103032840145
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.630514474251012, min: 0.0, mean: 1.928695227415944
Epoch 16/100
448/448 [==============================] - 176s 392ms/step - loss: 0.0596 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43278888155039213
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.628276811551956, min: 0.0, mean: 1.9290836077957334
Epoch 17/100
448/448 [==============================] - 175s 391ms/step - loss: 0.0586 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43337382725310264
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.629881040692251, min: 0.0, mean: 1.929192390441517
Epoch 18/100
448/448 [==============================] - 179s 399ms/step - loss: 0.0596 - val_loss: 0.0578
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43641358956221427
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.647887921291765, min: 0.0, mean: 1.9320366966356712
Epoch 19/100
448/448 [==============================] - 181s 404ms/step - loss: 0.0581 - val_loss: 0.0574
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4319236173024985
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.618251658117834, min: 0.0, mean: 1.9277287360864126
Epoch 20/100
448/448 [==============================] - 178s 397ms/step - loss: 0.0582 - val_loss: 0.0577
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43577032080563105
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.643265391880765, min: 0.0, mean: 1.9319813223651934
Epoch 21/100
448/448 [==============================] - 175s 390ms/step - loss: 0.0590 - val_loss: 0.0576
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43491683235639367
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.636342370123102, min: 0.0, mean: 1.930725580030419
Epoch 22/100
448/448 [==============================] - 175s 391ms/step - loss: 0.0575 - val_loss: 0.0577
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43605099070815917
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.644029611463816, min: 0.0, mean: 1.932109694005627
Epoch 23/100
448/448 [==============================] - 176s 393ms/step - loss: 0.0600 - val_loss: 0.0576
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4349828442200337
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.6351237555780225, min: 0.0, mean: 1.9301623836431292
Epoch 24/100
448/448 [==============================] - 175s 391ms/step - loss: 0.0590 - val_loss: 0.0577
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43535941328516214
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.636717458029346, min: 0.0, mean: 1.9302608126659286
Epoch 25/100
448/448 [==============================] - 175s 391ms/step - loss: 0.0598 - val_loss: 0.0577
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4358907221969446
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.641181783687231, min: 0.0, mean: 1.9315357385454965
Epoch 26/100
448/448 [==============================] - 179s 400ms/step - loss: 0.0598 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4343318625502241
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.6309799437131565, min: 0.0, mean: 1.929642795427705
Epoch 27/100
448/448 [==============================] - 176s 393ms/step - loss: 0.0580 - val_loss: 0.0577
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43568010987358785
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.6398408569274965, min: 0.0, mean: 1.9303908283937297
Epoch 28/100
448/448 [==============================] - 175s 391ms/step - loss: 0.0580 - val_loss: 0.0576
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4353098846525182
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.640186627605776, min: 0.0, mean: 1.930850335909694
Epoch 29/100
448/448 [==============================] - 171s 381ms/step - loss: 0.0591 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43384583743825617
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.631582602649182, min: 0.0, mean: 1.9290941301443123
Epoch 30/100
448/448 [==============================] - 174s 388ms/step - loss: 0.0594 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4339527874072555
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.634164311992462, min: 0.0, mean: 1.9294887038736461
Epoch 31/100
448/448 [==============================] - 172s 383ms/step - loss: 0.0594 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4343966724687394
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.639647448307715, min: 0.0, mean: 1.9305200555618143
Epoch 32/100
448/448 [==============================] - 174s 388ms/step - loss: 0.0588 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4339342858640299
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.636841607574638, min: 0.0, mean: 1.929214526755667
Epoch 33/100
448/448 [==============================] - 181s 403ms/step - loss: 0.0589 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.4339058371829378
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.639352859679976, min: 0.0, mean: 1.9298236395381172
Epoch 34/100
448/448 [==============================] - 1304s 3s/step - loss: 0.0573 - val_loss: 0.0575
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.40914328748382806
Max in predicted: 1.0, min: 0.0, mean: 0.43402226582789066
Info about image in RGB format
Max in target: 7.4309067635543675, min: 0.0, mean: 1.8841250422828806
Max in predicted: 5.641886542275319, min: 0.0, mean: 1.9297836959904215
Epoch 35/100
101/448 [=====>........................] - ETA: 22:45 - loss: 0.05792020-12-28 23:00:10.783703: E tensorflow/stream_executor/cuda/cuda_event.cc:29] Error polling for event status: failed to query event: CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered
2020-12-28 23:00:10.787524: F tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc:220] Unexpected Event status: 1

Process finished with exit code -1073740791 (0xC0000409)

```

**Рузльтаты**
Запуск 1. optimizer=tf.optimizers.Adam(lr=0.01)
![1run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_1.png)
![1run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars1.png)

Запуск 2. optimizer=tf.optimizers.Adam(lr=0.01)
![2run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_2.png)
![2run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars2.png)

Запуск 3. optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9)
![3run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_3.png)
![3run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars3.png)
