import tensorflow as tf
import glob
import os

def load_and_preprocess_image(filename, label):
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

def build_dataset(file_paths, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def get_file_paths_and_labels(data_dir):
    file_paths = glob.glob(os.path.join(data_dir, "*.png"))
    labels = [list(map(int, os.path.basename(fp).split('_')[0])) for fp in file_paths]
    return file_paths, labels


train_folder = 'path/to/train/folder'
val_folder = 'path/to/val/folder'

train_file_paths, train_labels = get_file_paths_and_labels(train_folder)
val_file_paths, val_labels = get_file_paths_and_labels(val_folder)

train_dataset = build_dataset(train_file_paths, train_labels, batch_size=32)
val_dataset = build_dataset(val_file_paths, val_labels, batch_size=32)