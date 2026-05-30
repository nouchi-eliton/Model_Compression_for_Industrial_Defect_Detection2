import tensorflow as tf
import numpy as np
import os
import imageio.v3 as imageio
from sklearn.model_selection import train_test_split


def load_images(folder_path):
    """
    Load all images within the folder_path and its subfolders.
    Assign label 0 for images whose parent folder is named "good", and label 1 otherwise.

    Args:
        folder_path: string indicating the path to the images root folder.

    Returns:
        images: numpy.array with all images found within the folder_path.
        y: numpy.array with all image labels
    """
    images = []
    y = []
    for root, direc, files in os.walk(folder_path): # searches all folders and subfolders for files
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    img = imageio.imread(file_path)
                    images.append(img)
                    if os.path.split(root)[1] == 'good':
                        y.append(0)
                    else:
                        y.append(1)
                except Exception as e:
                    print(f'Não foi possível abrir a imagem {file_path}: {e}')

    images = np.stack(images)
    y = np.stack(y)

    return images, y


def merge_dataset(img_train, img_test, y_train, y_test):
    """
    Concatenates train and test datasets.

    Args:
        img_train: numpyarray containing the train images.
        img_test: numpyarray containing the test images.
        y_train: training image labels
        y_test: test image labels

    Returns:
        imgs: numpyarray with the train and test images concatenated
        y: numpyarray with the train and test labels concatenated
    """

    imgs = np.vstack((img_train, img_test))
    y = np.concatenate((y_train, y_test), axis=0)

    return imgs, y


def preprocess_dataset(dataset_root_path, obj, seed=45):
    """
    Creates a new split containing images from both classes in training, validation, and test sets.

    Args:
        train_path: path to the training images folder.
        test_path: path to the test images folder.
        seed: int. Value used to initialize the pseudo-random number generator.

    Returns:
        Three tuples containing the training, validation, and test sets in a format: (images, labels)
    """
    train_path = os.path.join(dataset_root_path, obj, 'train')
    test_path = os.path.join(dataset_root_path, obj, 'test')
    X_train, y_train = load_images(train_path)
    X_test, y_test = load_images(test_path)
    X, y = merge_dataset(X_train, X_test, y_train, y_test)

    # Add color dimensions in grayscale images
    if len(X.shape) != 4:
        X = np.expand_dims(X, axis=-1)
        X = np.repeat(X, 3, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, 
                                                        stratify= y,
                                                        random_state= seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, 
                                                      stratify= y_train, 
                                                      random_state= seed)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

