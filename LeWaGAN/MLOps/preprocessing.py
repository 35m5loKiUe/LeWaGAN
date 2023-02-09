import tensorflow


#Create a dataset from a local directory
def create_dataset(path, image_size=(128, 128), labels=None, color_mode='rgb', batch_size=32,
                     validation_split=0.) :
    """This function create a dataset for tensorflow from a directory with images
    arguments :
    path : path of the directory (not the path of images inside the directory !!)
    image_size : (tuples like (28,28)) image_size
    labels : None for non-classification tasks (keep it to none)
    color_mode : 'rgb', 'grayscale' or 'rbga', 'rgb' by default
    batch_size : batch size for the dataset
    validation_split : float between 0 and 1
    """

    dataset = tensorflow.keras.utils.image_dataset_from_directory(path=path, color_mode=color_mode,
                                           labels=labels, image_size=image_size,
                                           batch_size=batch_size, validation_split=validation_split)
    return dataset
