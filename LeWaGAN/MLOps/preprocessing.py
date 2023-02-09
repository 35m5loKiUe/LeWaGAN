import tensorflow
from LeWaGAN.MLOps.params import IMAGE_SIZE

#Create a dataset from a local directory
def create_dataset(path, image_size=(IMAGE_SIZE, IMAGE_SIZE), labels=None, color_mode='rgb', batch_size=32,
                     validation_split=0.) :
    """This function create a dataset for tensorflow from a directory with images
    arguments :
    path : path of the directory (not the path of images inside the directory !!)
    image_size : (tuples like (28,28)) image_size
    color_mode : 'rgb', 'grayscale' or 'rbga', 'rgb' by default
    validation_split : float between 0 and 1
    """

    #Instanciate dataset
    dataset = tensorflow.keras.utils.image_dataset_from_directory(path=path, color_mode=color_mode,
                                           labels=None, image_size=image_size,
                                           batch_size=None, validation_split=validation_split)

    #Normalization / Scaling
    func = lambda x : x/255
    n_dataset = dataset.map(map_func=func)

    #Batch the dataset with specified batch_size
    f_dataset = n_dataset.batch(batch_size=batch_size, drop_remainder=False)

    return f_dataset
