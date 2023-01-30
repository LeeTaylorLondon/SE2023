from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from object_storage import *
import numpy as np
import os


def image_embedding(fd, model, debug=False):
    # Load the image
    img_path = fd
    img = Image.open(img_path)
    img = img.resize((224, 224))
    # Preprocess the image
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Extract the image embedding
    image_embedding = model.predict(x)
    if debug:
        # Print the image embedding (IE)
        print(image_embedding)
        # Print the IE shape
        print(image_embedding.shape)
    return image_embedding

def dict_embeddings(img_fd='Dataset/trial_images_v1'):
    # Dictionary of images
    images = {}
    # Load the VGG-16 model
    model = VGG16(weights='imagenet', include_top=False)
    # Get a list of all files in the directory
    files = os.listdir(img_fd)
    for file in files:
        try:
            images.update({file: image_embedding(
                f"{img_fd}/{file}"
            , model=model)})
        except:
            pass
    return images


if __name__ == '__main__':
    # _ = dict_embeddings()
    # print(_.keys())

    # Store object as readable file
    # store_obj(_, "Object Storage/image_dict_.pkl.gz")

    # Load object
    images_ = load_obj("Object Storage/image_dict_.pkl.gz")
    print(images_.keys())

    print(list(images_.values())[0].shape)
