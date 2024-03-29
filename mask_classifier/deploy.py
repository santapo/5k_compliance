import numpy as np
import cv2
import time
from .func import tf_init
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet, inception_resnet_v2
from tensorflow.keras.applications.resnet import ResNet50

tf_init()
model = load_model('mask_classifier/weights/resnet50.h5')
model_type = 'resnet50'
csize = 224


def classify_on_batch(img_arr=None, batch_size=8, thresh=0.5):
    if model_type == 'resnet50':
        img_arr = resnet.preprocess_input(img_arr)
    else:
        img_arr = inception_resnet_v2.preprocess_input(img_arr)

    match = model.predict_on_batch(img_arr)
    return (1 if match[0][0] > thresh else 0, match[0][0])


def classify(img_path='', img_arr=None, thresh=0.5):
    # import ipdb; ipdb.set_trace()
    if img_arr is None:
        img_arr = img_to_array(
            load_img(img_path, target_size=(csize, csize))
        )
    else:
        img_arr = cv2.resize(img_arr, (csize, csize), interpolation=cv2.INTER_NEAREST)

    img_arr = np.expand_dims(img_arr, axis=0)

    if model_type == 'resnet50':
        img_arr = resnet.preprocess_input(img_arr)
    else:
        img_arr = inception_resnet_v2.preprocess_input(img_arr)
    import ipdb; ipdb.set_trace()
    match = model.predict(img_arr)
    return (1 if match[0][0] > thresh else 0, match[0][0])


def classify_from_file(file_path, thresh=0.5, batch_size=32):
    imgs = open(file_path, 'r').readlines()
    labels = np.array([])
    matches = np.array([])

    batch_imgs = np.empty((0, csize, csize, 3))

    id = 0
    for img in imgs:
        img = img.strip().split(' ')
        img_path = img[0]
        label = img[1]

        labels = np.concatenate((labels, [label]), axis=0)

        img_arr = img_to_array(
            load_img(img_path, target_size=(csize, csize))
        )
        img_arr = np.expand_dims(img_arr, axis=0)
        if model_type == 'resnet50':
            img_arr = resnet50.preprocess_input(img_arr)
        else:
            img_arr = inception_resnet_v2.preprocess_input(img_arr)

        batch_imgs = np.concatenate((batch_imgs, img_arr), axis=0)

        if batch_imgs.shape[0] == batch_size:
            start_time = time.time()

            batch_matches = model.predict_on_batch(batch_imgs)

            print('Predict took', time.time() - start_time)

            for match in batch_matches:
                type = '1' if match[0] > thresh else '0'
                print(type, end=' ')
                matches = np.concatenate((matches, [type]), axis=0)
            print('')

            batch_imgs = np.empty((0, csize, csize, 3))
            id += 1
            print('Batch', id, '.....', 'acc=', np.sum(
                matches == labels) / matches.shape[0])

    if batch_imgs.shape[0] != 0:
        batch_matches = model.predict_on_batch(batch_imgs)

        for match in batch_matches:
            type = '0' if match[0] > thresh else '1'
            print(type, end=' ')
            matches = np.concatenate((matches, [type]), axis=0)
        print('')

        id += 1
        print('Batch', id, '.....', 'acc=', np.sum(
            matches == labels) / matches.shape[0])

    print('Accuracy:', np.sum(matches == labels) / len(imgs))


if __name__ == "__main__":
    classify_from_file('./data/test.txt')
