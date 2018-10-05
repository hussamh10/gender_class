from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

imageX = 100
imageY = 100

def loadData(dir):
    lines = open(dir).read().split('\n')[:-1]
    X = []
    Y = []
    for line in tqdm(lines):
        try:
            x, y = line.split('|')
            x = load_img(x, target_size=(imageX, imageY))
            x = img_to_array(x)
            yoh = [0, 0]
            yoh[int(y)]=1
            Y.append(yoh)
            X.append(x)
        except:
            print('missed image')

    X = np.array(X)
    Y = np.array(Y)
    X = preprocess_input(X)
    print(X.shape, Y.shape)

    return X, Y

def getModel(file_path):
    vgg = VGG16(include_top = False, classes=2, input_shape=(imageX, imageY, 3))

    input = Input(shape=(imageX, imageY, 3), name = 'image_input')

    vgg = vgg(input)

    x = Flatten(name='flatten')(vgg)
    x = Dense(2, activation='softmax', name='predictions')(x)

    vgg = Model(input=input, output=x)

    vgg.summary()

    vgg.load_weights(file_path)

    return vgg

def test():
    vgg = getModel('model-014.h5')
    X, y = loadData('testimgs')
    print(vgg.predict_classes(X))
    print(y)
