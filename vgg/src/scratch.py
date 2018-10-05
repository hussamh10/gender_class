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

def train():
    vgg = VGG16(include_top = False, classes=2, input_shape=(imageX, imageY, 3))

    input = Input(shape=(imageX, imageY, 3), name = 'image_input')

    vgg = vgg(input)

    x = Flatten(name='flatten')(vgg)
    x = Dense(2, activation='softmax', name='predictions')(x)

    vgg = Model(input=input, output=x)

    vgg.summary()

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    vgg.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    tb = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    X, Y = loadData("clean")
    X /= 255
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    batch_size = 32
    nb_epoch = 1000

    filepath="weights-improvement-{epoch:02d}.hdf5"

    cp = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')

    print('training ========================================================================================')

    vgg.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=[X_test, y_test],
              shuffle=True,
              callbacks=[tb, cp])

train()
