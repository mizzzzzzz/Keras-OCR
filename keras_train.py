
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils  import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras import optimizers

import csv

NUM_OF_DIGIT = 6
NUM_OF_DOMAIN = 10

def toOnelist(text):
    labelList = []
    for c in text:
        oneHot = [0 for _ in range(NUM_OF_DOMAIN)]
        oneHot[int(c)] = 1
        labelList.append(oneHot)
    return labelList

def toText(list):
    text=""
    for i in range(NUM_OF_DIGIT):
        for j in range(NUM_OF_DOMAIN):
            if(list[i][j]):
                text += str(j)


#creat CNN model
print('Creating CNN model...')
tensorIn = Input((48, 140, 3))
tensorOut = tensorIn
tensorOut = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(tensorOut)
tensorOut = BatchNormalization(axis=1)(tensorOut)
tensorOut = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(tensorOut)
tensorOut = MaxPooling2D(pool_size=(2, 2))(tensorOut)

tensorOut = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(tensorOut)
tensorOut = BatchNormalization(axis=1)(tensorOut)
tensorOut = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensorOut)
tensorOut = MaxPooling2D(pool_size=(2, 2))(tensorOut)

#tensorOut = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensorOut)
#tensorOut = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensorOut)
#tensorOut = MaxPooling2D(pool_size=(2, 2))(tensorOut)
#tensorOut = BatchNormalization(axis=1)(tensorOut)
#tensorOut = MaxPooling2D(pool_size=(2, 2))(tensorOut)

tensorOut = Flatten()(tensorOut)
tensorOut = Dropout(0.5)(tensorOut)

tensorOut = [Dense(NUM_OF_DOMAIN, name='digit1', activation='softmax')(tensorOut),\
              Dense(NUM_OF_DOMAIN, name='digit2', activation='softmax')(tensorOut),\
              Dense(NUM_OF_DOMAIN, name='digit3', activation='softmax')(tensorOut),\
              Dense(NUM_OF_DOMAIN, name='digit4', activation='softmax')(tensorOut),\
			  Dense(NUM_OF_DOMAIN, name='digit5', activation='softmax')(tensorOut),\
			  Dense(NUM_OF_DOMAIN, name='digit6', activation='softmax')(tensorOut)]

model = Model(inputs=tensorIn, outputs=tensorOut)
#model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.compile( loss = "categorical_crossentropy", optimizer = 'adam', metrics=['accuracy'])
model.summary()
              
print("Reading data...")

dataCsv = open('./label.csv', 'r', encoding = 'utf8')
readLabel = [toOnelist(row[0]) for row in csv.reader(dataCsv)]
numOfTrainData = 1 + int(len(readLabel) * 2 / 3)

trainLabel = [[] for _ in range(NUM_OF_DIGIT)]
for arr in readLabel:
    for index in range(NUM_OF_DIGIT):
        trainLabel[index].append(arr[index])
trainLabel = [arr for arr in np.asarray(trainLabel)]


trainData = np.stack([np.array(Image.open("./img_p/" + str(index) + ".jpg"))/255.0 for index in range(1, len(readLabel) + 1, 1)])
print("Shape of train data:", trainData.shape)

filepath='./model/cnn_model.hdf5'
try:
    model = load_model(filepath)
    print('model is loaded...')
except:
    model.save(filepath)
    print('training new model...')

checkpoint = ModelCheckpoint(filepath, monitor='val_digit4_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir = './logs', histogram_freq = 1)
callbacksList = [tensorBoard, earlystop, checkpoint]
#model.fit(trainData, trainLabel, batch_size=50, epochs=40, verbose=2, validation_data=(validData, validLabel), callbacks=callbacksList)
model.fit(trainData, trainLabel, validation_split=0.3, batch_size=500, epochs=40, verbose=2, callbacks=callbacksList)
# tensorboard --logdir= (dist)

