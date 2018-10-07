from PIL import Image, ImageDraw, ImageFont
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils  import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import csv
import time
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

print('model loading...')
model = load_model('./model/cnn_model.hdf5')

testStart = 1001
testEnd = 5001 #test number

print("Reading data...")
x_train = np.stack([np.array(Image.open("./img_p/" + str(index) + ".jpg"))/255.0 for index in range(testStart, testEnd, 1)])

print('predict start')
prediction = model.predict(x_train)
print('preficted ')
resultlist = ["" for _ in range(testEnd - testStart + 1)]

for predict in prediction:
	for index in range(testEnd - testStart):
		resultlist[index] += str(np.argmax(predict[index]))

#traincsv = open('/label.csv', 'r', encoding = 'utf8')
#cipher_label = [row[0] for row in csv.reader(traincsv)]
#read_label =  [to_onelist(row[0]) for row in csv.reader(traincsv)]

count = 1001 
#correct = 0
for result in resultlist:
	print(str(count) + " : " + result)
	# if result == cipher_label[count]:
		# correct += 1
	
	count += 1


