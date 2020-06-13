import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.core import input_data,dropout,fully_connected, time_distributed, flatten
from tflearn.layers.estimator import regression
import cv2
from sklearn.utils import shuffle

loadedImages = []
for i in range(0, 980):
    image = cv2.imread('Dataset/Training/A/a_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Berapa/berapa_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Kamu/kamu_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/L/l_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Nama/nama_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Samasama/sama_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
#
for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Saya/saya_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
#
for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Sayang/sayang_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Terimakasih/terimakasih_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 980):
    image = cv2.imread('Dataset/Training/Umur/umur_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

outputVectors = []
for i in range(0, 1000):
    outputVectors.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

testImages = []
for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/A/a_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Berapa/berapa_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Kamu/kamu_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/L/l_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Nama/nama_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Samasama/sama_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Saya/saya_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
#
for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Sayang/sayang_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Terimakasih/terimakasih_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

for i in range(0, 100):
    image = cv2.imread('Dataset/Validation/Umur/umur_' + str(i) + '.png')
    image = cv2.resize(image,(89,100),1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))


testLabels = []
for i in range(0, 100):
    testLabels.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


tf.reset_default_graph()
cnnlstmmodel=input_data(shape=[None,89,100,1],name='input')

cnnlstmmodel=conv_2d(cnnlstmmodel,32,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,64,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,128,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,256,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,512,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,256,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,128,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,64,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel=conv_2d(cnnlstmmodel,32,2,activation='relu')
cnnlstmmodel=max_pool_2d(cnnlstmmodel,2)

cnnlstmmodel = time_distributed(cnnlstmmodel, flatten, args=['flat'])
cnnlstmmodel = lstm(cnnlstmmodel, 1024)

cnnlstmmodel=fully_connected(cnnlstmmodel,1000,activation='relu')
cnnlstmmodel=dropout(cnnlstmmodel,0.7)

cnnlstmmodel=fully_connected(cnnlstmmodel,10,activation='softmax')

cnnlstmmodel=regression(cnnlstmmodel,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')
model=tflearn.DNN(cnnlstmmodel,tensorboard_verbose=0)

loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)
model.fit(loadedImages, outputVectors, n_epoch=30,
          validation_set = (testImages, testLabels),
          snapshot_step=100, show_metric=True, run_id='cnnlstmmodel_coursera')

model.save("TrainedModelCNNLSTM/CNNLSTMFIX.tfl")

graph = tf.Graph()
tf.Session(graph=graph)