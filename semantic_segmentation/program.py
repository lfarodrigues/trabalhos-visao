# Objective: segment 24 different objects
# Dataset: https://www.tugraz.at/index.php?id=22387
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.python.keras.metrics import MeanIoU
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.all_utils import to_categorical
from tensorflow.python.keras import layers, models
import random
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint,  EarlyStopping
from keras_applications.resnet import ResNet50
import segmentation_models as sm
from sklearn.model_selection import train_test_split

data_folder = 'data/semantic_drone_dataset/training_set/'

def load_data(folder):
    imgs_data = []
    imgs_num = 50
    for images in os.listdir(folder):
        #print('dirname: ')
        #print(os.path.basename(images))
        img_path = os.path.join(folder, images)
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = Image.fromarray(img)
        img = np.array(img)
        imgs_data.append(img)
        imgs_num -= 1
        if imgs_num == 0:
            break
    return imgs_data

img_dataset = load_data(data_folder + '/' + 'images/')
mask_dataset = load_data(data_folder + 'gt' + '/' + 'semantic' + '/' +'label_images/')

image_dataset = np.array(img_dataset)
mask_dataset = np.array(mask_dataset)

img_num = random.randint(0, len(img_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[img_num])
plt.subplot(122)
plt.imshow(mask_dataset[img_num])
plt.show()

mask_labels = pd.read_csv('data/semantic_drone_dataset/training_set/gt/semantic/class_dict.csv') # mask labels RGB values...

# converting 3 values (RGB) to 1 label values...
def cnv_rgb_labels(img, mask_labels):
    label_seg = np.zeros(img.shape,dtype=np.uint8)
    for i in range(mask_labels.shape[0]):
        label_seg[np.all(img == list(mask_labels.iloc[i, [1,2,3]]), axis=-1)] = i
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels...
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = cnv_rgb_labels(mask_dataset[i], mask_labels)
    labels.append(label)

labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

#Another Sanity check...
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()

n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes )# one hot enconding

X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42) # train test split for data...

#################################
# Network model
#################################
# Uses transfer learning from a pre trained resnet model to set the initial weights.
BACKBONE = 'resnet34'
# using weights from pretained resnet model...
preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')

metrics = ['accuracy']
# Create the U-Net model
model_resnet_backbone.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = metrics)

model_resnet_backbone.summary()

# training the model for 100 epochs with batch size of 16...
history = model_resnet_backbone.fit(X_train_prepr, 
          y_train,
          batch_size=16, 
          epochs=100,
          verbose=1,
          validation_data=(X_test_prepr, y_test))

# show the results
history = history
accuracy = history.history['accuracy']
val_accuraccy = history.history['accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'y', label = 'Training Accuracy')
plt.plot(epochs, val_accuraccy, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model_resnet_backbone.save('models/resnet_backbone.hdf5')
model = load_model('models/resnet_backbone.hdf5') # loading model...
# predict
y_pred = model.predict(X_test)
y_pred_argmax = np.argmax(y_pred, axis = 3)
y_test_argmax = np.argmax(y_test, axis = 3)

# randomly select an image
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
gt = y_test_argmax[test_img_number]
test_img_in = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_in))
predicted_img = np.argmax(prediction, axis = 3)[0, :, :]

# plotting the real image, test labeled image and predicted labeled image...
plt.figure(figsize=(16, 12))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(gt)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()