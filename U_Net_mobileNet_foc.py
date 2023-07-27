!pip install -U -q segmentation-models
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

path_1 = '/content/data/train_images/train/'
path_2 = '/content/data/train_masks/train/'
path_3 = '/content/data/val_images/val/'
path_4 = '/content/data/val_masks/val/'
paths = [path_1,path_2,path_3,path_4]

for path in paths:
  if not os.path.exists(path):
    os.makedirs(path)


X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size = 0.01, random_state = None)


datas = [X_train,y_train,X_test,y_test]
count = 0
for i in range(len(paths)):
    path = paths[i]
    for file_name in datas[i]:
      count += 1
      shutil.copy(file_name, path)

seed=42
batch_size= 8

BACKBONE = 'mobilenetv2'
preprocess_input = sm.get_preprocessing(BACKBONE)


def preprocess_data(img,mask, num_class):
    mask = to_categorical(mask, num_class) # конвертация в категориальный формат
    return img,mask
def trainGenerator(train_img_path, train_mask_path, num_class):
  img_data_gen_args = dict(rescale = 1/255.,
                           #horizontal_flip=True
                           )

  mask_data_gen_args = dict(#rescale = 1/255.,
                      )


  image_data_generator = ImageDataGenerator(**img_data_gen_args)
  mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

  image_generator = image_data_generator.flow_from_directory(train_img_path,
                                                           seed=seed,
                                                           target_size = (224,224),
                                                           batch_size=batch_size,
                                                           class_mode=None)
  mask_generator = mask_data_generator.flow_from_directory(train_mask_path,
                                                         seed=seed,
                                                         target_size = (224,224),
                                                         batch_size=batch_size,
                                                         color_mode = 'grayscale',
                                                         class_mode=None)

  train_generator = zip(image_generator, mask_generator)

  for (img, mask) in train_generator:
        img, mask = preprocess_data(img,mask, num_class)
        yield (img, mask)

train_img_path = "/content/data/train_images/"
train_mask_path = "/content/data/train_masks/"
train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=4)

val_img_path = "/content/data/val_images/"
val_mask_path = "/content/data/val_masks/"
val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=4)

x, y = train_img_gen.__next__()
print(y.shape)
for i in range(0,3):
    image = x[i]
    #mas = y[i]
    mask = np.argmax(y[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

x_val, y_val = val_img_gen.__next__()

for i in range(0,3):
    image = x_val[i]
    mask = np.argmax(y_val[i], axis=2)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.show()

num_train_imgs = len(os.listdir('/content/data/train_images/train/'))
num_val_images = len(os.listdir('/content/data/val_images/val'))
steps_per_epoch = num_train_imgs//batch_size
val_steps_per_epoch = num_val_images//batch_size


IMG_HEIGHT = 224#x.shape[1]
IMG_WIDTH  = 224#x.shape[2]
IMG_CHANNELS = 3#x.shape[3]
n_classes=4
BACKBONE = 'mobilenetv2'
from tensorflow.keras.optimizers import Adam
lr = 0.0001
optim = keras.optimizers.Adam(lr)
model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                 classes=n_classes, activation='softmax')
model.compile('Adam', loss=sm.losses.categorical_focal_loss, metrics=['accuracy',sm.metrics.iou_score,sm.metrics.FScore()])

model.summary()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir('/content/result/')

#model_path = os.path.join("/content/result/", "model_0.h5")
model_path = os.path.join('***/result/', "model_last_512_unet_foc_4.0.h5")

#csv_path = os.path.join("/content/result/", "data_0.csv")
csv_path = os.path.join('***/result/', "data_last_512_unet_foc_4.0.csv")

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard

#Step 3: Initialize Tensorboard to monitor changes in Model Loss
import datetime
%load_ext tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#Visualize on tensorboard (move this above)
%tensorboard --logdir logs/fit

""" Training """
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path, append=True),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False),
    TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)
]

history=new_model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=40,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch,
          callbacks = callbacks)


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

iou = history.history['iou_score']
val_iou = history.history['val_iou_score']

plt.plot(epochs, iou, 'y', label='Training IoU')
plt.plot(epochs, val_iou, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


acc = history.history['iou_score']
val_acc = history.history['val_iou_score']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

from tensorflow.keras.metrics import MeanIoU
test_image_batch, test_mask_batch = val_img_gen.__next__()
print(test_image_batch.shape)
print(test_image_batch[0].shape)
#Convert categorical to integer for visualization and IoU calculation
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
test_pred_batch = new_model.predict(test_image_batch)
print(test_pred_batch.shape)
print(test_pred_batch[0].shape)
test_pred_batch_another = (new_model.predict(test_image_batch)[0,:,:,0] > 0.5).astype(np.uint8)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
print(test_pred_batch_argmax.shape)
print(test_pred_batch_argmax[0].shape)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

# plt.imshow(train_images[0, :,:,0], cmap='gray')
# plt.imshow(train_masks[0], cmap='gray')

import random
#######################################################
#View a few images, masks and corresponding predictions.
img_num = random.randint(0, test_image_batch.shape[0]-1)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_image_batch[img_num])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_batch_argmax[img_num])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_pred_batch_argmax[img_num])#test_pred_batch_another


plt.show()

CLASSES_COLORS = {0: (68, 1, 84), #  черный цвет для фона
                1: (51, 99, 141),     # красный цвет для класса 1
                2: (60, 187, 117),     # зеленый цвет для класса 2
                3: (253, 231, 37)}     # синий цвет для класса 3

def grayscale_to_rgb(pred, class_colors):
    h, w = pred.shape

    pred = pred.astype(np.int32)
    output = []
    for i, pixel in enumerate(pred.flatten()):
        output.append(class_colors[pixel])
    output = np.reshape(output,(h,w,3))
    return output

def save_results(image_x,mask_x,pred,save_image_path):
    h,w = image_x.shape
    line = np.ones((h,10,3))*255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES_COLORS)
    mask_x = grayscale_to_rgb(mask_x,CLASSES_COLORS)


    cat_images = np.concatenate([image_x, line,mask_x,line,pred],axis = 1)
    cv2.imwrite(save_image_path, cat_images)


def threshold_soil(pred,mask):
    #pred
    values, counts = np.unique(pred, return_counts=True) #число пикселей каждого класса
    total = sum(counts)
    percent_soil = list(map(lambda x: (x*100)/total,counts)) # процент каждого класса
    percent_soil_dict_pred = dict(zip(values, percent_soil))
    #mask
    values, counts = np.unique(mask, return_counts=True) #число пикселей каждого класса
    total = sum(counts)
    percent_soil = list(map(lambda x: (x*100)/total,counts)) # процент каждого класса
    percent_soil_dict_mask = dict(zip(values, percent_soil))
    print(percent_soil_dict_mask)
    if 0 in percent_soil_dict_pred:

        if 2 not in percent_soil_dict_pred and 1 in percent_soil_dict_pred and 3 not in percent_soil_dict_pred:
            if percent_soil_dict_pred[0]>=percent_soil_dict_pred[1]:

              text = "Well: "+str(int(percent_soil_dict_pred[0]))+"% **"

              color = (0,255,0)
              signal_value = 0
              print(f'Well: only clean {int(percent_soil_dict_pred[0])}% and transparent {int(percent_soil_dict_pred[1])}%-> Status Cleanup:{signal_value}')

            else:
              text = "Must clean: "+str(int(percent_soil_dict_pred[0]))+"%"

              color = (0,0,255)
              signal_value = 1

              print(f'Must clean: only clean {int(percent_soil_dict_pred[0])}% and transparent {int(percent_soil_dict_pred[1])}% -> Status Cleanup:{signal_value}')
              
        elif percent_soil_dict_pred[0] < 80:
            text = "Must clean: "+str(int(percent_soil_dict_pred[0]))+"%"
            color = (0,0,255)
            signal_value = 1

            print(f'Must clean:{int(percent_soil_dict_pred[0])}% -> Status Cleanup:{signal_value}')

        elif percent_soil_dict_pred[0] >= 80:
            text = "Well: "+str(int(percent_soil_dict_pred[0]))+"%"

            color = (0,255,0)
            signal_value = 0

            print(f'Well:{int(percent_soil_dict_pred[0])}% -> Status Cleanup:{signal_value}')


    else:
        print('desirable'+'% is unknown -> Status Cleanup: 1')
        text = 'desirable'+'% is unknown'

        color = (205,87,0)
        signal_value = 1
    position = (5,10)
    if 0 in percent_soil_dict_mask:
        text_1 = 'Tru area: '+str(int(percent_soil_dict_mask[0]))+"%"

    else:
        text_1 = 'Tru area: % is unknown'
    img_per = cv2.putText(image_x,text, position,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    position = (5,23)
    img_per = cv2.putText(image_x,text_1, position,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return img_per,signal_value

# from keras.models import load_model
# new_model_1 = load_model('****/model_224_mobilenetv2_unet_last_2.0.h5',custom_objects={'sm.losses.categorical_focal_loss':sm.losses.categorical_focal_loss},compile=False)
# images_test = sorted(images)
# masks_test = sorted(masks)

# save_image_path = '*****/images_results/'

# from PIL import Image


# #for name in images_test:
# for i in range(len(images_test)):

#   name_img = images_test[i]
#   name_msk = masks_test[i]
#   image = cv2.imread(name_img)
#   mask = cv2.imread(name_msk,0)

#   image = cv2.resize(image,(224,224))
#   mask = cv2.resize(mask,(224,224))
#   onehot_mask = []
#   for j in range(4):
#     onehot_mask.append(np.where(np.equal(mask, j), 1, 0))
#   mask_test = np.stack(onehot_mask, axis=-1)
#   mask_test_argmax = np.argmax(mask_test, axis=-1)

#   image_x = image
#   image = image/255.0
#   image = np.expand_dims(image, axis=0)

#   pred = new_model_1.predict(image, verbose=0)[0]

#   pred = np.argmax(pred, axis=-1)


#   line = np.ones((224,10,3))*255
#   mask_test_rgb = grayscale_to_rgb(mask_test_argmax,CLASSES_COLORS)
#   mask_pred_rgb = grayscale_to_rgb(pred,CLASSES_COLORS)

#   img_per,signal_value = threshold_soil(pred,mask_test_argmax)

#   print(f'status:{signal_value}')
#   name = str(i)+'_image.png'
#   print(name)
#   cat_images = np.concatenate([img_per,line,mask_test_rgb,line,mask_pred_rgb],axis = 1)

#   path = os.path.join(save_image_path,name)
#   cv2.imwrite(path, cat_images)
