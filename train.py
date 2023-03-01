import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,ReLU,Add,LayerNormalization,UpSampling2D,concatenate
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tfds.disable_progress_bar()
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2,os,glob,time,pathlib
import numpy as np
#from keras_flops import get_flops
from tensorflow.keras import backend as K
#import keras_metrics as km
AUTOTUNE = tf.data.experimental.AUTOTUNE

import os
def set_gpus(gpu_index):
    if type(gpu_index) == list:
        gpu_index = ','.join(str(_) for _ in gpu_index)
    if type(gpu_index) ==int:
        gpu_index = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

set_gpus(1)


train_image_path="../graduation_project/jianzhu/train/image/"
train_label_path="../graduation_project/jianzhu/train/label/"
val_image_path="../graduation_project/jianzhu/val/image/"
val_label_path="../graduation_project/jianzhu/val/label/"
train_image_path1=pathlib.Path(train_image_path)
train_label_path1=pathlib.Path(train_label_path)
val_image_path1=pathlib.Path(val_image_path)
val_label_path1=pathlib.Path(val_label_path)


train_images,train_labels,val_images,val_labels=[],[],[],[]
train_image_names=os.listdir(train_image_path)
print('train_image_names',train_image_names)
def read_train_labels(path_name):
    label_name=path_name.split("/")
    name=label_name[-1]
    name=name.split(".")[0]
    train_label_paths=train_label_path+str(name)+".png"
    return train_label_paths

def read_val_labels(path_name1):
    label_name1=path_name1.split("/")
    name1=label_name1[-1]
    name1=name1.split(".")[0]
    val_label_paths=val_label_path+str(name1)+".png"
    return val_label_paths

train_image_paths=list(train_image_path1.glob("*"))
train_image_paths=[str(path) for path in train_image_paths]
train_label_paths=list(map(read_train_labels,train_image_paths))
print('train_image_path',train_image_paths)
print('train_label_path',train_label_paths)

val_image_paths=list(val_image_path1.glob("*"))
val_image_paths=[str(path1) for path1 in val_image_paths]
val_label_paths=list(map(read_val_labels,val_image_paths))
print('val_image',val_image_paths)
print('val_label',val_label_paths)

for i in train_image_paths:
    train_images.append(cv2.imread(i))

for j in train_label_paths:
    train_labels.append(cv2.imread(j))
#print(train_labels)

for k in val_image_paths:
    val_images.append(cv2.imread(k))

for m in val_label_paths:
    val_labels.append(cv2.imread(m))
print(val_labels)

colormap = [[128,128,128],[0,0,255],[0,192,192],[0,255,255],[0,255,0],[255,0,0]]

classes = ['barsoil','buildling','pavement','roal','vegetation','water']
print('分类类别',len(colormap))


def label_to_mask(label,color_map=colormap):
    colormap_to_mask=np.zeros(256*256*256)

    for i,color in enumerate(color_map):
        colormap_to_mask[(color[0]*256+color[1])*256+color[2]]=i#根据color中的值定义标号（类别号）

    label = cv2.resize(label, (256,256))
    label=np.array(label).astype(np.int32)
    index=((label[:,:,0]*256+label[:,:,1])*256+label[:,:,2])#获取标签图像素的索引值

    return np.array(colormap_to_mask[index], dtype="int64")


def shujuyuchuli(img):

    img=tf.cast(img,tf.float32)/255.0
    img=tf.image.resize(img,(256,256))
    return img


#查看单通道掩膜是否正确
train_images=list(map(shujuyuchuli,train_images))
train_label=list(map(label_to_mask,train_labels))
print(train_label[5][105:115, 130:140])
plt.subplot(121)
plt.title('image')
plt.imshow(train_images[0])
plt.subplot(122)
plt.title('label')
plt.imshow(train_label[0])
plt.show()
val_images=list(map(shujuyuchuli,val_images))
val_label=list(map(label_to_mask,val_labels))

train_images=tf.cast(train_images,tf.float32)
train_label=tf.cast(train_label,tf.int32)
#print(train_label)
val_images=tf.cast(val_images,tf.float32)
val_label=tf.cast(val_label,tf.int32)

#train_dataset=tf.data.Dataset.zip((train_images,train_label))
plt.figure()
plt.imshow(train_images[0])
plt.imshow(train_label[0])
plt.show()
train_dataset=tf.data.Dataset.from_tensor_slices((train_images,train_label))
val_dataset=tf.data.Dataset.from_tensor_slices((val_images,val_label))


train_dataset=train_dataset.shuffle(800).repeat().batch(8).prefetch(buffer_size=AUTOTUNE)
val_dataset=val_dataset.shuffle(800).repeat().batch(8).prefetch(buffer_size=AUTOTUNE)

#print(train_dataset)


def SeNet(input_tensor3):
    x6 = tf.keras.layers.GlobalAveragePooling2D()(input_tensor3)
    x6 = tf.keras.layers.Reshape((1, 1, x6.shape[1],))(x6)
    x6 = tf.keras.layers.Conv2D(filters=576// 16, kernel_size=1, strides=1, activation='relu')(x6)
    x6 = tf.keras.layers.Conv2D(filters=576, kernel_size=1, strides=1, activation='sigmoid')(x6)
    out = input_tensor3 * x6
    return out

def h_sigmoid(x):
    output = tf.keras.layers.Activation('hard_sigmoid')(x)

    return output


def h_swish(x):
    output = x * h_sigmoid(x)

    return output


def Squeeze_excitation_layer(x):
    inputs = x
    squeeze = inputs.shape[-1] / 2
    excitation = inputs.shape[-1]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(squeeze)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(excitation)(x)
    x = h_sigmoid(x)
    x = tf.keras.layers.Reshape((1, 1, excitation))(x)
    x = inputs * x

    return x


def BottleNeck(inputs, exp_size, out_size, kernel_size, strides, is_se_existing, activation):
    x = tf.keras.layers.Conv2D(filters=exp_size,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == "HS":
        x = h_swish(x)
    elif activation == "RE":
        x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                        strides=strides,
                                        padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation == "HS":
        x = h_swish(x)
    elif activation == "RE":
        x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    if is_se_existing:
        x = Squeeze_excitation_layer(x)
    x = tf.keras.layers.Conv2D(filters=out_size,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.linear)(x)
    if strides == 1 and inputs.shape[-1] == out_size:
        x = tf.keras.layers.add([x, inputs])

    return x


def MobileNetV3Large():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(filters=16,
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = h_swish(x)
    f1=x
    #(128 128 16)
    x = BottleNeck(x, exp_size=16, out_size=16, kernel_size=3, strides=2, is_se_existing=True, activation="RE")
    #(64 64 16)
    f2=x
    x = BottleNeck(x, exp_size=72, out_size=24, kernel_size=3, strides=2, is_se_existing=False, activation="RE")
    x = BottleNeck(x, exp_size=88, out_size=24, kernel_size=3, strides=1, is_se_existing=False, activation="RE")
    #(32 32 24)
    f3=x
    x = BottleNeck(x, exp_size=96, out_size=40, kernel_size=5, strides=2, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=240, out_size=40, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=240, out_size=40, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=120, out_size=48, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=144, out_size=48, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    #(16 16 48)
    f4=x
    x = BottleNeck(x, exp_size=288, out_size=96, kernel_size=5, strides=2, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=576, out_size=96, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    x = BottleNeck(x, exp_size=576, out_size=96, kernel_size=5, strides=1, is_se_existing=True, activation="HS")
    #(8 8 96)
    x = tf.keras.layers.Conv2D(filters=576,
                               kernel_size=(1, 1),
                               strides=1,
                               padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = h_swish(x)
    f5 = SeNet(x)
    #(8 8 576)
    up6 = concatenate([UpSampling2D(size=(2, 2))(f5), f4], axis=3)#1616 1616
    #conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(up6)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    conv6 = Activation('relu')(x)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), f3], axis=3)
    #conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(up7)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    conv7 = Activation('relu')(x)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), f2], axis=3)
    #conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(up8)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    conv8 = Activation('relu')(x)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), f1], axis=3)
    #conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(up9)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    conv9 = Activation('relu')(x)

    conv10=UpSampling2D(size=(2, 2))(conv9)
    conv10 = Conv2D(6, (1, 1), activation="sigmoid")(conv10)
    # conv10 = Conv2D(n_label, (1, 1), activation="softmax")(conv9)

    # x1=tf.keras.layers.Conv2D(filters=21,kernel_size=(1,1),kernel_regularizer=tf.keras.regularizers.l2())(x)
    # out=tf.keras.layers.Conv2DTranspose(6,kernel_size=(64,64),strides=(32,32),padding='same',activation='softmax')(x1)
    model = Model(inputs=inputs, outputs=conv10)

    return model

model = MobileNetV3Large()
print(model.summary())

#flops = get_flops(model, batch_size=16)
#print(f"FLOPS: {flops / 10 ** 9:.03} G")


checkpoint_path= '../graduation_project/checkpointconcatx/cp.ckpt'
#logs='E:/DDFcode/Image Se/draft/FCNS34/FCNS/FCN34_log2'
#u_net_tensorboard=tf.keras.callbacks.TensorBoard(log_dir=logs,histogram_freq=1)
cp_callback =tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 verbose=1)

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.5,patience=8,verbose=1)



class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

cosine_decay = tf.keras.experimental.CosineDecay(
                initial_learning_rate=0.005, decay_steps=10,)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy',MeanIoU(num_classes=6)])

history = model.fit(train_dataset, steps_per_epoch=200,batch_size=16, epochs=200,validation_steps=16,validation_freq=1,
                          validation_data=val_dataset,callbacks=[cp_callback])



model.save('FCN32_UAV.h5')
model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(200)
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.title('Xception Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Xception Training and Validation Loss')
plt.ylim([0, 1])
plt.legend()
plt.show()
