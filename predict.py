import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, \
    concatenate, Dropout, UpSampling2D
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,ReLU,Add
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tfds.disable_progress_bar()
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.image as imgplt
import cv2, os, glob, time, pathlib
import numpy as np
from tensorflow.keras.models import Sequential
from sklearn import metrics, neighbors
import tensorflow.python.keras.layers as layers
import tensorflow.python.keras.metrics as metric
import tensorflow.python.keras.models as models
AUTOTUNE = tf.data.experimental.AUTOTUNE

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def set_gpus(gpu_index):
    if type(gpu_index) == list:
        gpu_index = ','.join(str(_) for _ in gpu_index)
    if type(gpu_index) ==int:
        gpu_index = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

set_gpus(1)






from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D
from tensorflow.keras import Model
import tensorflow as tf

img_w = 256
img_h = 256
#有一个为背景
n_label = 6
INPUT_SHAPE = (256, 256, 3)

#预测的时候用哪种模型预测就把那种模型的代码换上去，然后在compile，这里用的是bisenetV2
#（Yu, C., Gao, C., Wang, J. et al. BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation. Int J Comput Vis 129, 3051–3068 (2021). https://doi.org/10.1007/s11263-021-01515-2）

def ge_layer(x_in, c, e=6, stride=1):
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if stride == 2:
        x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)

        y = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding='same')(x_in)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)
    else:
        y = x_in

    x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, y])
    x = layers.Activation('relu')(x)
    return x


def stem(x_in, c):
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x_split = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=c // 2, kernel_size=(1, 1), padding='same')(x_split)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    y = layers.MaxPooling2D()(x_split)

    x = layers.Concatenate()([x, y])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def detail_conv2d(x_in, c, stride=1):
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=stride, padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def context_embedding(x_in, c):
    x = layers.GlobalAveragePooling2D()(x_in)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape((1, 1, c))(x)

    x = layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # broadcasting no needed

    x = layers.Add()([x, x_in])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding='same')(x)
    return x


def bilateral_guided_aggregation(detail, semantic, c):
    # detail branch
    detail_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(detail)
    detail_a = layers.BatchNormalization()(detail_a)

    detail_a = layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(detail_a)

    detail_b = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding='same')(detail)
    detail_b = layers.BatchNormalization()(detail_b)

    detail_b = layers.AveragePooling2D((3, 3), strides=2, padding='same')(detail_b)

    # semantic branch
    semantic_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(semantic)
    semantic_a = layers.BatchNormalization()(semantic_a)

    semantic_a = layers.Conv2D(filters=c, kernel_size=(1, 1), padding='same')(semantic_a)
    semantic_a = layers.Activation('sigmoid')(semantic_a)

    semantic_b = layers.Conv2D(filters=c, kernel_size=(3, 3), padding='same')(semantic)
    semantic_b = layers.BatchNormalization()(semantic_b)

    semantic_b = layers.UpSampling2D((4, 4), interpolation='bilinear')(semantic_b)
    semantic_b = layers.Activation('sigmoid')(semantic_b)

    # combining
    detail = layers.Multiply()([detail_a, semantic_b])
    semantic = layers.Multiply()([semantic_a, detail_b])

    # this layer is not mentioned in the paper !?
    # semantic = layers.UpSampling2D((4,4))(semantic)
    semantic = layers.UpSampling2D((4, 4), interpolation='bilinear')(semantic)

    x = layers.Add()([detail, semantic])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    return x


def seg_head(x_in, c_t, s, n):
    x = layers.Conv2D(filters=c_t, kernel_size=(3, 3), padding='same')(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=n, kernel_size=(3, 3), padding='same')(x)
    x = layers.UpSampling2D((s, s), interpolation='bilinear')(x)

    return x


class ArgmaxMeanIOU(metric.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def bisenetv2(num_classes=6, out_scale=8, input_shape=INPUT_SHAPE, l=4, seghead_expand_ratio=2):
    x_in = layers.Input(input_shape)

    # semantic branch
    # S1 + S2
    x = stem(x_in, 64 // l)

    # S3
    x = ge_layer(x, 128 // l, stride=2)
    x = ge_layer(x, 128 // l, stride=1)

    # S4
    x = ge_layer(x, 64, stride=2)
    x = ge_layer(x, 64, stride=1)

    # S5
    x = ge_layer(x, 128, stride=2)

    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)

    x = context_embedding(x, 128)

    # detail branch
    # S1
    y = detail_conv2d(x_in, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)

    # S2
    y = detail_conv2d(y, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)
    y = detail_conv2d(y, 64, stride=1)

    # S3
    y = detail_conv2d(y, 128, stride=2)
    y = detail_conv2d(y, 128, stride=1)
    y = detail_conv2d(y, 128, stride=1)

    x = bilateral_guided_aggregation(y, x, 128)

    x = seg_head(x, num_classes * seghead_expand_ratio, out_scale, num_classes)

    model = models.Model(inputs=[x_in], outputs=[x])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.HeNormal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.HeNormal()
    print(model.summary())
    return model
model=bisenetv2()



class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)



model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001,nesterov=True,momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy',MeanIoU(num_classes=6)])


path ='../graduation_project/checkpointlwsgd/cp.ckpt'
model.load_weights(path)

test_image_path="../graduation_project/jianzhu/train/image/"
test_label_path="../graduation_project/jianzhu/train/label/"
test_image_path1=pathlib.Path(test_image_path)
test_label_path1=pathlib.Path(test_label_path)
test_images,test_labels=[],[]

test_image_paths=list(test_image_path1.glob("*"))
test_image_paths=[str(path) for path in test_image_paths]
print('test_image_paths',test_image_paths)
b=[]
for k in test_image_paths:
    a=k.split("/")
    b.append(a[5].split('.')[0])
    test_images.append(cv2.imread(k))
print(b)

for m in range(len(b)):
    labe_path=test_label_path+str(b[m])+'.png'
    test_labels.append(cv2.imread(labe_path))


colormap = [[128,128,128],[0,0,255],[0,192,192],[0,255,255],[0,255,0],[255,0,0]]

classes = ['barsoil','water','pavement','roal','vegetation','building']
print('分类类别',len(colormap))


def label_to_mask(label, color_map=colormap):
    colormap_to_mask = np.zeros(256 * 256 * 256)
    for i, color in enumerate(color_map):
        colormap_to_mask[(color[0] * 256 + color[1]) * 256 + color[2]] = i  # 根据color中的值定义标号（类别号）
    label = cv2.resize(label, (256,256))
    label = label.astype(np.int32)
    index = ((label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2])  # 获取标签图像素的索引值
    print(colormap_to_mask[index])
    return colormap_to_mask[index].astype(np.uint8)


def shujuyuchuli(img):
    # img=tf.io.read_file(img)
    # img=tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, (256,256))
    return img


def shujuyuchuli11(img):
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    img = img / 255.
    img = tf.image.resize(img, (256,256))
    #print('image.shape', img.shape)
    return img


def predicts(val_images):
    # print(val_images.shape)
    print("11111111")
    predict = model.predict(val_images)
    clock = time.time()
    predict = model.predict(val_images)
    print(time.time() - clock, 's')
    predict = tf.argmax(predict, axis=-1)
    predict = predict[..., tf.newaxis]
    predict = np.squeeze(predict, axis=0)
    predict = np.squeeze(predict, axis=-1)
    return predict


def mask_to_onehot(mask, num_class=21):
    semantic_map = [mask == i for i in range(num_class)]
    abc = np.array(semantic_map)
    return np.array(semantic_map).astype(np.uint32)


def onehot_to_label(mp, color_map=colormap):
    index = np.argmax(mp, axis=0)
    #print('index', index.shape)
    color_maparr = np.array(color_map)
    #print('color_maparr', color_maparr.shape)
    label2 = np.uint32(color_maparr[index.astype(np.uint32)])
    return label2



image = (test_images[14])
# image= np.zeros([256,256, 3], dtype='float32')
image1 = shujuyuchuli11(image)
labell = (test_labels[14])
labell=cv2.resize(labell,(256,256))
#print(labell)
mask=label_to_mask(labell)
print('label的单通道掩膜',mask[130:150,130:150],mask.shape)

mask2=predicts(image1)

print("1111111111111111111111111111111111111111111111111111111111111111111")
print('测试的标签',mask2[130:150,130:150])

#print('mask.shape', mask.shape)
kong=np.zeros((256,256,3),dtype=int)

kong1=np.zeros((256,256,3),dtype=int)

camp=np.array(colormap)
#print(camp.shape)
#kong[:,:,0][mask]=camp[1,0]
#print(kong[:,:,0][mask].shape)
for label in range(0,len(camp)):
    mask1=mask==label
    #print(mask==label,(mask==label).shape)
    kong[:,:,0][mask1]=camp[label,0]
    kong[:,:,1][mask1]=camp[label,1]
    kong[:,:,2][mask1]=camp[label,2]

for label2 in range(0,len(camp)):
    mask3=mask2==label2
    #print(mask==label,(mask==label).shape)
    kong1[:,:,0][mask3]=camp[label2,0]
    kong1[:,:,1][mask3]=camp[label2,1]
    kong1[:,:,2][mask3]=camp[label2,2]

print(kong.shape)
print(kong1.shape)
plt.figure()
plt.subplot(221)
plt.axis('off')
plt.title('origin_image')
plt.imshow(image1[0])
#plt.show()
plt.subplot(222)
plt.axis('off')
plt.title('label')
labell = labell[:,:,(2,1,0)]
plt.imshow(labell)
#plt.show()
plt.subplot(223)
plt.title('predict')
kong1 = kong1[:,:,(2,1,0)]
plt.imshow(kong1)
plt.axis('off')
plt.show()

plt.imshow(kong1)
plt.show()
#cv2.imwrite("../graduation_project/111.jpg",kong)
#cv2.imwrite("../graduation_project/1111.jpg",kong1)


im = kong1
im = np.array(im)
im2 = labell
im2 = np.array(im2)
cal_outputs = im.flatten()
label = im2.flatten()
c = metrics.confusion_matrix(cal_outputs, label)
FP = c.sum(axis=0) - np.diag(c)
FN = c.sum(axis=1) - np.diag(c)
TP = np.diag(c)
TN = c.sum() - (FP + FN + TP)
#print(FP)
list_diag = np.diag(c)
list_raw_sum = np.sum(c, axis=1)

#each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
#print(each_acc)
#average = np.mean(each_acc)
#all_acc += average
MIou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
mean_iou=np.mean(MIou)
print(mean_iou)
print(type(MIou))
'''
#合图
plt.figure()
plt.subplot(121)
plt.title('origin')
plt.axis('off')
plt.imshow(image1[0])
plt.imshow(labell,alpha=0.5)
plt.subplot(122)
plt.axis('off')
plt.title('predict')
plt.imshow(image1[0])
plt.imshow(kong1,alpha=0.5)
plt.show()
'''






