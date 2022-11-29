import os
from PIL import Image
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import xlrd
import cv2 as cv
import time

# 定义一个转化excel为字典的函数
def exl_read(paths):
    data = xlrd.open_workbook(paths)
    table = data.sheet_by_name('train')
    row = table.nrows  # 行数
    datas = {}  # 这步也要转字典类型
    for i in range(1, row):
        kk = dict([table.row_values(i)])  # 这一步就要给它转字典类型，不然update没法使用
        datas.update(kk)
    list_key = []
    list_value = []
    for i, j in datas.items():  # 这一步的主要作用是将文件名中的'.jpg'去掉
        list_key.append(i[:-4])
        list_value.append(j)
        new = dict(zip(list_key, list_value))
    return new


# 读取路径下的所有文件并将文件路径保存到filelist中，并返回。
def img_read(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


# 图片数据转化为np.array数组
def img_array(paths):
    M = []
    for filename in paths:
        im = Image.open(filename)
        Core = im.getdata()
        arr1 = np.array(Core, dtype='float32') / 255.0
        list_img = arr1.tolist()
        M.extend(list_img)
    return M


# ***********************************数据的初始化*************************** #
path = 'D:/pla/tray'  
path_1 = 'D:/pla/train.xls'
filelist = img_read(path)
lable_dict = exl_read(path_1)
M = []
M = img_array(filelist)
train_imgs = np.array(M).reshape(len(filelist), 64, 64, 3)  # 训练数据准备
lable = []
for filename in filelist:  # 把filelist中的文件名截出来，然后作为字典的键，从字典中取标签。
    lable.append(lable_dict[filename[12:-4]])
train_lables = np.array(lable)  # 数据标签转化为array数组
# print(train_images.shape)  # 输出验证一下数据尺寸

# ********************************训练数据、输出并保存模型**************************** #
if __name__=='__main__':   
    start = time.time()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())

    model.add(layers.Dropout(0.2))  # drop层：进一步降低过拟合
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(6, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # epochs为训练多少轮、batch_size为每次训练多少个样本
    baocun = model.fit(train_imgs, train_lables, batch_size=25,epochs=50,validation_split=0.1)

    # ranges = range(25)
    # tacc =baocun.history['accuracy']
    # tloss = baocun.history['loss']
    # # vtacc = baocun.history['val_accuracy']
    # # vloss = baocun.history['val_loss']
    # plt.figure(figsize= (16,8))
    # plt.subplot(1, 2, 1)
    # plt.plot(ranges,tacc,label='train acc')
    # plt.plot(ranges,tloss,label='train loss')
    # plt.title('ACC')
    # plt.xlabel('Epochs')
    # plt.ylabel('acc')
    # plt.show()
    # if ord('q') == cv.waitKey(0):
    #     plt.close()
    # cv.destroyAllWindows()

    ranges = range(50)
    tacc =baocun.history['accuracy']
    tloss = baocun.history['loss']
    vtacc = baocun.history['val_accuracy']
    vloss = baocun.history['val_loss']
    plt.figure(figsize= (16,8))
    plt.subplot(1, 2, 1)
    plt.plot(ranges,tacc,label='train acc')
    plt.plot(ranges,vtacc,label='train loss')
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(ranges,tloss,label='loss')
    plt.plot(ranges,vloss,label='val_loss')
    plt.title('loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    end = time.time()
    print(str(end-start))
    if ord('q') == cv.waitKey(0):
            plt.close()
    cv.destroyAllWindows()
    model.save('my_model.h5')  # 保存为h5模型
    tf.keras.models.save_model(model,"D://pla/plt/model")  # pb模型
    print("模型保存成功！")
