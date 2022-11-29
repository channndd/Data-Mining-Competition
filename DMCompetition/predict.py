import os
from PIL import Image
import numpy as np
import tensorflow as tf

# 导入图像数据
# 测试外部图片
model = tf.keras.models.load_model('game_model.h5')
model.summary()  # 看一下网络结构

print("模型加载完成！")
dict_label = {0: "Cassava Bacterial Blight (CBB)", 1: "Cassava Brown Streak Disease (CBSD)", 2: "Cassava Green Mottle (CGM)", 3: "Cassava Mosaic Disease (CMD)", 4: "Healthy",5:"apple leaves(AL)"}


def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".jpg":
                filelist.append(os.path.join(root, file))
    return filelist


def im_xiangsu(paths):
    for filename in paths:
        try:
            im = Image.open(filename)
            newim = im.resize((128, 128))
            newim.save('CNN/test_a/' + filename[10:-4] + '.jpg')
            print('图片' + filename[10:-4] + '.jpg' + '像素转化完成')
        except OSError as e:
            print(e.args)

test = 'CNN/test_a'  # 你要测试的图片的路径
filelist = read_image(test)
im_xiangsu(filelist)

# f = open('shiyan.csv','w')
# cbb = cbsd = cgm = cmd = hea = al = 0

for filename in filelist:
    im = Image.open(filename)
    im_L = im.convert("L")  # 模式L
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0#标准化
    list_img = arr1.tolist()
    images = np.array(list_img).reshape(-1, 128, 128, 1)
    img = images
    # 预测图像
    predictions_single = model.predict(img)
    print(f"{filename[10:-4]}的预测结果为:{dict_label[np.argmax(predictions_single)]}")