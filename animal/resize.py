
import os
import numpy as np
from PIL import Image

file_path = r''
img = Image.open(file_path)  # 读取图片，格式为Image

# 把Image格式的图片转化为numpy格式进性图像处理
img = np.array(img, dtype=np.uint8)

height, width, mode = img.shape[0], img.shape[1], img.shape[2]  # 取出高、宽、通道数
print(height, width, mode)  # (275 183 3)

# 缩放的目标大小，这里以缩放为原图的1/2为例
desWidth = int(width * 0.5)
desHeight = int(height * 0.5)
desImage = np.zeros((desHeight, desWidth, mode), np.uint8)  # 定义一个目标图片代表的array，纯黑图片

# 像素填充
# 方法1：最近邻插值法
for des_x in range(0, desHeight):
    for des_y in range(0, desWidth):
        # 判断新像素点在原图中的像素点坐标
        src_x = int(des_x * (height/desHeight))
        src_y = int(des_y * (width/desWidth))

        desImage[des_x, des_y] = img[src_x, src_y]  # 填充
print(desImage.shape)
des_img = Image.fromarray(desImage)
des_img.save('./image/jerry.jpg')  # 图片另存为

