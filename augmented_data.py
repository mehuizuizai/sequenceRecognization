from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
# RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"K","L","R","S","U","X","Y"]
RANGE_DIR = [0,1,2,3,4,5,6,7,8,9,"A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
class DataAugmentation:
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        image.save(path)


if __name__ == '__main__':
    for i in RANGE_DIR:
        dir = 'E:/胎号所有/训练_图片大于50resize_28/%s/' % i
        # dir_ran_color = 'E:/胎号所有/训练_图片大于50resize_28_ran_color/%s/' %i
        dir_ran_Gaussion ='E:/胎号所有/训练_图片大于50resize_28_ran_Gassu/%s/' %i
        try:
            os.listdir(dir)
        except Exception:
            continue
        # if not os.path.exists(dir_ran_color):
        #     os.makedirs(dir_ran_color)
        if not os.path.exists(dir_ran_Gaussion):
            os.makedirs(dir_ran_Gaussion)
        for rt, dirs, files in os.walk(dir):
            for filename in files:
                split = filename.find('.')

                filename1 = dir + filename
                img = Image.open(filename1)
                random_color  = DataAugmentation.randomColor(img)
                random_Gaussion = DataAugmentation.randomGaussian(img)
                # DataAugmentation.saveImage(random_color,dir_ran_color+filename[:split]+"_clor.jpg")
                DataAugmentation.saveImage(random_Gaussion,dir_ran_Gaussion+filename[:split]+"_gauss.jpg")

