import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from utils.utils import get_classes

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
#   annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0
# -------------------------------------------------------------------#
#   必须要修改，用于生成2007_train.txt、2007_val.txt的目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#   仅在annotation_mode为0和2的时候有效
# -------------------------------------------------------------------#
classes_path = 'model_data/LYC_third_filter_classes.txt'
# --------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
#   仅在annotation_mode为0和1的时候有效
# --------------------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
# -------------------------------------------------------#
data_path = 'E:/postgraduate_data/detection'
LYC_third_filter_sets = [('LYC_third_filter', 'train'), ('LYC_third_filter', 'val')]

classes, _ = get_classes(classes_path)

# -------------------------------------------------------#
#   统计目标数量
# -------------------------------------------------------#
photo_nums = np.zeros(len(LYC_third_filter_sets))
nums = np.zeros(len(classes))


def convert_annotation(xml_path, image_id, list_file):
    in_file = open('%s/%s.xml' % (xml_path, image_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

        nums[classes.index(cls)] = nums[classes.index(cls)] + 1


if __name__ == "__main__":
    trainval_percent = 0.7
    train_percent = 0.9
    xmlfilepath = os.path.join(data_path, 'LYC_third_Annotations')
    saveBasePath = os.path.join(data_path, 'LYC_third_filter_ImageSets')
    random.seed(0)

    # print("Generate txt in ImageSets.")
    # temp_xml = os.listdir(xmlfilepath)
    # total_xml = []
    # for xml in temp_xml:
    #     if xml.endswith(".xml") and xml.split('_')[0] in ['001', '021', '025', '033', '045']:
    #         total_xml.append(xml)
    #
    # num = len(total_xml)
    # print(num)
    # tv = int(num * trainval_percent)
    # tr = int(tv * train_percent)
    # trainval = random.sample(range(num), tv)
    # train = random.sample(trainval, tr)
    #
    # print("train and val size", tv)
    # print("train size", tr)
    # ftrainval = open(os.path.join(saveBasePath, 'trainval_ori.txt'), 'w')
    # ftest = open(os.path.join(saveBasePath, 'test_ori.txt'), 'w')
    # ftrain = open(os.path.join(saveBasePath, 'train_ori.txt'), 'w')
    # fval = open(os.path.join(saveBasePath, 'val_ori.txt'), 'w')
    #
    # for i in range(num):
    #     name = total_xml[i][:-4] + '\n'
    #     if i in trainval:
    #         ftrainval.write(name)
    #         if i in train:
    #             ftrain.write(name)
    #         else:
    #             fval.write(name)
    #     else:
    #         ftest.write(name)
    # ftrainval.close()
    # ftrain.close()
    # fval.close()
    # ftest.close()
    # print("Generate txt in ImageSets done.")

    print("Generate LYC_third_filter_train.txt and LYC_third_filter_val.txt for train.")
    type_index = 0
    for flag, image_set in LYC_third_filter_sets:
        image_ids = open(os.path.join(data_path, '%s_ImageSets/%s_ori.txt' % (flag, image_set)),
                         encoding='utf-8').read().strip().split()
        list_file = open('%s/%s_%s1.txt' % (saveBasePath, flag, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/LYC_third_JPEGImages/%s.png' % (os.path.abspath(data_path), image_id))

            convert_annotation(xmlfilepath, image_id, list_file)
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print("Generate 2007_train.txt and 2007_val.txt for train done.")


    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()


    str_nums = [str(int(x)) for x in nums]
    tableData = [
        classes, str_nums
    ]
    colWidths = [0] * len(tableData)
    len1 = 0
    for i in range(len(tableData)):
        for j in range(len(tableData[i])):
            if len(tableData[i][j]) > colWidths[i]:
                colWidths[i] = len(tableData[i][j])
    printTable(tableData, colWidths)

    if photo_nums[0] <= 500:
        print("训练集数量小于500，属于较小的数据量，请注意设置较大的训练世代（Epoch）以满足足够的梯度下降次数（Step）。")

    if np.sum(nums) == 0:
        print(
            "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
        print(
            "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
        print(
            "在数据集中并未获得任何目标，请注意修改classes_path对应自己的数据集，并且保证标签名字正确，否则训练将会没有任何效果！")
        print("（重要的事情说三遍）。")
