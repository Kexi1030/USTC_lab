#coding=utf-8
from PIL import Image
import os
from tqdm import tqdm
import time
import glob
import platform
from skimage.feature import hog
import numpy as np
import joblib
from sklearn.svm import LinearSVC
import shutil


def fixed_size(filePath,savePath):
    """
    按照固定尺寸处理图片
    """
    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)


def changeSize():
    filePath = './featherdataset'
    destPath = './featherdataset/imgSizeChanged'
    #for eachfilePath
    print("正在进行图像初始化...")
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1]=='g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))
    print('图像初始化已完成')

def datasetmade(target_path,filenames):
    for file in filenames:
        # 所有的子文件夹
        sonDir = "featherdataset/" + file
        print("正在对{}数据集进行初始化".format(file))
        time.sleep(0.5)
        # 遍历子文件夹中所有的文件
        for root, dirs, files in os.walk(sonDir):
            #如果文件夹中有文件
            if len(files) > 0:
                for f in tqdm(files):
                    newDir = sonDir + '/' + f
                    # 将文件移动到新文件夹中
                    im = Image.open(newDir)
                    out = im.resize((image_width, image_height), Image.ANTIALIAS)
                    # out = out.crop((0,0,600,900))
                    # out.show()
                    if not os.path.exists(target_path):
                        os.mkdir(target_path)
                        # print("数据集合并文件夹新建完成")
                    out.save(target_path+'/'+f)
                    # 将文件改名改成训练集中带\1格式
                    # print(target_path+'/'+f)
                    # print(target_path + '/' +file+'-'+ f)
                    os.rename(target_path+'/'+f,target_path + '/' +file+'-'+ f)    
                    time.sleep(0.01)
            else:
                print(sonDir + "文件夹是空的")
        time.sleep(1)


def txtProcess():
    # 对train.txt和val.txt操作 使其与newdataset中的文件名相同
    txtChange = ["featherdataset/train.txt","featherdataset/val.txt"]
    for eachtxtChange in txtChange:
        with open(eachtxtChange,'r+') as f:
            print("---------正在对{}进行初始化---------".format(eachtxtChange.split('/')[-1]))
            time.sleep(0.5)
            newf = open("featherdataset/new"+eachtxtChange.split('/')[-1] ,'w')
            Alllines = f.readlines()
            for eachstringline in tqdm(Alllines):
                eachstringline = eachstringline.replace('/', '-')
                newf.write(eachstringline)
                time.sleep(0.005)
            newf.close()

def makedirs():
    os.mkdir("train")
    os.mkdir("test")

# 获得图片列表
def get_image_list(filePath, nameList):
    print('正在进行图像读取... ',filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath,name))
        img_list.append(temp.copy())
        temp.close()
    return img_list

# 提取特征并保存
def get_feat(image_list, name_list, label_list, savePath):
    i = 0
    for image in image_list:
        try:
            # 如果是灰度图片  把3改为-1
            image = np.reshape(image, (image_height, image_width, 3))
        except:
            print('发送了异常，图片大小size不满足要求：',name_list[i])
            continue
        gray = rgb2gray(image) / 255.0
        fd = hog(gray, orientations=12,block_norm='L1', pixels_per_cell=[13, 13],
                 cells_per_block=[4, 4], visualize=False,transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]]))
        fd_name = name_list[i] + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
        i += 1
    print("features are extracted and saved.")


# 变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# 获得图片名称与对应的类别
def get_name_label(file_path):
    print("正在从文件载入...",file_path)
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            #一般是name label  三部分，所以至少长度为3  所以可以通过这个忽略空白行
            if len(line)>=3: 
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n','').replace('\r',''))
                if not str(label_list[-1]).isdigit():
                    print("label必须为数字，得到的是：",label_list[-1],"程序终止，请检查文件")
                    exit(1)
    return name_list, label_list


# 提取特征
def extra_feat():
    train_name, train_label = get_name_label(train_label_path)
    test_name, test_label = get_name_label(test_label_path)

    train_image = get_image_list(train_image_path, train_name)
    test_image = get_image_list(test_image_path, test_name)
    get_feat(train_image, train_name, train_label, train_feat_path)
    get_feat(test_image, test_name, test_label, test_feat_path)


# 创建存放特征的文件夹
def mkdir():
    """
    
    Returns
    -------
    None.

    """
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


# 训练和测试
def train_and_test():
    features = []
    labels = []
    correct_number = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])
    print("SVM Training......")
    clf = LinearSVC()
    clf.fit(features, labels)
    # 下面的代码是保存模型的
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')
    # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
    # clf = joblib.load(model_path+'model')
    print("Model has saved......")
    # exit()
    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
        if int(result[0]) == int(data_test[-1]):
            correct_number += 1
    write_to_txt(result_list)
    rate = float(correct_number) / total
    #print('准确率为： %f' % rate)


def write_to_txt(list):
    with open('result.txt', 'w') as f:
        f.writelines(list)
    print('Result has been saved in result.txt')


def trainandtestdatasetmade():
    txtlist = ["featherdataset/newtrain.txt","featherdataset/newval.txt"]
    for eachtxtlist in txtlist:
        with open(eachtxtlist,"r") as f:
            allLines = f.readlines()
            load = "featherdataset_"+eachtxtlist.split('/')[-1].split('.')[0]
            os.mkdir(load)
            print("正在创建{} 集".format(eachtxtlist.split('new')[-1]).split('.')[0])
            time.sleep(0.5)
            for eachline in tqdm(allLines):
                fileload = seek_files('featherdataset/newdataset',eachline.split(' ')[0])
                shutil.copy(fileload,load)
                time.sleep(0.05)
            shutil.copy(eachtxtlist,load)


def seek_files(id1, name):
    """根据输入的文件名称查找对应的文件夹有无改文件，有则输出文件地址"""
    for root, dirs, files in os.walk(id1):
        if name in files:
            # 当层文件内有该文件，输出文件地址
            return root+'/'+name

def accandrecall():
    '''
    计算准确率和召回率
    '''
    array_1 = np.zeros((5,5))
    alltest = 0
    with open("result.txt",'r') as f:
        allLines = f.readlines()
        alltest = len(allLines)
        for eachline in allLines:
            old = int(eachline.split("-")[0])
            if old == 56:
                old = 5
            pre = int(eachline.split(" ")[1])
            if pre == 56:
                pre = 5
            array_1[old-1][pre-1] += 1
    print("混淆矩阵为：")
    print(array_1)
    
    Acc = 0
    Recall = 0
    
    for i in range(5):
        acc = array_1[i][i] / sum([array_1[i][j] for j in range(5)])
        if array_1[i][i] != 0:
            recall = array_1[i][i] / sum([array_1[j][i] for j in range(5)])
        else:
            recall = 0
        
        Acc += acc * (sum([array_1[i][j] for j in range(5)])/alltest)
        Recall += recall * (sum([array_1[i][j] for j in range(5)])/alltest)
            
    print("准确率为：{0}".format(Acc))
    print("召回率为 {0}".format(Recall))


        

if __name__ == '__main__':
    
    t0 = time.time()
    makedirs()
    
    label_map = {1: '1',
             2: '2',
             3: '3',
             4: '4',
             56:'56'
             }
    
    train_feat_path = 'train/'
    test_feat_path = 'test/'
    model_path = 'model/'
    
    
	# 设定图片的统一尺寸
    image_width = 128
    image_height = 100
    
    targetpath =r"featherdataset/newdataset"
    # 原文件夹
    old_path = r"featherdataset"
    # 查看原文件夹下所有的子文件夹
    filenames = os.listdir(old_path)
    
    datasetmade(targetpath,filenames=filenames)
    print("数据集生成完成")
    
    txtProcess()
    trainandtestdatasetmade()
    
    
    # 训练集图片的位置
    train_image_path = 'featherdataset_newtrain'
    # 测试集图片的位置
    test_image_path = 'featherdataset_newval'
    
    # 训练集标签的位置
    train_label_path = train_image_path+'/newtrain.txt'
    # 测试集标签的位置
    test_label_path = test_image_path+'/newval.txt'
    

    shutil.rmtree(train_feat_path)
    shutil.rmtree(test_feat_path)
    mkdir()
    extra_feat()  # 获取特征并保存在文件夹
    print("特征提取成功")
    
    train_and_test()
    
    accandrecall()
    
    t1 = time.time()
    print('耗时: %f' % (t1 - t0))