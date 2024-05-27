import os
import torch
import json
import random
import numpy as np
import scipy.io as sio
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

LABELS_PATH_PREFIX = "../../datasets/origin"
DATA_PATH_PREFIX = '../../datasets/origin'

CONFIG_PATH_PREFIX = '../paramconfig'
#
# kernel_LoG = torch.tensor([[0, 1, 1, 2, 2, 2, 1, 1, 0],
#                             [1, 2, 4, 5, 5, 5, 4, 2, 1],
#                             [1, 4, 5, 3, 0, 3, 5, 4, 1],
#                             [2, 5, 3, -12, -24, -12, 3, 5, 2],
#                             [2, 5, 0, -24, -40, -24, 0, 5, 2],
#                             [2, 5, 3, -12, -24, -12, 3, 5, 2],
#                             [1, 4, 5, 3, 0, 3, 4, 4, 1],
#                             [1, 2, 4, 5, 5, 5, 4, 2, 1],
#                             [0, 1, 1, 2, 2, 2, 1, 1, 0]],requires_grad=False)

# kernel_LoG = torch.tensor([[0, 1, 0],
#                            [1, -4, 1],
#                            [0, 1, 0]], requires_grad=False)

kernel_LoG = torch.tensor([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]], requires_grad=False)

def get_param(data_sign):
    # 若数据不在以下四个数据集中则报错
    assert data_sign in ['Indian', 'PaviaU', 'Salinas', 'KSC', 'Houston', 'MUUFL']
    config_path_prefix = CONFIG_PATH_PREFIX
    path_param = '%s/%s.json' % (config_path_prefix, data_sign)
    print("Path of Dataset：", '%s' % path_param)
    with open(path_param, 'r') as fin:
        params = json.loads(fin.read()) # https://blog.csdn.net/xrinosvip/article/details/82019844  建议read(size) 可以防止爆内存

    return params

def applyPCA(data_data, num_components=30):
    # 输入 HWC  输出  HWC
    data_data_pca = np.reshape(data_data, (-1, data_data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    data_data_pca = pca.fit_transform(data_data_pca)
    data_data_pca = StandardScaler().fit_transform(data_data_pca)
    data_data_pca = np.reshape(data_data_pca, (data_data.shape[0], data_data.shape[1], num_components))
    return data_data_pca


def padding_zeros(data_data, margin):
    # 输入 H W C  输出 （H+2margin）（W+2margin）C
    h_origin, w_origin, c_origin = data_data.shape
    h_new, w_new, c_new = h_origin + 2 * margin, w_origin + 2 * margin, c_origin
    data_data_new = np.zeros((h_new, w_new, c_new))
    # 中心原始数据填充
    data_data_new[margin:h_new - margin, margin:w_new - margin, :] = data_data
    return data_data_new


# 输入 H W C   输出  （H+2*margin） （W+2*margin） C
def padding_same(data_data, margin):
    margin = int(margin)
    h_origin = int(data_data.shape[0])
    w_origin = int(data_data.shape[1])
    c_origin = int(data_data.shape[2])
    h_new, w_new, c_new = int(h_origin + 2 * margin), int(w_origin + 2 * margin), c_origin
    data_data_new = np.zeros((h_new, w_new, c_new))
    # 中心原始数据填充
    data_data_new[margin:int(h_new - margin), margin:int(w_new - margin), :] = data_data
    # 边缘相同填充
    # 左右
    for i in range(margin, h_new-margin):
        for j in range(0, margin):
            data_data_new[i][j][:] = data_data_new[i][margin][:]        # 左边
            data_data_new[i][w_new-j-1][:] = data_data_new[i][w_new-margin-1][:]  # 右边
    # 上下以及四个角落
    for i in range(0, margin):
        for j in range(0, w_new):
            data_data_new[i][j][:] = data_data_new[margin][j][:]    # 上边
            data_data_new[h_new-i-1][j][:] = data_data_new[h_new-margin-1][j][:]  # 下边

    return data_data_new

def patch_padding_same(data_data, margin):
    # 输入 C H W  输出 C H W
    margin = int(margin)
    c_origin = int(data_data.shape[0])
    h_origin = int(data_data.shape[1])
    w_origin = int(data_data.shape[2])
    h_new, w_new, c_new = int(h_origin + 2 * margin), int(w_origin + 2 * margin), c_origin
    data_data_new = torch.zeros((c_new, h_new, w_new))
    # 中心原始数据填充
    data_data_new[:, margin:int(h_new - margin), margin:int(w_new - margin)] = torch.tensor(data_data, dtype=torch.float32)
    # 边缘相同填充
    # 左右
    for i in range(margin, h_new-margin):
        for j in range(0, margin):
            data_data_new[:][i][j] = data_data_new[:][i][margin]        # 左边
            data_data_new[:][i][w_new-j-1] = data_data_new[:][i][w_new-margin-1]  # 右边
    # 上下以及四个角落
    for i in range(0, margin):
        for j in range(0, w_new):
            data_data_new[:][i][j] = data_data_new[:][margin][j]    # 上边
            data_data_new[:][h_new-i-1][j][:] = data_data_new[:][h_new-margin-1][j]  # 下边

    return data_data_new


def my_ln(data_data):
    # 输入H W C  array 格式  输出 H W C  array
    ln = nn.LayerNorm([data_data.shape[0], data_data.shape[1]], elementwise_affine=False)   # H W 层面进行归一化

    return rearrange(ln(torch.tensor(rearrange(data_data, 'h w c->c h w'), dtype=torch.float32)),'c h w -> h w c')


def get_img_LoG(data_sign, data_data_origin, kernel_LoG):
    # 要求输入 H W C array   输出 H W C  array    不改变形状
    # 边缘同值pad
    margin = int((kernel_LoG.shape[0]-1)/2)
    data_data_padded = padding_same(data_data_origin, margin)

    # 转换格式
    img_tensor = torch.tensor(data_data_padded, dtype=torch.float32)
    img_tensor = rearrange(img_tensor, 'h w c -> c h w')    # C H W

    # 卷积获取边缘
    img_LoG = np.zeros((img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]))
    for c in range(0, img_tensor.shape[0]):
        img_LoG[c, :, :] = F.conv2d(img_tensor[c, :, :].unsqueeze(0).unsqueeze(0), kernel_LoG.unsqueeze(0).unsqueeze(0), stride=1, padding=margin, groups=1)  # group 为分组卷积组数

    img_LoG_chw = torch.FloatTensor(img_LoG)
    img_LoG = rearrange(img_LoG, 'c h w->h w c')
    img_LoG_central = img_LoG[margin:int(img_LoG.shape[0]-margin), margin:int(img_LoG.shape[1]-margin), :]
    sio.savemat('../res_recorder/%s/%s_LoG_Conv2d.mat' % (data_sign, data_sign), {'%s_LoG_conv' % data_sign: img_LoG_central})

    # 将原图 和 经过LoG的图 相加 获取新图     H W C
    img_LoG_add = np.zeros((data_data_origin.shape[0], data_data_origin.shape[1], data_data_origin.shape[2]))
    for c in range(0, data_data_origin.shape[2]):
        img_LoG_add[:, :, c] = img_LoG_central[:, :, c] + data_data_origin[:, :, c]
    print("img_LoG_add.shape = ", img_LoG_add.shape)
    sio.savemat("../res_recorder/%s/%s_added.mat" % (data_sign,data_sign), {"%s_added" % data_sign: img_LoG_add})

    return img_LoG_add  # H W C


def get_patch_LoG(data_sign, patch_origin, kernel_LoG=kernel_LoG.cuda()):
    # 要求输入 B C H W  tensor   输出 B C H W   tensor    不改变形状
    # 边缘同值pad
    b, c, h, w = patch_origin.shape
    margin = int((kernel_LoG.shape[0]-1)/2)
    dw_conv = nn.Conv2d(in_channels=patch_origin.shape[1],
                        out_channels=patch_origin.shape[1],
                        kernel_size=kernel_LoG.shape[1],
                        stride=1,
                        padding=margin,
                        groups=patch_origin.shape[1],
                        padding_mode="replicate",
                        bias=False
                        ).cuda()  # groups=C，实现深度卷积
    # 将自定义的卷积核赋值给深度卷积层
    dw_kernel = kernel_LoG.unsqueeze(0).unsqueeze(1).repeat(patch_origin.shape[1], 1, 1, 1) # 50, 3 , 3
    dw_conv.weight.data.copy_(dw_kernel)
    patch_LoG = dw_conv(patch_origin)  # group 为分组卷积组数
    # patch_LoG_add = patch_LoG + patch_origin
    patch_LoG_add = patch_LoG

    return patch_LoG_add  # B H W C


class TrainConfig(object):
    def __init__(self, params):
        self.train_param_config = params['train']

        # 数据集路径相关
        # 路径分为 前缀 + 文件名
        # self.cls_mode = self.train_param_config.get("cls_mode", "pooling")  # 分类方式cls_cat或者 pooling
        self.train_ratio = self.train_param_config.get("train_ratio", 0.20)
        self.use_val = self.train_param_config.get('use_val', "True")
        self.val_ratio = self.train_param_config.get("val_ratio", 0.10)
        self.epochs = self.train_param_config.get('epochs', 200)
        self.lr = self.train_param_config.get('lr', 0.05)
        self.att_deep = self.train_param_config.get('att_deep', 3)
        self.weight_decay = self.train_param_config.get('weight_decay', 1e-4)
        self.momentum = self.train_param_config.get('momentum', 0.9)


def random_unison(a, b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]


class Dataloader(object):
    def __init__(self, params):
        # 划分
        self.data_whole = None
        self.labels_whole = None

        # 参数配置
        self.data_param_config = params['data']

        # 数据集路径相关
        # 路径分为 前缀 + 文件名
        self.data_path_prefix = self.data_param_config.get('data_path_prefix', DATA_PATH_PREFIX)
        self.data_sign = self.data_param_config.get('data_sign', 'Indian')
        self.split_data_way = self.data_param_config.get('split_data_way', "use_pixels_patch")
        self.generate_pos2d = self.data_param_config.get('generate_pos2d', "True")

        # 类别个数
        self.num_classes = self.data_param_config.get('num_classes', 16)

        # [B, C, H, W] 配置
        self.batch_size = self.data_param_config.get('batch_size', 1)
        self.channel_size = self.data_param_config.get('channel_size', 200)
        self.high_size = self.data_param_config.get('high_size', 145)
        self.wide_size = self.data_param_config.get('wide_size', 145)

        # vit patch 大小设置  超参数
        self.patch_size = self.data_param_config.get('patch_size', 13)
        self.embedding_dim = self.data_param_config.get('embedding_dim', 128)
        self.pca_size = self.data_param_config.get('pca_size', 50)
        self.patch_dim = self.patch_size * self.patch_size * self.pca_size
        self.dim_heads = self.data_param_config.get('dim_heads', 64)

        self.patches_num = self.high_size

        # 加载未处理过的原始数据
    def load_data_raw(self):
        # 若数据不在以下四个数据集中则报错
        assert self.data_sign in ['Indian', 'PaviaU', 'Salinas', 'KSC', 'Houston', 'MUUFL']

        data_path_pre_ori = self.data_path_prefix
        if self.data_sign == 'Indian':
            data_data_ori = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
            data_labels = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Indian_pines_gt.mat'))['indian_pines_gt']
            print("Path of Dataset：", '%s/%s' % (data_path_pre_ori, 'Indian_pines_corrected.mat'))
        elif self.data_sign == 'Salinas':
            data_data_ori = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Salinas_corrected.mat'))['salinas_corrected']
            data_labels = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Salinas_gt.mat'))['salinas_gt']
            print("Path of Dataset：", '%s/%s' % (data_path_pre_ori, 'Salinas_corrected.mat'))
        elif self.data_sign == 'PaviaU':
            data_data_ori = sio.loadmat('%s/%s' % (data_path_pre_ori, 'PaviaU.mat'))['paviaU']
            data_labels = sio.loadmat('%s/%s' % (data_path_pre_ori, 'PaviaU_gt.mat'))['paviaU_gt']
            print("Path of Dataset：", '%s/%s' % (data_path_pre_ori, 'PaviaU.mat'))
        elif self.data_sign == 'KSC':
            data_data_ori = sio.loadmat('%s/%s' % (data_path_pre_ori, 'KSC.mat'))['KSC']
            data_labels = sio.loadmat('%s/%s' % (data_path_pre_ori, 'KSC_gt.mat'))['KSC_gt']
            print("Path of Dataset：", '%s/%s' % (data_path_pre_ori, 'KSC.mat'))
        elif self.data_sign == 'Salinas':
            data_data_ori = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Salinas_corrected.mat'))['salinas_corrected']
            data_labels = sio.loadmat('%s/%s' % (data_path_pre_ori, 'Salinas_gt.mat'))['salinas_gt']
            print("Path of Dataset：", '%s/%s' % (data_path_pre_ori, 'Salinas_corrected.mat'))
        else:
            print("Datasets don't exist！！！")
            exit()

        self.data_whole_ori = data_data_ori
        self.labels_whole = data_labels

        return data_data_ori, data_labels

    def load_pos2d(self, pos_num=1):
        # 若数据不在以下四个数据集中则报错
        assert self.data_sign in ['Indian', 'PaviaU', 'KSC']

        data_path_pre = "../pos2d/%s" % self.data_sign
        file_path = data_path_pre + "/%s_%s" % (self.data_sign, pos_num)
        if self.data_sign == 'Indian':
            pos2d_file = sio.loadmat(file_path)
            pos2d_train = pos2d_file['pos2d_train']
            pos2d_test = pos2d_file['pos2d_test']
            class_train_num = pos2d_file["class_train_num"]
            print("Path of Pos2D：", file_path)
        elif self.data_sign == 'PaviaU':
            pos2d_file = sio.loadmat(file_path)
            pos2d_train = pos2d_file['pos2d_train']
            pos2d_test = pos2d_file['pos2d_test']
            class_train_num = pos2d_file["class_train_num"]
            print("Path of Pos2D：", file_path)
        elif self.data_sign == 'KSC':
            pos2d_file = sio.loadmat(file_path)
            pos2d_train = pos2d_file['pos2d_train']
            pos2d_test = pos2d_file['pos2d_test']
            class_train_num = pos2d_file["class_train_num"]
            print("Path of Pos2D：", file_path)
        else:
            print("Datasets don't exist！！！")
            exit()

        for i in range(len(class_train_num)):
            print("The number of class_train is :", class_train_num[i])
        print("Number of Train set：", len(pos2d_train))

        return pos2d_train, pos2d_test, class_train_num


    def get_pos2d_trainandtest(self, data_labels, train_ratio, cls_num, seed=None):
        '''
        随机产生训练集和测试集patch的二维中心坐标
        :param data_labels: 原始标签
        :param train_ratio: 训练集比率
        :param seed: 随机种子，默认不固定
        :return: 不重复的训练集二维坐标，测试集二维坐标 且 标签不为0
        '''
        img_h, img_w = data_labels.shape
        # 获取类别为0的点的 二维坐标 和个数
        pos2d_label0 = []
        for i in range(img_h):
            for j in range(img_w):
                if data_labels[i, j] == 0:
                    pos2d_label0.append((i, j))
        print("The num of Label 0：", len(pos2d_label0))

        train_test_pos2d_num = img_h * img_w - len(pos2d_label0)

        pos2d_train_temp = set()  # 使用集合来存储坐标，确保不重复
        cls_train_num = [[] for i in range(cls_num)]
        cls_train_num_save = []     # 保存训练样本中各个类别的
        if seed:
            print("暂不支持随机种子")
        else:
            while len(pos2d_train_temp) < int(train_test_pos2d_num * train_ratio):
                X_pos_train = random.randint(0, img_w - 1)  # [0, img_w -1]
                Y_pos_train = random.randint(0, img_h - 1)
                if data_labels[Y_pos_train, X_pos_train] != 0:
                    pos2d_train_temp.add((Y_pos_train, X_pos_train))
                    if (Y_pos_train, X_pos_train) not in cls_train_num[data_labels[(Y_pos_train, X_pos_train)]-1]:
                        cls_train_num[data_labels[Y_pos_train, X_pos_train]-1].append((Y_pos_train, X_pos_train))

            pos2d_train = list(pos2d_train_temp)
            # 再次打乱坐标
            random.shuffle(pos2d_train)
            print("The num of train sample:", len(pos2d_train))

        num_train_temp = 0

        for i in range(cls_num):
            num_train_temp += len(cls_train_num[i])
            print("The number of class_%s is :" % (i+1), len(cls_train_num[i]))
            cls_train_num_save.append(len(cls_train_num[i]))
        print("total number of train:", num_train_temp)

        # 获取测试集的二维坐标（不能与训练集重复）,且无 0标签
        pos2d_test = []
        for i in range(img_h):
            for j in range(img_w):
                if (i, j) not in pos2d_train:
                    if data_labels[i, j] != 0:
                        pos2d_test.append((i, j))
        print("The num of test sample:", len(pos2d_test))

        if (len(pos2d_label0) + len(pos2d_train) + len(pos2d_test)) == (img_w * img_h):
            print("Correctly generate training and test sets!")

        return pos2d_train, pos2d_test, cls_train_num_save

    def get_patch_from_pos2d(self, image, labels_whole, pos2d, patch_size, padding_style="same", removeZeroLabels=True):
        # 输入  patch  H W C
        # 输出  patchesData   N C Patch_size Patch_size
        # 输出 patchesLabels  N * 1
        margin = int(patch_size / 2)
        if padding_style == "same":
            # 边缘填充  从 HW->(H+2margin)(W+2margin)
            data_padded = padding_same(image, margin)  # (H+2margin)(W+2margin)C
        else:
            data_padded = padding_zeros(image, margin)  # (H+2margin)(W+2margin)C

            # 划分 patches
        patchesData = np.zeros((len(pos2d), patch_size, patch_size, image.shape[2]))  # 训练集个数/测试集个数  win*win*c 大小的patch
        patchesLabels = np.zeros((len(pos2d)))  # 训练集个数/测试集个数  每个数据patch对应一个像素点的标签
        patchIndex = 0

        for (x, y) in pos2d:
            patch = data_padded[x:(x + 2 * margin + 1), y:(y + 2 * margin + 1), :]   # 1: 10 是从1开始 到 9  不包括10
            # patch = data_samepadded[x:(x + 2 * margin + 1), y:(y + 2 * margin + 1)]  # 从前到后  所以第三维度可以省略
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = labels_whole[x, y]
            patchIndex = patchIndex + 1

        if removeZeroLabels:  # 除去 无标签类
            patchesData = patchesData[patchesLabels > 0, :, :, :]
            patchesLabels = patchesLabels[patchesLabels > 0]
            patchesLabels -= 1
            return patchesData, patchesLabels.astype("int")


    def createImageCubes(self, data_whole, labels_whole, patchsize=13, removeZeroLabels=True):
        '''
        逐像素划分Patch   所以一张 H W的图像 能划分成 H*W 个patch
        :param data_whole: H W C  全部数据部分  输入  numpy  用 .detach().numpy()转换成 numpy数据类型
        :param labels_whole: H W  全部标签部分
        :param patchsize:
        :param removeZeroLabels: true时 移除标签为0（即无类别的部分）
        :return: H*W 个 patch  ->  H W p p  输出 numpy
        '''
        margin = int((patchsize - 1) / 2)
        data_zeropadded = padding_zeros(data_whole, margin=margin)
        # split patches
        patchesData = np.zeros((data_whole.shape[0] * data_whole.shape[1], patchsize, patchsize, data_whole.shape[2]))  # H*W个 win*win*c 大小的patch
        patchesLabels = np.zeros((data_whole.shape[0] * data_whole.shape[1]))   # H W 大小  每个数据patch对应一个像素点的标签
        patchIndex = 0
        for r in range(margin, data_zeropadded.shape[0] - margin):
            for c in range(margin, data_zeropadded.shape[1] - margin):
                patch = data_zeropadded[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = labels_whole[r - margin, c - margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:    # 除去 无标签类
            patchesData = patchesData[patchesLabels > 0, :, :, :]
            patchesLabels = patchesLabels[patchesLabels > 0]
            patchesLabels -= 1
        return patchesData, patchesLabels.astype("int")



    def splitTrainTestSet(self, data_data_cube, data_labels_cube, train_ratio, randomState=345):
        # 输入 numpy 输出 numpy
        data_train_patched, data_test_patched, labels_train_patched, labels_test_patched = train_test_split(data_data_cube,
                                                                                                            data_labels_cube,
                                                                                                            train_size=train_ratio,
                                                                                                            random_state=randomState,
                                                                                                            stratify=data_labels_cube)
        return data_train_patched, data_test_patched, labels_train_patched, labels_test_patched


    def split_data(self, pixels, labels, percent, splitdset="custom", rand_state=69):
        splitdset = "sklearn"
        if splitdset == "sklearn":
            return train_test_split(pixels, labels, test_size=(1 - percent), stratify=labels, random_state=rand_state)
        elif splitdset == "custom":
            pixels_number = np.unique(labels, return_counts=1)[1]
            train_set_size = [int(np.ceil(a * percent)) for a in pixels_number]
            tr_size = int(sum(train_set_size))
            te_size = int(sum(pixels_number)) - int(sum(train_set_size))
            sizetr = np.array([tr_size] + list(pixels.shape)[1:])
            sizete = np.array([te_size] + list(pixels.shape)[1:])
            train_x = np.empty((sizetr))
            train_y = np.empty((tr_size))
            test_x = np.empty((sizete))
            test_y = np.empty((te_size))
            trcont = 0
            tecont = 0
            for cl in np.unique(labels):
                pixels_cl = pixels[labels == cl]
                labels_cl = labels[labels == cl]
                pixels_cl, labels_cl = random_unison(pixels_cl, labels_cl, rstate=rand_state)
                for cont, (a, b) in enumerate(zip(pixels_cl, labels_cl)):
                    if cont < train_set_size[cl]:
                        train_x[trcont, :, :, :] = a
                        train_y[trcont] = b
                        trcont += 1
                    else:
                        test_x[tecont, :, :, :] = a
                        test_y[tecont] = b
                        tecont += 1
            train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
            return train_x, test_x, train_y, test_y
