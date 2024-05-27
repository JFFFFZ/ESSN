import torch
import numpy as np
from einops import rearrange
import torch.nn as nn
import data_raw
import scipy.io as sio
from torch.utils.data.dataset import Dataset


def get_config(Params):
    # 配置参数
    dataloader_config = data_raw.Dataloader(Params)
    train_config = data_raw.TrainConfig(Params)

    return dataloader_config, train_config

class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]:
            self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels

# 获取的输入
def get_net_inputs(Params, pos_num=1):
    # 如果性能不佳  参考 sstn中 loadData 中 的fit_transform  切记
    # 配置参数
    dataloader_config, train_config = get_config(Params)

    # 加载数据
    print('\n... ... loading {} origin data ... ...'.format(dataloader_config.data_sign))
    data_whole_ori, labels_whole = dataloader_config.load_data_raw()
    num_classes = dataloader_config.num_classes
    print('Data Origin Shape: ', data_whole_ori.shape)     # H W C  (610, 340, 103)

    # # 获取分支2 mycnn 的输入
    # data_whole_add = data_raw.get_img_LoG(dataloader_config.data_sign, data_whole_ori, data_raw.kernel_LoG)
    # print("Data after Edge enhance Shape: ", data_whole_add.shape)

    # PCA
    print('\n... ... PCA tranformation ... ...')  # 2024.04.13 输入由data_whole_ori_ln 修改为data_whole_ori
    data_whole_pca = data_raw.applyPCA(data_whole_ori, dataloader_config.pca_size)  #  内含  标准化
    print('Data shape after PCA: ', data_whole_pca.shape)  # H W pca_size (610, 340, 50)
    # sio.savemat('../res_recorder/%s/%s_pca.mat' % (dataloader_config.data_sign, dataloader_config.data_sign), {'%s_pca' % dataloader_config.data_sign: data_whole_pca})
    #
    # data_add_pca = data_raw.applyPCA(data_whole_add, dataloader_config.pca_size)  #  内含  标准化
    # print('Data add shape after PCA: ', data_add_pca.shape)  # H W pca_size (610, 340, 50)
    # sio.savemat('../res_recorder/%s/%s_add_pca.mat' % (dataloader_config.data_sign, dataloader_config.data_sign), {'%s_add_pca' % dataloader_config.data_sign: data_add_pca})

    if dataloader_config.generate_pos2d == "True":
        # 随机 按比率 获取训练集坐标
        pos2d_train, pos2d_test, class_train_num = dataloader_config.get_pos2d_trainandtest(labels_whole, train_config.train_ratio, num_classes)
        print("Generating Pos2D")
    else:
        pos2d_train, pos2d_test, class_train_num = dataloader_config.load_pos2d(pos_num)
        print("Loading Pos2D")

    # 获取patch并划分训练集和测试集
    if train_config.train_ratio < 1:  # split by percent  # sub_1 和 sub_2 的标签是相同的
        x_train_sub_1, y_train_sub_1 = dataloader_config.get_patch_from_pos2d(data_whole_pca, labels_whole, pos2d_train, dataloader_config.patch_size)
        x_test_sub_1, y_test_sub_1 = dataloader_config.get_patch_from_pos2d(data_whole_pca, labels_whole, pos2d_test, dataloader_config.patch_size)
        print("sub1：after percent splite patch: x_train_sub_1.shape, x_test_sub_1.shape", x_train_sub_1.shape, x_test_sub_1.shape)
        # sio.savemat('../res_recorder/%s/%s_patch_trte.mat' % (dataloader_config.data_sign, dataloader_config.data_sign), {'%s_tr' % dataloader_config.data_sign: x_train_sub_1, "%s_te" % dataloader_config.data_sign:x_test_sub_1})

        # x_train_sub_2, y_train_sub_2 = dataloader_config.get_patch_from_pos2d(data_add_pca, labels_whole, pos2d_train, dataloader_config.patch_size)
        # x_test_sub_2, y_test_sub_2 = dataloader_config.get_patch_from_pos2d(data_add_pca, labels_whole, pos2d_test, dataloader_config.patch_size)
        # sio.savemat('../res_recorder/%s/%s_add_patch_trte.mat' % (dataloader_config.data_sign, dataloader_config.data_sign),
        #             {'%s_add_tr' % dataloader_config.data_sign: x_train_sub_2, "%s_add_te" % dataloader_config.data_sign:x_test_sub_2})
        # print("sub1：after percent splite patch: x_train_sub_1.shape, x_test_sub_1.shape", x_train_sub_2.shape, x_test_sub_2.shape)

    if train_config.use_val == "True":    # 若使用验证集
        x_val_sub_1, x_test_sub_1, y_val_sub_1, y_test_1 = dataloader_config.split_data(x_test_sub_1, y_test_sub_1, train_config.val_ratio)
        print("sub1: after percent splite patch: x_val_sub_1.shape, x_test_sub_1.shape", x_val_sub_1.shape, x_test_sub_1.shape)
    else:
        x_val_sub_1 = None

    print('\n... ... create trainset and testset ... ...')
    train_hyper_sub_1 = HyperData((np.transpose(x_train_sub_1, (0, 3, 1, 2)).astype("float32"), y_train_sub_1))   # 输入 data类型  N C P P  label 类型 n*1
    test_hyper_sub_1 = HyperData((np.transpose(x_test_sub_1, (0, 3, 1, 2)).astype("float32"), y_test_sub_1))

    if train_config.use_val == "True":
        val_hyper_sub_1 = HyperData((np.transpose(x_val_sub_1, (0, 3, 1, 2)).astype("float32"), y_val_sub_1))
    else:
        val_hyper_sub_1 = None
        # val_hyper_sub_2 = None
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader_sub_1 = torch.utils.data.DataLoader(train_hyper_sub_1, batch_size=dataloader_config.batch_size, shuffle=False, **kwargs) # 不用打乱  因为坐标是随机产生的本身就是乱的
    test_loader_sub_1 = torch.utils.data.DataLoader(test_hyper_sub_1, batch_size=dataloader_config.batch_size, shuffle=False, **kwargs)
    val_loader_sub_1 = torch.utils.data.DataLoader(val_hyper_sub_1, batch_size=dataloader_config.batch_size, shuffle=False, **kwargs)


    return train_loader_sub_1, test_loader_sub_1, val_loader_sub_1, data_whole_ori, labels_whole, pos2d_train, pos2d_test, class_train_num, dataloader_config, train_config



def get_net_inputs_SSFTT(Params):

    # 配置参数
    dataloader_config, train_config = get_config(Params)

    # 加载数据
    print('\n... ... loading origin data ... ...')
    data_whole_ori, data_whole_add, labels_whole = dataloader_config.load_data_raw()
    print('Data Origin Shape: ', data_whole_ori.shape)     # H W C  (610, 340, 103)
    print('Data Add Shape: ', data_whole_add.shape)  # H W C  (610, 340, 103)

    # 转换 类型 与 维度
    data_whole_ori = rearrange(torch.tensor(data_whole_ori.astype(float), dtype=torch.float32), "H W C->C H W")
    data_whole_add = rearrange(torch.tensor(data_whole_add.astype(float), dtype=torch.float32), "H W C->C H W")

    # 2D 卷积
    print('\n... ... conv2d of data  ... ...')
    conv2d_pre = nn.Conv2d(in_channels=dataloader_config.channel_size,
                           out_channels=dataloader_config.channel_size,
                           kernel_size=5,
                           stride=5,
                           padding=1,
                           groups=1)
    data_whole_ori_conv2d = conv2d_pre(data_whole_ori)
    data_whole_add_conv2d = conv2d_pre(data_whole_add)
    print("Origin data shape after Conv2d", data_whole_ori_conv2d.shape)
    print("Added data shape after Conv2d", data_whole_add_conv2d.shape)

    # 3D卷积
    print('\n... ... conv3d of data ... ...')
    conv3d_pre = nn.Conv3d(in_channels=1,
                           out_channels=1,
                           kernel_size=3, # (3,3,3)
                           stride=1,
                           padding=1,
                           groups=1)
    data_whole_ori_conv3d = conv3d_pre(data_whole_ori.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    data_whole_add_conv3d = conv3d_pre(data_whole_add.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    print("Origin shape after Conv3d：", data_whole_ori_conv3d.shape) # C H W
    print("Added shape after Conv3d：", data_whole_ori_conv3d.shape)  # C H W

    # # 2d,3d 拼接
    # data_ori_cat = torch.cat((data_whole_ori_conv2d, data_whole_ori_conv3d), dim=0) # 2C H W
    # data_add_cat = torch.cat((data_whole_add_conv2d, data_whole_add_conv3d), dim=0) # 2C H W
    #
    # data_whole_cat = torch.cat((data_ori_cat,data_add_cat),dim=0)
    # print("Data shape after Conv2d,Conv3d cat：", data_whole_cat.shape)   # 4C H W

    # 相加
    data_whole_conv_all = data_whole_ori_conv2d + data_whole_ori_conv3d + data_whole_add_conv2d + data_whole_add_conv3d
    print("Data shape after Conv2d,Conv3d add：", data_whole_conv_all.shape, data_whole_conv_all.dtype)  # C H W

    #LN
    print('\n... ... LayerNorm  the data of cat... ...')
    # data_whole = data_raw.my_ln(rearrange(data_whole_cat, "C H W-> H W C"))
    # data_whole = rearrange(data_whole_cat, "C H W-> H W C")
    data_whole = data_raw.my_ln(rearrange(data_whole_conv_all, "C H W->H W C"))
    print("Data shape after ln:", data_whole.shape)  # H W C

    # PCA
    print('\n... ... PCA tranformation ... ...')
    # data_whole = data_raw.applyPCA(data_whole.detach().numpy(), dataloader_config.pca_size)   #   H W  pca_size
    data_whole = data_raw.applyPCA(data_whole.numpy(), dataloader_config.pca_size)   #   H W  pca_size
    print('Data shape after PCA: ', data_whole.shape)  # H W pca_size (610, 340, 50)
    return data_whole


if __name__ == "__main__":
    data_sign = "KSC"
    params = data_raw.get_param(data_sign)
    get_net_inputs(params)
