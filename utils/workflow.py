import os
import time
import torch
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch.nn.parallel
import data_raw
import spectral
import data_prepare
from ESSNet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(y_pred, y_test, name):
    if name == 'Indian':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'Salinas':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PaviaU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'KSC':
        target_names = ['Scrub', 'Willow swamp', 'CP hammock', 'Slash pine', 'Oak/broadleaf', 'Hardwood', 'Swamp',
                        'Graminoid marsh', 'Spartina marsh', 'Cattail marsh', 'Salt marsh',  'MUd flats', 'Water']

    classification = classification_report(y_test, y_pred, digits=6, target_names=target_names)

    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)

    return classification, confusion, oa, aa, kappa, list(np.round(np.array(list(each_acc)) * 100, 2))


def train_sub(trainloader, model, criterion, optimizer, epoch):
    model.train()
    accs = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)  # output.shape [N, cls_num]
        loss = criterion(outputs, targets)  # targets [N, 1]   loss 1*1 tensor
        losses[batch_idx] = loss.item()
        accs[batch_idx] = accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.average(losses), np.average(accs)


def mytest_sub(testloader, model, criterion, epoch):
    model.eval()
    accs = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    test_predict_all = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            test_predict_all = [*test_predict_all, *np.argmax(outputs.detach().cpu().numpy().astype(int), axis=1)]
            losses[batch_idx] = criterion(outputs, targets).item()
            accs[batch_idx] = accuracy(outputs.data, targets.data, topk=(1,))[0].item()

    return np.average(losses), np.average(accs), test_predict_all


def predict_sub(testloader, model):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        with torch.no_grad():
            [predicted.append(a) for a in model(inputs).data.cpu().numpy()]  # 每次加一个  batch_size,cls  存入list中
    return np.array(predicted)


def train_test(data_sign, mode="ESSN", pos_num=1):
    assert data_sign in ['Indian', 'PaviaU', 'Salinas', 'KSC', 'Houston', 'MUUFL']
    params = data_raw.get_param(data_sign)
    train_loader_sub_1, test_loader_sub_1, val_loader_sub_1, data_whole_ori, labels_whole, pos2d_train, pos2d_test, class_train_num, dataloader_config, train_config = data_prepare.get_net_inputs(params, pos_num)
    labels_gt = labels_whole
    lr = train_config.lr

    if mode == "ESSN":  # ori
        train_loader, test_loader, val_loader = train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
        del train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
        model_name = "ESSN"
        model = ESSNet(pca_component=dataloader_config.pca_size, num_classes=dataloader_config.num_classes, patch_size=dataloader_config.patch_size, att_deep=train_config.att_deep)
    # elif mode == 'cmp_SSRN':
    #     train_loader, test_loader, val_loader = train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
    #     del train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
    #     model_name = "cmp_SSRN"
    #     model = SSRN(class_num=dataloader_config.num_classes)
    # elif mode == 'cmp_SSTN':
    #     train_loader, test_loader, val_loader = train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
    #     del train_loader_sub_1, test_loader_sub_1, val_loader_sub_1
    #     model_name = "cmp_SSTN"
    #     model = SSTN(num_classes=dataloader_config.num_classes, pca_size=dataloader_config.pca_size)
    else:
        raise ValueError("mode is error! please correct it. ")

    predict_map = labels_whole  # 保存最终预测结果

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=train_config.momentum, weight_decay=train_config.weight_decay, nesterov=True)

    best_acc = -1
    time_start = time.time()  # 记录开始时间

    print("Start Train...")
    for epoch in range(train_config.epochs):
        train_loss, train_acc = train_sub(train_loader, model, criterion, optimizer, epoch)

        if train_config.use_val == "True":
            test_loss, test_acc, test_predict_no_use = mytest_sub(val_loader, model, criterion, epoch)
        else:
            test_loss, test_acc, test_predict_no_use = mytest_sub(test_loader, model, criterion, epoch)
        print("\nEPOCH: %s/%s" % (epoch + 1, train_config.epochs), "\nTRAIN LOSS", str(format(train_loss, '.6f')),
              "\tTRAIN ACCURACY", str(format(train_acc, '.6f')))

        # save model
        if test_acc > best_acc:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, "../res_recorder/%s/%s_%s_%s_best_model.pth.tar" % (data_sign, model_name, data_sign, str(pos_num)))
            best_acc = test_acc

    time_end = time.time()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print('Run time：', time_sum)

    checkpoint = torch.load("../res_recorder/%s/%s_%s_%s_best_model.pth.tar" % (data_sign, model_name, data_sign, str(pos_num)))
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    test_loss, test_acc, test_predict = mytest_sub(test_loader, model, criterion, train_config.epochs)  # 预测结果为one-hot 编码
    print("FINAL Result:", "\nTOTAL LOSS", str(format(test_loss, '.6f')), "\tTOTAL ACCURACY", str(format(test_acc, '.6f')))

    predict_total = predict_sub(test_loader, model)
    classification, confusion, oa_test, aa_test, kappa_test, test_acc_list = reports(np.argmax(predict_total, axis=1), np.array(test_loader.dataset.__labels__()), data_sign)

    print("Final Result")
    print(classification)
    print(data_sign, mode, model_name)
    print("oa aa kappa ", oa_test, aa_test, kappa_test)
    print("Acc of each class：", test_acc_list)

    results_path = '../res_recorder/%s/%s_%s_results.txt' % (data_sign, model_name, data_sign)

    with open(results_path, "a") as file:
        str_results = '\n\n======================' \
                      + "\n mode=" + str(mode) \
                      + "\n model name=" + str(model_name) \
                      + "\t train data path=" + str(data_raw.DATA_PATH_PREFIX) \
                      + "\n learning rate=" + str(train_config.lr) \
                      + "\t train ratio=" + str(train_config.train_ratio) \
                      + "\t patch size=" + str(dataloader_config.patch_size) \
                      + "\t time sum=" + str(time_sum) \
                      + "\t epochs=" + str(train_config.epochs) \
                      + "\tval_epochs=" + str(start_epoch) \
                      + "\tval best acc=" + str(best_acc) \
                      + "\ntrain num per class" + str(class_train_num) \
                      + '\nacc per class:' + str(test_acc_list)\
                      + "\nTOTAL OA=" + str(oa_test) \
                      + "\tTOTAL AA=" + str(aa_test) \
                      + '\tTOTAL kpp=' + str(kappa_test)
        file.write(str_results)

    # 创建文件夹  保存图片
    file_path = '../res_recorder/%s/%s' % (data_sign, mode)  # ../res_recorder/KSC/sub_1/
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    file_path = file_path + '/%s%s_ps%s_tr%s_lr%s_pos%s_oa_%s' % (model_name, str(start_epoch), str(dataloader_config.patch_size), str(train_config.train_ratio), str(train_config.lr), str(pos_num), str(format(oa_test*100, '.2f')))  # ../res_recorder/KSC/fusion/98.98
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    train_labels = np.zeros((labels_whole.shape[0], labels_whole.shape[1]))  # 保存训练集坐标
    predict_error_map = np.zeros((labels_whole.shape[0], labels_whole.shape[1]))    # 保存 判断错误的点
    predict_error_gt = np.zeros((labels_whole.shape[0], labels_whole.shape[1]))  # 保存 判断错误的点的真实标签

    # 绘制训练集所在的位置
    for index in range(len(pos2d_train)):
        pos2d = pos2d_train[index]
        train_labels[pos2d[0], pos2d[1]] = labels_whole[pos2d[0], pos2d[1]]

    # 总预测
    for index in range(len(test_predict)):
        pos2d = pos2d_test[index]
        if (test_predict[index] + 1) != labels_whole[pos2d[0], pos2d[1]]:
            predict_error_map[pos2d[0], pos2d[1]] = test_predict[index]+1
            predict_error_gt[pos2d[0], pos2d[1]] = labels_whole[pos2d[0], pos2d[1]]
        predict_map[pos2d[0], pos2d[1]] = test_predict[index]+1

    spectral.save_rgb(file_path + "/ps%s_oa%s_trainsets_gt.jpg" % (str(dataloader_config.patch_size), str(format(oa_test * 100, '.2f'))), train_labels.astype(int), colors=spectral.spy_colors)   # 训练集位置图

    spectral.save_rgb(file_path + "/ps%s_oa%s_predict.jpg" % (str(dataloader_config.patch_size), str(format(oa_test*100, '.2f'))), predict_map.astype(int), colors=spectral.spy_colors)  # 总预测图
    spectral.save_rgb(file_path + "/ps%s_oa%s_predict_error.jpg" % (str(dataloader_config.patch_size), str(format(oa_test*100, '.2f'))), predict_error_map.astype(int), colors=spectral.spy_colors)   # 总预测失败图
    spectral.save_rgb(file_path + "/ps%s_oa%s_predict_error_gt.jpg" % (str(dataloader_config.patch_size), str(format(oa_test*100, '.2f'))), predict_error_gt.astype(int), colors=spectral.spy_colors)    # 预测失败的地方的正确标签图


if __name__ == "__main__":
    datasets_name = ["KSC", "Indian", "PaviaU"]
    # Total_MODE = ["ESSN", "cmp_2D_CNN", "cmp_3D_CNN", "cmp_SSRN", "cmp_SSTN", "cmp_HybridSN", "cmp_SSFTT", "cmp_vit_conv", "cmp_CTMXier"]

    Current_Mode = ["ESSN"]
    for mode in Current_Mode:
        for data_sign in datasets_name:
            for i in range(1, 11):
                print("\n======================{}=============================".format(i))
                train_test(data_sign=data_sign, mode=mode, pos_num=i)
