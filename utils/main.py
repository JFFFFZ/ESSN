from utils.workflow import train_test
datasets_name = ["KSC", "Indian", "PaviaU", ]  # 数据库
MODE = ["ESSN", "cmp_2D_CNN", "cmp_3D_CNN", "cmp_SSRN", "cmp_SSTN", "cmp_HybridSN", "cmp_SSFTT", "cmp_vit_conv", "cmp_CTMXier"]  # 模型库

if __name__ == '__main__':
    for mode in MODE:
        for data_sign in datasets_name:
            for i in range(1, 11):
                train_test(data_sign=data_sign, mode=mode, pos_num=i)
