class DataConfig(object):
    backbone = 'res50'#模型类别
    num_classes = 130 #分类数量
    loss = 'CrossEntropyLoss'#focal_loss/CrossEntropyLoss
    #
    input_size = 224#输入大小
    train_batch_size = 30 # batch size
    val_batch_size = 30#验证集的batch_size
    test_batch_size = 1#用于预测
    optimizer = 'sgd'#优化器采用sgd优化器
    lr = 1e-3  # adam 0.00001
    MOMENTUM = 0.9
    device = "cuda"  # cuda  or cpu设备是gpu,还是cpu
    gpu_id = [0,1]
    num_workers = 4  # how many workers for loading data
    max_epoch = 20
    lr_decay_epoch = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay 当损失值上升时，学习率减小
    weight_decay = 5e-4
    val_interval = 1
    print_interval = 100#时间间隔
    save_interval = 2
    min_save_epoch=2
    #
    data_dir = 'C:\\Users\\xmj\\PycharmProjects\\dog_class\\Data\\dog_img'
    train_dir = 'C:\\Users\\xmj\\PycharmProjects\\dog_class\\Data\\dog_img\\train'
    val_dir = 'C:\\Users\\xmj\\PycharmProjects\\dog_class\\Data\\dog_img\\val'
    #
    checkpoints_dir = 'ckpt/'

