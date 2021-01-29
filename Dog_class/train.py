import os
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms,datasets
import torch.optim as optim
import time
from models import resnet50
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from Config import DataConfig
from sklearn.metrics import accuracy_score


def train_model(model,criterion,optimizer):
    data_transform = transforms.Compose([
        transforms.Resize([DataConfig.input_size,DataConfig.input_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])
    train_dataset = datasets.ImageFolder(root=DataConfig.train_dir,transform=data_transform)
    train_loader = DataLoader(train_dataset,
                              batch_size=DataConfig.train_batch_size,
                              shuffle=True,
                              drop_last=True
                              )
    print(len(train_loader))
    print(len(train_dataset))
    all_iters = len(train_loader)
    model_name = DataConfig.backbone
    train_loss = []
    since = time.time()
    best_score = 0.0
    best_epoch = 0
    for epoch in range(1,DataConfig.max_epoch+1):
        model.train(True)
        begin_time = time.time()
        running_corrects_linear = 0
        count = 0
        for i ,data in enumerate(train_loader):
            count+=1
            inputs,labels = data
            labels = labels.type(torch.LongTensor)
            #没有GPU底下这句话不用管
            if torch.cuda.is_available():
                inputs,labels = inputs.cuda(),labels.cuda()

            out_linear = model(inputs)
            _, linear_preds = torch.max(out_linear.data,1)
            loss = criterion(out_linear,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i %DataConfig.print_interval==0 or out_linear.size()[0]<DataConfig.train_batch_size:#每训练100次输出一次，或者是训练到最后一次也要输出
                spend_time = time.time() - begin_time
                print(
                    ' Epoch:{}({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch, count, all_iters,
                        loss.item(), optimizer.param_groups[-1]['lr'],
                        spend_time / count * all_iters // 60 - spend_time // 60))
                train_loss.append(loss.item())
                running_corrects_linear +=torch.sum(linear_preds == labels.data)

        weight_score = val_model(model,criterion)
        epoch_acc_linear = running_corrects_linear.double() / all_iters / DataConfig.train_batch_size
        print('Epoch:[{}/{}] train_acc={:.3f} '.format(epoch, DataConfig.max_epoch,
                                                       epoch_acc_linear))
        model_save_dir = (DataConfig.checkpoints_dir,DataConfig.backbone)
        model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + str(epoch) + '.pth'
        best_model_out_path = model_save_dir + "/" + '{}_'.format(model_name) + 'best' + '.pth'

        #保存效果最好的那一组参数
        if weight_score>best_score:
            best_score = weight_score
            best_epoch = epoch
            torch.save(model.state_dict(),best_model_out_path)
            print("best epoch: {} best acc: {}".format(best_epoch, weight_score))

        #按间隔保存参数
        if epoch % DataConfig.save_interval == 0 and epoch>DataConfig.min_save_epoch:
            torch.save(model.state_dict(), model_out_path)



    print('Best acc: {:.3f} Best epoch:{}'.format(best_score,best_epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




@torch.no_grad()
def val_model(model,criterion):
    data_transform = transforms.Compose([
        transforms.Resize([DataConfig.input_size,DataConfig.input_size]),

        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
    ])
    val_dataset = datasets.ImageFolder(root=DataConfig.val_dir,transform=data_transform)
    val_loader = DataLoader(val_dataset,
                              batch_size=DataConfig.val_batch_size,
                              shuffle=False,
                              drop_last=True
                              )


    dset_sizes = len(val_dataset)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    pres_list = []
    labels_list = []
    for data in val_loader:
        inputs,labels = data
        labels = labels.type(torch.LongTensor)
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)
        loss = criterion(outputs,labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        pres_list+=preds.cpu().numpy().tolist()
        labels_list+=labels.data.cpu().numpy().tolist()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += 1
    val_acc = accuracy_score(labels_list, pres_list)
    print('val_size: {}  valLoss: {:.4f} valAcc: {:.4f}'.format(dset_sizes, running_loss / dset_sizes,
                                                                      val_acc))
    return val_acc

if __name__=="__main__":
    if torch.cuda.is_available():
        # opt = Config()
        torch.cuda.empty_cache()#释放缓存
        device = torch.device('cuda:0')
        criterion = torch.nn.CrossEntropyLoss().cuda()#转换为GPU的张量类型
        model_name=DataConfig.backbone#网络模型名
        print(model_name)
        model_save_dir =os.path.join(DataConfig.checkpoints_dir , model_name)
        print(model_save_dir)
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)#不存在对应文件夹时，创建一个
        model = resnet50(pretrained=True)#使用预训练末模型
        num_ftrs = model.fc.in_features
        print(num_ftrs)
        model.fc = nn.Linear(num_ftrs, DataConfig.num_classes)#修改全连接层，2048Xopt.num_class
        model.to(device)#将模型中的运算放在指定的GPU中
        # model = nn.DataParallel(model)#多GPU计算
        optimizer = optim.SGD((model.parameters()), lr=DataConfig.lr, momentum=DataConfig.MOMENTUM, weight_decay=0.0004)
        train_model(model, criterion, optimizer)#开始训练
    else:
        torch.cuda.empty_cache()#释放缓存
        criterion = torch.nn.CrossEntropyLoss()
        model_name=DataConfig.backbone#网络模型名
        print(model_name)
        model_save_dir =os.path.join(DataConfig.checkpoints_dir , model_name)
        print(model_save_dir)
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)#不存在对应文件夹时，创建一个
        model = resnet50(pretrained=True)#使用预训练末模型
        num_ftrs = model.fc.in_features
        print(num_ftrs)
        model.fc = nn.Linear(num_ftrs, DataConfig.num_classes)#修改全连接层，2048Xopt.num_class
        optimizer = optim.SGD((model.parameters()), lr=DataConfig.lr, momentum=DataConfig.MOMENTUM, weight_decay=0.0004)
        train_model(model, criterion, optimizer)#开始训练
