import os
import csv
import random
import shutil
from Config import DataConfig
# 修改文件夹名称
class DataMade():
    def ChangeFolder(self):#重新构建文件，生成一个train
        try:
            f = open('dog_data.csv','w',newline='')
            csv_writer = csv.writer(f)
            # 构建表头
            csv_writer.writerow(['name','label'])

            path = DataConfig.data_dir+'\\train';
            i=0
            for item in os.listdir(path):
                if os.path.isdir(os.path.join(path,item))==True:
                    i+=1
                    string = ''
                    string = item.split('-')
                    if string.__len__()!=1:#重命名
                        os.rename(os.path.join(path,item),os.path.join(path,str(int(string[1][-3:]))))
                        p =os.path.join(path,item)
                        print(p)
                        _list = []
                        label1 = string[2]
                        label2 = string[1][-3:]
                        _list.append(label1)
                        _list.append(label2)
                        csv_writer.writerow(_list)
            print("All folders have processed:",i)
            f.close()
        except:
            print("Process failed")


    def MakeVal(self):
        if os.path.isdir(os.path.join(DataConfig.data_dir, 'val')) == False:
            os.mkdir(DataConfig.data_dir+'\\val')
            print("made a val folder")
            val_path = os.path.join(DataConfig.data_dir,'val')
            train_path = DataConfig.data_dir+'\\train'
            all1 = 0
            all2 = 0
            i=1
            for item in os.listdir(train_path):
                train_item = os.path.join(train_path,item)
                val_item =os.path.join(val_path,item)
                if os.path.isdir(val_item) == False:
                    os.mkdir(val_item)

                rate = 0.3#移动的比例
                ordinaryLen = len(os.listdir(train_item))
                pickNumber = int(ordinaryLen*rate)
                all1+=ordinaryLen
                all2+=pickNumber

                val_img = random.sample(os.listdir(train_item),pickNumber)#随机取一部分出来
                for img_name in val_img:
                    shutil.move(os.path.join(train_item,img_name),os.path.join(val_item,img_name))
                print("第{}个文件夹移动结束，原有文件数量{}个，移动了{}个".format(i,ordinaryLen,pickNumber))
                i+=1
            print(all1)
            print(all2)

data = DataMade()
data.ChangeFolder()
data.MakeVal()