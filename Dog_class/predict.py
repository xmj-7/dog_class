import torch
import torch.nn as nn
from torchvision import transforms
from models import resnet50
import os
from Config import DataConfig
from PIL import Image
import json
import tqdm


dog_labels = {'1': 0, '10': 1, '100': 2, '101': 3, '102': 4, '103': 5, '104': 6, '105': 7, '106': 8, '107': 9,
          '108': 10, '109': 11, '11': 12, '110': 13, '111': 14, '112': 15, '113': 16, '114': 17, '115': 18,
          '116': 19, '117': 20, '118': 21, '119': 22, '12': 23, '120': 24, '121': 25, '122': 26, '123': 27,
          '124': 28, '125': 29, '126': 30, '127': 31, '128': 32, '129': 33, '13': 34, '130': 35, '14': 36,
          '15': 37, '16': 38, '17': 39, '18': 40, '19': 41, '2': 42, '20': 43, '21': 44, '22': 45, '23': 46,
          '24': 47, '25': 48, '26': 49, '27': 50, '28': 51, '29': 52, '3': 53, '30': 54, '31': 55, '32': 56,
          '33': 57, '34': 58, '35': 59, '36': 60, '37': 61, '38': 62, '39': 63, '4': 64, '40': 65, '41': 66,
          '42': 67, '43': 68, '44': 69, '45': 70, '46': 71, '47': 72, '48': 73, '49': 74, '5': 75, '50': 76,
          '51': 77, '52': 78, '53': 79, '54': 80, '55': 81, '56': 82, '57': 83, '58': 84, '59': 85, '6': 86,
          '60': 87, '61': 88, '62': 89, '63': 90, '64': 91, '65': 92, '66': 93, '67': 94, '68': 95, '69': 96,
          '7': 97, '70': 98, '71': 99, '72': 100, '73': 101, '74': 102, '75': 103, '76': 104, '77': 105, '78': 106,
          '79': 107, '8': 108, '80': 109, '81': 110, '82': 111, '83': 112, '84': 113, '85': 114, '86': 115, '87': 116,
          '88': 117, '89': 118, '9': 119, '90': 120, '91': 121, '92': 122, '93': 123, '94': 124, '95': 125, '96': 126,
          '97': 127, '98': 128, '99': 129}

class dog_class(object):
    def __init__(self,modelName,modelPth):
        self.modelName = modelName
        self.modelPth = modelPth
        self.input_size = 224

        self.transform = transforms.Compose([
            transforms.Resize([self.input_size,self.input_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])

        self.model = resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,130)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)


        weight = torch.load(modelPth)
        state_dict = {}
        for k, v in weight.items():
            state_dict[k] = v
        print('successfully load ' + str(len(state_dict.keys())) + ' keys')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.result ={}
    def ImgProcess(self,img):
        img = self.transform(img)
        return img

    def MakeSubmitDict(self,img_name,img):
        img = img.unsqueeze(0)
        list_labels = []
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            output = self.model(img)
            res = torch.topk(output,5)[1].cpu().numpy().tolist()
            # print("output.size",output.size())
            # print("output",output)
            # print("预测的labels",res)
            for l1 in res:
                for l2 in l1:
                    list_labels.append(int(list(dog_labels.keys())[list(dog_labels.values()).index(l2)]))
            self.result[img_name] = list_labels




if __name__=="__main__":
    model_name = DataConfig.backbone
    model_path = "C:\\Users\\xmj\\PycharmProjects\\dog_class\\ckpt\\res50\\res50_best.pth"

    dc = dog_class(model_name,model_path)

    for img_name in tqdm.tqdm(os.listdir(DataConfig.TEST_A_Dir)):
        img_path = os.path.join(DataConfig.TEST_A_Dir,img_name)
        img = Image.open(img_path).convert('RGB')
        img = dc.ImgProcess(img)
        dc.MakeSubmitDict(img_name,img)
        # break


    with open("result.json", "w") as f:
        json.dump(dc.result, f)
        print("写入文件完成...")
