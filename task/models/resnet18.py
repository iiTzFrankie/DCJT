import torch
import torch.nn as nn
import torchvision.models as models

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        resnet = models.resnet18(True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, 7)
        
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def Model():
    model = resnet18()       
    model = torch.nn.DataParallel(model).cuda()
    ##加载用Affectnet pretrain过的模型
    #checkpoint = torch.load("/home/frank/project/CCFER202206/checkpoints_Oct22/pretrainOnAff/[10-08]-[17-07]-model_best.pth")
    #model.load_state_dict(checkpoint['state_dict'],strict=True)
    #print('载入Pretrain模型')
    return model