import torch
from models.iresnet.iresnet import load_model
import os
# def load_model(path):
#     model = IResNet(IBasicBlock, layers=[3, 13, 30, 3])
#     model.load_state_dict(torch.load(path))
#     return model
    # ckpt = torch.load(path, map_location=device)

def extract_features(x):
    '''
        x is a batch of images with shape (batch_size, n_chanels, height, width)
    '''
    root = os.getcwd()
    path = os.path.join(root, 'models', 'iresnet', 'iresnet_weight.pt')
    model = load_model(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model.forward(x)


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(10, 3, 112, 112)
    x = x.to(device)
    print(extract_features(x).shape)
    # print(os.getcwd())
    # path = '/home/tandattran772/persional_project/FaceID/models/iresnet/iresnet_weight.pt'

    # model = load_model(path)
