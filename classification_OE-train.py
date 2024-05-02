import os
import json
import torch
import cv2
import torch.nn as nn
import glob
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.quantization import QuantStub, DeQuantStub
from timm import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class cfg:
    gpu = 0
    train_base = '/mnt/ssd4tb/bogoni/Datasets/training/4'
    valid_base = '/mnt/ssd4tb/bogoni/Datasets/valid/4'
    epochs = 10
   
    num_classes = 7
    class_list = ['00', 'a3', 'a4', 'b1', 'b6', 'b7', 'b8']
    #오이
    #정상 00 오이노균병 a3 오이흰가루병 a4
    #냉해피해 b1 다량원소결핍(N) b6 다량원소결핍(P) b7 다량원소결핍(K) b8
    batch_size = 32
    input_size = (256, 256) # for ResNet50
    #input_size = (224, 224) # for simple ViT & MAE - ViT

    init_lr = 0.0001
    
    patience = 5
    
    save_name = '4_best'

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(ResNet50, self).__init__()
        self.base_model = resnet50(pretrained=pretrained)
        self.base_model.fc = nn.Identity()
        self.mid1 = nn.Linear(2048, 1024)
        self.mid2 = nn.Linear(1024, 512)
        self.mid3 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(256, num_classes)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.base_model(x)
        x = nn.ReLU()(self.mid1(x))
        x = nn.ReLU()(self.mid2(x))
        x = nn.ReLU()(self.mid3(x))
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        
        return out1, out2

class ViT_MAE(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224.mae', pretrained=False):
        super(ViT_MAE, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

class ViT(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=False):
        super(ViT, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
def cnt_correct(y_true, y_pred):
    top_N, top_class = y_pred.topk(1, dim=-1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.sum(equals.type(torch.FloatTensor)).item()

def train_epoch(model, train_loader, criterion, optimizer, epoch, gpu):
    
    model.train()

    pbar = tqdm(train_loader, position=0, leave=True)
    
    len_data = 0
    loss_sum = 0
    correct_sum = 0
    
    for (image, label) in pbar:
        
        if torch.cuda.is_available() and cfg.gpu is not None:
            image = image.cuda(gpu, non_blocking=True)
            label = label.cuda(gpu, non_blocking=True)
    

        optimizer.zero_grad()
        pred1,pred2 = model(image)

        label2 = torch.where(label > 0, torch.tensor(1,device=label.device), torch.tensor(0, device=label.device))
        loss1 = criterion(pred1, label)
        loss2 = criterion(pred2, label2)#label 
        loss = loss1+loss2
        
        loss_sum += loss.item()
       
        loss.backward()
        optimizer.step()
       
        len_data += image.shape[0]
        correct_sum += cnt_correct(label, pred1)
       
        pbar.set_postfix(Epoch=f'{epoch+1}/{cfg.epochs}', Train_Loss=f'{loss_sum/len_data:.5f}', Train_Acc=f'{correct_sum/len_data:.5f}')
       
    return loss_sum/len_data, correct_sum/len_data

def valid_epoch(model, valid_loader, criterion, epoch, gpu):
  
    model.eval()
    
    pbar = tqdm(valid_loader, position=0, leave=True)

    len_data = 0
    loss_sum = 0
    correct_sum = 0

    for (image, label) in pbar:
        if torch.cuda.is_available() and cfg.gpu is not None:
            image = image.cuda(gpu, non_blocking=True)
            label = label.cuda(gpu, non_blocking=True)

        with torch.no_grad():
            pred1, pred2 = model(image)
        
        label2 = torch.where(label > 0, torch.tensor(1,device=label.device), torch.tensor(0,device=label.device))
        loss1 = criterion(pred1, label)
        loss2 = criterion(pred2, label2)
        loss = loss1 +loss2
        loss_sum += loss.item()

        len_data += image.shape[0]
        correct_sum += cnt_correct(label, pred1)

        pbar.set_postfix(Epoch=f'{epoch+1}/{cfg.epochs}', Valid_Loss=f'{loss_sum/len_data:.5f}', Valid_Acc=f'{correct_sum/len_data:.5f}')

    return loss_sum/len_data, correct_sum/len_data

class CustomDataset(Dataset):
    def __init__(self, path, cfg, mode='valid'):
        self.image_list = []
        self.label_list = []

        self.path = path
        self.mode = mode

        for root, dirs, files in os.walk(self.path):
            if 'image' in root:
                for file in glob.glob(root + '/*.jpg') + glob.glob(root + '/*.png'):
                    
                    label_path = os.path.join(os.path.dirname(os.path.dirname(file)), 'label', os.path.basename(file))
                    label_path = os.path.splitext(label_path)[0] + '.json'

                    if os.path.isfile(label_path):
                        self.label_list.append(label_path)
                        self.image_list.append(file)
                    
        self.class_list = cfg.class_list
        self.input_size = cfg.input_size
                    
    def __getitem__(self, idx):

        img_path = self.image_list[idx]
        img = cv2.imread(img_path)

        lbl_path = self.label_list[idx]

        with open(lbl_path, 'r') as json_file:
            json_dict = json.load(json_file)
            disease = json_dict["annotations"]["disease"]

        lbl = torch.tensor(self.class_list.index(disease), dtype=torch.long)

        img = self.transform(img)

        return img.float(), lbl
        
    def transform(self, img):
 
        img = cv2.resize(img, dsize=self.input_size)

        if self.mode == 'train':
            inversion_augs = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5)
            ]
            transformed = A.Compose(inversion_augs)
            img = transformed(image=img)['image']

            color_augs = [
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.01,
                    p=0.5
                )
            ]
            transformed = A.Compose(color_augs)
            img = transformed(image=img)['image']

        base_augs = [
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0),
            ToTensorV2()
        ]
        transformed = A.Compose(base_augs)
        img = transformed(image=img)['image']
        return img

    def __len__(self):
        return len(self.image_list)

def main():

    model = ResNet50(num_classes=cfg.num_classes, pretrained=True)
    #model = ViT(num_classes=cfg.num_classes, model_name='vit_base_patch16_224', pretrained=True)
    #model = ViT_MAE(num_classes=cfg.num_classes, model_name='vit_base_patch16_224.mae', pretrained=True)

    device = torch.device("cuda")
    print(torch.cuda.is_available())
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.init_lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
 
    train_dataset = CustomDataset(cfg.train_base, cfg, 'train')
    valid_dataset = CustomDataset(cfg.valid_base, cfg, 'valid')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=10)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=10)

    early_stopping = EarlyStopping(patience=cfg.patience, verbose=True, path=f'{cfg.save_name}.pt')

    min_valid_loss = np.inf
    max_valid_acc = 0.

    for epoch in range(cfg.epochs):

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, cfg.gpu)
        valid_loss, valid_acc = valid_epoch(model, valid_loader, criterion, epoch, cfg.gpu)
        scheduler.step()

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('Early stopping...')
            break

        if (valid_loss < min_valid_loss) or (valid_acc > max_valid_acc):
            if (valid_loss < min_valid_loss):
                min_valid_loss = valid_loss
                torch.save({'model_state_dict': model.state_dict()}, f'{cfg.save_name}_min_loss.pt')

            if (valid_acc > max_valid_acc):
                max_valid_acc = valid_acc
                torch.save({'model_state_dict': model.state_dict()}, f'{cfg.save_name}_max_acc.pt')

    torch.save({'model_state_dict': model.state_dict()}, f'{cfg.save_name}_final.pt')
    print('Finished training')

if __name__ == "__main__":
    main()