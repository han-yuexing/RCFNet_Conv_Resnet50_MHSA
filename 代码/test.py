import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import MyDataSet
from Model_Conv_Resnet50_MHSA_scse import main_model as create_model
from utils import read_train_data, read_val_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate
import numpy as np
import random
import torchvision.models as models


def main(args):

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(20)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    test_images_path, test_images_label = read_val_data(args.test_data_path)


    img_size = 224
   
    data_transform = {
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0, 0, 0], [1, 1, 1]),
                                    transforms.Lambda(lambda x: x/255.0)])}


    test_dataset = MyDataSet(images_path=test_images_path,
                            images_class=test_images_label,
                            transform=data_transform["test"])
    

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
   

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=test_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes).to(device)

    #加载参数
    model.load_state_dict(torch.load("/home/projects/CoatPred/UniModel/Low-Temperature-TC/project-4/set/checkpoint/best_model-106.pth"))
    test_loss,label,pred = evaluate(model=model,
                                    data_loader=test_loader,
                                    device=device,
                                    epoch=1)
        
        
    labels = []
    for i in range(len(label)): #将里面的每个tensor拿出来，每个tensor里有多个值
        #拿到当前的tensor后将里面的每个数值取出来放到列表中取
        for a in range(len(label[i])):
                labels.append(np.round(label[i][a].cpu().detach().item(),3))
        
    preds = []
    for j in range(len(pred)): #将里面的每个tensor拿出来，每个tensor里有多个值
        #拿到当前的tensor后将里面的每个数值取出来放到列表中取
        for b in range(len(pred[j])):
            preds.append(np.round(pred[j][b].cpu().detach().item(),3))
                
                
    print("label:",labels)  
    print("pred:",preds)  
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
   
    parser.add_argument('--batch-size', type=int, default=32) 
   
    parser.add_argument('--test_data_path', type=str, default="/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/Big_dataset100/test")
   
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    
    
    opt = parser.parse_args()

    main(opt)
