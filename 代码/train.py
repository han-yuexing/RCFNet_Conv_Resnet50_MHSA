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

    # print(args)
    # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    #tb_writer = SummaryWriter(log_dir="/home/projects/CoatPred/UniModel/Low-Temperature-TC/project-3/runs")
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    train_images_path, train_images_label = read_train_data(args.train_data_path)
    val_images_path, val_images_label = read_val_data(args.val_data_path)


    img_size = 224
   
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0, 0, 0], [1, 1, 1]),
                                      transforms.Lambda(lambda x: x/255.0)]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0, 0, 0], [1, 1, 1]),
                                    transforms.Lambda(lambda x: x/255.0)])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes).to(device)

    if args.RESUME == False:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load(args.weights, map_location=device)['state_dict']

            # Delete the weight of the relevant category
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All weights except head are frozen
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    #optimizer = optim.Adadelta(pg, lr=args.lr, weight_decay=args.wd) #设置正则化项的超参数λ的值为args.wd,optim里只集成了L2正则化方法
    optimizer = optim.SGD(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_loss = float('inf') #正无穷
    start_epoch = 0

    if args.RESUME:
        path_checkpoint = "./model_weight/test/ckpt_best_100.pth"
        print("model continue train")
        checkpoint = torch.load(path_checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_schedule'])
    valarr = []
    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        #..train_loss是每个epoch的损失值(具体的算法是，如果一个epoch分为n个batch，则train_loss=n个batch的损失之和/n)
        train_loss,train_images_label_pre = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        if epoch == 106:
            torch.save(model.state_dict(), "/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/checkpoint/best_model-{}.pth".format(epoch))
            print("Saved epoch{} as new best model!".format(epoch))

             #print("[epoch {}] val_loss: {}".format(epoch, round(val_loss, 3)))
        # validate
        #..val_loss是每个epoch的损失值(具体的算法是，如果一个epoch分为n个batch，则val_loss=n个batch的损失之和/n)
        val_loss,label,pred = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        valarr.append(val_loss)
        print(valarr)
        with open(args.log_path, 'a') as file:
            file.write(f"epoch: {epoch}\n")
            file.write(f"val_loss: {val_loss}\n")
            
        print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))
        
        tb_writer.add_scalar("train_loss", train_loss, epoch)#随着epoch的变化，训练损失值的变化
        tb_writer.add_scalar("val_loss", val_loss, epoch)#随着epoch的变化，验证损失值的变化

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
                
                
        # print("label:",labels)  
        # print("pred:",preds)  
        with open(args.log_path, 'a') as file:

            file.write(f"label: {labels}\n")
            
            file.write(f"pred: {preds}\n")
            file.write(f"-----------------------------------\n")
        

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5000) #最好的是5000
    parser.add_argument('--batch-size', type=int, default=4) #最好结果4
    parser.add_argument('--lr', type=float, default=1e-8) #最好的结果是le-5
    parser.add_argument('--wd', type=float, default=0.00003) #最好的结果是0.00003                                                       
    parser.add_argument('--RESUME', type=bool, default=False)

    parser.add_argument('--train_data_path', type=str, default="/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/Big_dataset100/train")
    parser.add_argument('--val_data_path', type=str, default="/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/Big_dataset100/val")
    #parser.add_argument('--test_data_path', type=str, default="")
    parser.add_argument('--weights', type=str, default='', #..载入预训练权重，注意导入是什么模型就下载对应的预训练权重
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--log_path', type=str, default='/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/log.txt',help='Path to the log file')
    parser.add_argument('--log_dir', type=str, default='/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/runs',help='Path to the TensorBoard log directory')
    
    opt = parser.parse_args()

    main(opt)
