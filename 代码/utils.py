import os
import sys
import json
import pickle
import random
import math
from PIL import Image
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def read_train_data(root: str):
    random.seed(1)
    #...断言的一般用法是assert condition，如果condition为false，那么raise一个AssertionError出来
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    #...listdir(path)即以列表的形式展示出path路径下所有文件的名称
    #...list=[x for x in a if x%2==0],即list的意思是从a中挑选x%2==0的元素组成的列表
    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    #...enumerate(sequence,[start=0])用于将一个可遍历的数据对象(如列表、元组或字符串)sequence组合为一个索引序列，同时列出数据和数据下标(这里就是
    #...[(0,'Dermatosis_Category_1'),(1,'Dermatosis_Category_2'),...])，start是规定第一个元素下标起始位置，默认0。一般用在 for 循环当中.
    #...生成一个字典，同时将列表category里的“键值”顺序颠倒过来
    class_indices = dict((k, k) for v, k in enumerate(category))
    #...json.dumps()是把python对象(如字典)转换成json对象的一个过程，生成的是字符串，即将对象转换为字符串，indent参数决定缩进几个空格
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)

    with open('/home/projects/CoatPred/UniModel/Low-Temperature-TC/project/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        #..os.path.splitext(file)的作用是将文件名和文件格式分开，返回的是一个元组，[-1]即直接拿到最后一个元素即文件格式
        #..一个个遍历data/train/Dermatosis_Category_1路径下的所有文件，对于每一个文件如果它的格式被支持，就将其和之前的路径组合起来放到列表images里
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cls]
        #..将images中里每一个具体的如图(路径)拿出来放到训练集中，并保存它的标签，实现获得数据和标签分离且对应
        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(float(image_class))

    print("{} images for training.".format(len(train_images_path)))

    return train_images_path, train_images_label


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, k) for v, k in enumerate(category))

    val_images_path = []
    val_images_label = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cls in category:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(root, cls, i) for i in os.listdir(cls_path)
                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cls]

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(float(image_class))

    print("{} images for validation.".format(len(val_images_path)))

    return val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)

def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    #loss_function = torch.nn.CrossEntropyLoss()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    train_images_label_pre = []
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        
        pred = torch.squeeze(pred)
        train_images_label_pre.append(pred)
        #...交叉熵的输入第一个位置的输入应该是在每个label下的概率, 而不是对应的label,所以这里是pred而非pred_classes
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        #..返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, lr: {:.5f}".format(
            epoch,
            #..计算的是每个step的平均loss
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]
        )

        #..判断loss是否有界
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
    

    return accu_loss.item() / (step + 1),train_images_label_pre

class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        #..tuple()函数用于将列表、区间（range）等转换为元组
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.MSELoss()

    model.eval()


    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    predresult = []
    labels_r = []
    
    for step, data in enumerate(data_loader):
        images, labels = data
        labels_r.append(labels)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred = torch.squeeze(pred)#删除为1的维度
        predresult.append(pred)

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1)
        )

    return accu_loss.item() / (step + 1),labels_r,predresult


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}
    # print("start!")
    # print("parameter_group_varss:",parameter_group_vars)
    # print("parameter_group_names",parameter_group_names)

    #..requires_grad=True即允许更新参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

        # print("parameter_group_vars:",parameter_group_vars)
        # print("parameter_group_names",parameter_group_names)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
