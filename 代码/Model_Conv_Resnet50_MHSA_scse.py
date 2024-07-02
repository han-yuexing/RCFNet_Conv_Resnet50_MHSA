import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)
'''..
patch_size是在Patch Partition模块中下采样的倍数；in_chans为输入图像的深度(RGB即为3);num_classes为分类类别数；embed_dim是H/4*W/4*C中的C；depths为每个stage当中重复Swin Transformer Block的次数；
num_heads是Swin Transformer Block当中采用的multi-self-attention-head的个数；window_size是W-MSA和SW-MSA所采用window的大小；mlp_ratio实在MLP模块中第一个全连接层将channel翻多少倍；qkv_bias是在multi-head-
self-attention中是否使用偏置；drop_rate是接在PatchEmbed后面的；attn_drop_rate对应的是ulti-head-self-attention中采用的drop；drop_path_rate对应的是在每一个Swin Transformer Block模块当中所采用drop rate;
patch_norm也是接在PatchEmbed后面的；use_checkpoint使用的话可以节省内存，这里没用上；
'''

class MHSA(nn.Module):
    def __init__(self, n_dims=512, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)

        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x): #x:torch.Size([128, 512, 2, 2])  224的尺寸是torch.Size([128, 512, 14, 14])
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1) #q:torch.Size([128, 4, 128, 4])  torch.Size([128, 4, 128, 196])
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1) #k:torch.Size([128, 4, 128, 4])
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1) #v:torch.Size([128, 4, 128, 4])

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k) #注意力分数 = 查询向量（q） · 键向量（k）得到q和k两个向量的相似性程度
        
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)

        content_position = torch.matmul(content_position, q) #将位置信息和查询q相乘，首先内容和位置的关联

        energy = content_content + content_position #内容和位置相加，即综合考虑了内容和位置的关系
        
        attention = self.softmax(energy)#进行softmax得到最后的得分

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out
    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=int(resolution[0]), height=int(resolution[1]), heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x): #torch.Size([128, 1024, 14, 14])

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out))) #在这个地方通过多头自注意力机制
        out = self.bn3(self.conv3(out)) #torch.Size([128, 2048, 1, 1])
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class main_model(nn.Module):

    def __init__(self, num_classes, in_chans=3, depths=(2,2,6,2),
                 drop_rate=0,
                 drop_path_rate=0.1,
                
                 conv_depths: list = [3, 3, 9, 3], conv_dims: list = [96, 192, 384, 768], conv_drop_path_rate: float = 0.,
                 conv_layer_scale_init_value: float = 1e-6,
                 conv_head_init_scale: float = 1., z_dim=10, nc=3, 
                 block = Bottleneck, num_blocks = [3, 4, 6, 3], resolution=(224, 224), heads=4,**kwargs):
        super().__init__()
        
        #resnet
        self.in_planes = 64
        self.resolution = list(resolution)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # for ImageNet
        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, heads=heads, mhsa=True)


        #下采样
        #下采样部分
        self.downsample1 = nn.Conv2d(256, 96, kernel_size=1, stride=1, padding=0)
        self.downsample2 = nn.Conv2d(512, 192, kernel_size=1, stride=1, padding=0)
        self.downsample3 = nn.Conv2d(1024, 384, kernel_size=1, stride=1, padding=0)
        self.downsample4 = nn.Conv2d(2048, 768, kernel_size=1, stride=1, padding=0)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.3), # All architecture deeper than ResNet-200 dropout_rate: 0.2
            nn.Linear(512 * block.expansion, num_classes)
        )

    


        ###### ConvNeXt Branch Setting ######
        self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample
        #..创建最初下采样的卷积层，其是由一个卷积层和一个LayerNorm组成的
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),#..conv_dims列表是每个stage输入特征层的维度
                                LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4 downsample #..构建stage2~stage4的Downsample，其中的卷积层输入和输出conv_dims[i], conv_dims[i+1]正好对应图纸结构
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                                nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks #..用来存储每个stage所构建的一系列block
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))] #..(conv_depth对应的是每个stage里所构建block的个数),dp_rates是一个等差数列列表，sum(conv_depths)是进行求和，也就是说之后每个block都会采用一个不同的dp_rates并且其是递增的
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4): #一共有4个stage所以需要遍历4次
            stage = nn.Sequential( #每个stage又有多个blocks，因此再一次循环
                *[Block(dim=conv_dims[i], drop_rate=dp_rates[cur + j], #..针对每个stage输入特征层的channel
                        layer_scale_init_value=conv_layer_scale_init_value)
                    for j in range(conv_depths[i])] #for j in range(conv_depths[i])即在当前这个stage中构建第几个block
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # final norm layer #..最后一个stage后的Global Avg Pooling之后的归一化层，输入的维度是conv_dims的最后一个元素(网络图中并没有画出来，似乎用不上)
        self.conv_head = nn.Linear(conv_dims[-1], num_classes) #..归一化后的全连接层(网络图里没画，似乎也用不上)
        self.apply(self._init_weights)#调用父类的这个apply方法传入指定好的初始化权重方法_init_weights，就会对当前模型的每一个子模块进行初始化
        self.conv_head.weight.data.mul_(conv_head_init_scale) #..对初始化后的conv_head这一层的权重乘以一个数conv_head_init_scale，这里默认为1，也就是没有做任何缩放
        self.conv_head.bias.data.mul_(conv_head_init_scale)#对初始化后的conv_head层的偏置乘以一个数conv_head_init_scale
        

        ###### Hierachical Feature Fusion Block Setting #######

        self.fu1 = HFF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=drop_rate / 2)
        self.fu2 = HFF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=drop_rate / 2)
        self.fu3 = HFF_block(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=drop_rate / 2)
        self.fu4 = HFF_block(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=drop_rate / 2)

        #在这定义通道注意力层
        self.se1 = senet(96)
        self.se2 = senet(192)
        self.se3 = senet(384)
        self.se4 = senet(768)
    
     #for resnet
    def _make_layer(self, block, planes, num_blocks, stride=1, heads=4, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, heads, mhsa, self.resolution))
            if stride == 2:
                self.resolution[0] /= 2
                self.resolution[1] /= 2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear): #如果当前的子模块是全连接层
            nn.init.trunc_normal_(m.weight, std=.02)#则对其权重进行初始化
            if isinstance(m, nn.Linear) and m.bias is not None: #如果当前的子模块是全连接层且其偏置不是None
                nn.init.constant_(m.bias, 0) #则将其偏置置为0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            #nn.init.constant_(m.bias, 0) 在Conv2d、Linear 和 BacthNorm2d 中设置了bias = True 才能赋值，否则需要注释bias 的赋值(这里没有设置为Ture);是卷积里的bias吗，一般定义卷积时会定义bias标志位为True或false，就是定义当前卷积是否有偏置，如果标志位为True，卷积层输出为Wx+b，如果标志位为False，卷积层输出为Wx

    def forward(self, imgs):

        ######resnet50######
        x = self.conv1(imgs)
        out = self.relu(self.bn1(self.conv1(imgs)))
        out = self.maxpool(out) # for ImageNet

        out = self.layer1(out)  #56,56,256
        x_s_1 = self.downsample1(out) #56,56,96
        out = self.layer2(out)  #torch.Size([2, 512, 28, 28])
        x_s_2 = self.downsample2(out) #需要([2, 192, 28, 28])
        out = self.layer3(out)  #torch.Size([2, 1024, 14, 14])
        x_s_3 = self.downsample3(out) #需要([2, 384, 14, 14])
        out = self.layer4(out)  #torch.Size([2, 2048, 7, 7])
        x_s_4 = self.downsample4(out) #需要([2, 768, 7, 7])
        
        ######  ConvNeXt Branch ######
        x_c = self.downsample_layers[0](imgs) #通过的是Conv2d k4,s4卷积层和后面的Layer Norm
        x_c_1 = self.stages[0](x_c) #..通过第一个stage中的3个block
        x_c = self.downsample_layers[1](x_c_1) #..通过第一个downsample_layers（即由Layer Norm和Conv2d k2,s2）组成的整体
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        ##### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None) #[2, 96, 56, 56]
        #考虑在这里加上通道注意力机制
        x_f_1 = self.se1(x_f_1)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1) #[2, 192, 28, 28]
        x_f_2 = self.se2(x_f_2)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2) #[2, 384, 14, 14]
        x_f_3 = self.se3(x_f_3)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3) #[2, 768, 7, 7]
        x_f_4 = self.se4(x_f_4)
       
        tmp = x_f_4.mean([-2, -1]) 
        x_fu = self.conv_norm(tmp) 
        x_fu = self.conv_head(x_fu)

        return x_fu

##### ConvNeXt Component #####

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    #..在pytorch官方有现成的LayerNorm方法，但是其默认是从最后一个维度开始做归一化的，在ConvNext中是对channel维度做归一化的，所以如果channel维度放在最后面那么是可以直接使用官方提供的，否则不能使用，这里直接重写了
    #..一个LayerNorm
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last": #即如果channel维度放在最后的话([batch_size,H,W,C])就直接调用官方实现的归一化方法就可以了
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first": #否则，如果Channel维度放在开始的话([batch_size,channel,H,W])则用自己些的归一化方法
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True) #channel所在的维度为1，这里就是对channel这个维度求均值
            var = (x - mean).pow(2).mean(1, keepdim=True) #方差
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv #..输入输出都是dim，即对照了结构图中不会改变数据形状；groups即分组卷积https://zhuanlan.zhihu.com/p/472069431
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers #..定义1*1的卷积层(即结构图中的Conv2d k1,s1)，代码中通过全连接层代替，因为1*1的卷积层作用和全连接层一样
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim) #..定义第二个1*1的卷积层，同样通过全连接层代替
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None  #..定义gamma，即对应的Layer scale层，对每一个通道数据进行放缩；其元素个数和输入特征层的dim是相同的，layer_scale_init_value是一个初始值
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity() #..当传入的drop_rate>0才构建DropPath方法，否则就直接来一个恒等映射

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C] #..将通道放在最后一个维度上，就可以使用归一化方法了
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x #Scale
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        tmp = self.drop_path(x) #若为Identity()则就是直接全等过来
        x = shortcut + tmp
        return x

# Hierachical Feature Fusion Block
class HFF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0., re=16):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.W = Conv(ch_int*2, ch_int, 3, bn=True, relu=True)
        self.W3 = Conv(ch_int * 3, ch_int, 3, bn=True, relu=True)
        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        
        #关于scse注意力机制部分
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch_1, ch_1 // re, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_1 // re, ch_1, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(ch_1, ch_1, 1),
            nn.Sigmoid()
        )

    def forward(self, l, g, f): #convnext网络提取局部特征 l是convnext   g是resnet

        #正对l,g,f三个变量分别用上scse注意力机制
        #针对l
        cse_l = self.cSE(l)
        re1_l = l * cse_l
        sse_l = self.sSE(l)
        re2_l = l * sse_l
        l = re1_l + re2_l
        
        #针对g
        cse_g = self.cSE(g)
        re1_g = g * cse_g
        sse_g = self.sSE(g)
        re2_g = g * sse_g
        g = re1_g + re2_g
        
        W_local = self.W_l(l)   # local feature from ConvNeXt
        W_global = self.W_g(g)   # global feature from SwinTransformer
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            tmp1 = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.W3(tmp1)#记录word
        else:
            tmp2 = torch.cat([W_local, W_global], 1)
            X_f = self.W(tmp2)#记录word

        # spatial attention for ConvNeXt branch
        l_jump = l     #对convnext的操作
        max_result, _ = torch.max(l, dim=1, keepdim=True) #作用是将96个通道中大的那个通道直接取出来了(不做任何计算操作，如果batch_size=n则是每个图像的最大的那个通道给取出来)
        avg_result = torch.mean(l, dim=1, keepdim=True) #在96个通道上的对应元素加起来除以96即求得每个位置的平均值，最后结果是1个通道
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)  #torch.Size([4, 1, 56, 56])  #记录word
        
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g  #对resnet的操作
        max_result=self.maxpool(g)
        avg_result=self.avgpool(g)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        
        g = self.sigmoid(output) * g_jump  #这里对g重新赋值不会影响到g_jump = g

        fuse = self.residual(torch.cat([g, l, X_f], 1)) #记录word
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual #最后一个融合模块取这里的特征用作类激活图
        return out




#通道注意力机制类
class senet(nn.Module):
    def __init__(self,channel,ratio = 16):
        super(senet,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False),
            nn.Sigmoid(),

        )
    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avg_pool(x).view([b,c])
        fc = self.fc(avg).view([b,c,1,1])
        #print(fc)
        return x*fc



