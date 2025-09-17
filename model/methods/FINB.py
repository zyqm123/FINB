import torch
import torch.nn as nn
import numpy as np

from model.registry import MODEL
import gol
#在AN模块的代码中出现的import gol是一个全局状态管理工具，通常用于在程序的不同模块间共享和访问全局状态变量
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)


#再看看，不理解
class Domain_transform(nn.Module):
    def __init__(self, planes):
        super(Domain_transform, self).__init__()
        self.planes = planes
        self.avg = torch.nn.AdaptiveAvgPool2d((1,1)) # 全局平均池化
        self.linear=torch.nn.Linear(planes, 1) # 全连接层(降维到1)
        self.relu = torch.nn.ReLU() # 激活函数

    def forward(self, x): #[B, C, H, W]
        x = x.detach().data # 切断梯度回传(不影响主网络)，[B,C,H,W]
        x = self.avg(x).view(-1, self.planes) # 全局平均池化+reshape [B,C,1,1]->[B,C]
        x = self.linear(x) # 计算域偏移量 [B,1]
        x = self.relu(x) # 确保偏移量≥0 [B,1]
        domain_offset = x.mean() # 取batch平均 [1]
        return domain_offset # 返回标量

#normalized distribution alignment,目标自适应归一化
class AN(nn.Module):

    def __init__(self, planes):
        super(AN, self).__init__()
        self.IN = nn.InstanceNorm2d(planes, affine=False)
        self.BN = nn.BatchNorm2d(planes, affine=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True) # 1. 声明为可训练参数2. 初始化为0.0的浮点张量启用梯度计算(默认True可省略)
        self.alpha_t = torch.Tensor([0.0])
        self.domain_transform = Domain_transform(planes)

    def forward(self, x):
        a=gol.get_value('is_ft')
        b=gol.get_value('use_transform')
        if gol.get_value('is_ft') and gol.get_value('use_transform'): #目前没有使用动态比例因子
            self.alpha_t = self.alpha + 0.01 * self.domain_transform(x) #tao（F）
            t = torch.sigmoid(self.alpha_t).cuda()
        else:
            t = torch.sigmoid(self.alpha).cuda()
        out_in = self.IN(x) #这里的均值和方差都是支持集的？
        out_bn = self.BN(x)
        out = t * out_in + (1 - t) * out_bn
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, block_size=1, is_maxpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.adafbi1 = AN(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.adafbi2 = AN(planes)
        self.stride = stride
        self.downsample = downsample
        self.block_size = block_size #未使用
        self.is_maxpool = is_maxpool
        self.maxpool = nn.MaxPool2d(stride)
        self.num_batches_tracked = 0

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.adafbi1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.adafbi2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.is_maxpool:
            out = self.maxpool(out)
        return out


class ResNetAN(nn.Module):

    def __init__(self, block, resolution):
        super(ResNetAN, self).__init__()
        self.inplanes = 64 # 初始通道数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.adafbi = AN(64)
        self.relu = nn.ReLU()
        # 四个层级构造
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2)
        self.resolution = resolution

        # 初始化权重
        for m in self.modules(): # 遍历所有子模块
            if isinstance(m, nn.Conv2d): # 判断是否为卷积层
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, stride=1, block_size=1, is_maxpool=True):
        downsample = None
        #当 stride != 1 时（特征图尺寸需要减半）或当输入/输出通道数不匹配时
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        #layer块
        layers = []
        layer = block(self.inplanes, planes, stride, downsample,
                      block_size, is_maxpool=is_maxpool)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, AN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): #[128,3,160,160]
        x = self.conv1(x) #[128,64,80,80]
        x = self.adafbi(x) #[128,64,80,80]
        x = self.relu(x)
        x = self.layer1(x) #[128,64,40,40]
        x = self.layer2(x)#[128,128,20,20]
        x = self.layer3(x) #[128,256,10,10]
        x = self.layer4(x) #[128,512,5,5]
        return x

#使用装饰器注册模型
@MODEL.register
class FINB(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_channel = 512
        resolution = config.resolution
        self.feature_extractor = ResNetAN(BasicBlock, resolution=resolution) # 自定义ResNet变体
        self.resolution = self.feature_extractor.resolution
        
        # number of channels for the feature map, correspond to d in the paper
        self.d = num_channel
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True) # 温度系数
        self.r = nn.Parameter(torch.zeros(2), requires_grad=not config.is_pretraining) # 重构系数（β控制）

        # number of categories during pre-training
        self.num_classes = config.num_classes

        # category matrix
        # 可学习的类别原型矩阵
        self.cat_mat = nn.Parameter(torch.randn(self.num_classes, self.resolution, self.d), requires_grad=True)

    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp) # [N, C, H, W]
        feature_map = feature_map / np.sqrt(640) # 特征标准化
        feature_map = feature_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()  # N,HW,C
        return feature_map

    def get_recon_dist(self, query, support, beta):
        # query: way*query_shot*resolution, d 3200,512
        # support: way, shot*resolution , 64,25,512 batch_size,resolution, d
        lam = support.size(1) / support.size(2)# 正则化系数计算 (shot*res)/d
        rho = beta.exp() # 重构强度系数 e^β保证正值
        st = support.permute(0, 2, 1)  # way, d, shot*resolution # 支持特征转置 [way, d, shot*res]
        # correspond to Equation 3 in the paper
        sst = support.matmul(st) # 计算S*S^T [way, d, d]
        sst_plus_ri = sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)# 添加正则项 [way, d, d]
        sst_plus_ri_np = sst_plus_ri.detach().cpu().numpy()
        sst_plus_ri_inv_np = np.linalg.inv(sst_plus_ri_np)# 矩阵求逆（当前实现）
        sst_plus_ri_inv = torch.tensor(sst_plus_ri_inv_np).cuda()
        w = query.matmul(st.matmul(sst_plus_ri_inv))  # way, d, d # 重构权重
        Q_bar = w.matmul(support).mul(rho)  # way, way*query_shot*resolution, d # 重构特征
        # way*query_shot*resolution, way
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0) # 平方欧氏距离 (公式4？)
        return dist

    def forward(self, inp): #[B，C,W,H], [128,3,160,160]
        feature_map = self.get_feature_map(inp)
        batch_size = feature_map.size(0)
        feature_map = feature_map.view(batch_size * self.resolution, self.d) #[128,3,160,160]
        beta = self.r[1]
        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat, beta=beta) #[3200,64]
        logits = recon_dist.neg().view(batch_size, self.resolution, self.num_classes).mean(1) #[128,64]
        logits = logits * self.scale #分类输出
        return logits
