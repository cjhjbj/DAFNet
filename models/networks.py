import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            # lr_l = 0.3 ** max(0, epoch + opt.epoch_count - opt.n_epochs)
            # return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


class StyleOffsetPredictor(nn.Module):
    def __init__(self,
                 style_in_channels,  # 风格特征F_s.s的输入通道数
                 content_in_channels,  # 内容特征F_c.c的输入通道数（如960）
                 hidden_channels=128,
                 out_channels=512):  # 新增：目标输出通道数（固定为512）
        super(StyleOffsetPredictor, self).__init__()

        # 1. 风格特征处理分支：Conv → ReLU → AvgPool（完全保留）
        self.style_encoder = nn.Sequential(
            nn.Conv2d(style_in_channels, hidden_channels, kernel_size=3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        # 2. 生成缩放因子和偏移量（输出维度仍匹配content_in_channels，保证仿射效果）
        self.scale_predictor = nn.Linear(hidden_channels, content_in_channels)
        self.shift_predictor = nn.Linear(hidden_channels, content_in_channels)

        # 3. 新增：通道融合层（1×1卷积将content_in_channels→512，无激活，避免破坏特征）
        self.channel_fusion = nn.Conv2d(
            in_channels=content_in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 仅调整通道，不改变空间维度
            bias=True
        )

    def forward(self, F_c_c, F_s_s):
        """
        参数:
            F_c_c: 内容特征图，shape=[B, C_content, H, W]（如B,960,16,16）
            F_s_s: 风格特征图，shape=[B, C_style, H_s, W_s]
        返回:
            F_c_s_1: 调整后的内容特征图，shape=[B, 512, H, W]
        """
        # Step 1: 处理风格特征，得到全局统计向量（完全保留）
        F_c_c_norm = mean_variance_norm(F_c_c)
        style_feat = self.style_encoder(F_s_s)  # shape=[B, hidden_channels, 1, 1]
        style_feat = style_feat.flatten(1)  # shape=[B, hidden_channels]

        # Step 2: 预测缩放因子和偏移量（完全保留）
        scale = self.scale_predictor(style_feat)  # shape=[B, C_content]
        shift = self.shift_predictor(style_feat)  # shape=[B, C_content]

        # Step 3: 调整维度（完全保留）
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)

        # Step 4: 仿射变换（完全保留）
        F_c_s_affine = F_c_c_norm * scale + shift  # shape=[B, C_content, H, W]

        # Step 5: 新增：通道融合，缩减到
        F_c_s_1 = self.channel_fusion(F_c_s_affine)  # shape=[B, 512, H, W]

        return F_c_s_1

class Attention(nn.Module):

    def __init__(self, in_planes, max_sample=256 * 256, key_planes=None):
        super(Attention, self).__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style, seed=None):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, _, h_g, w_g = G.size()
        G = G.view(b, -1, w_g * h_g).contiguous()
        if w_g * h_g > self.max_sample:
            if seed is not None:
                torch.manual_seed(seed)
            index = torch.randperm(w_g * h_g).to(content.device)[:self.max_sample]
            G = G[:, :, index]
            style_flat = H.view(b, -1, w_g * h_g)[:, :, index].contiguous()
        else:
            style_flat = H.view(b, -1, w_g * h_g).contiguous()
        b, _, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        S = torch.bmm(F, G)
        # S: b, n_c, n_s
        S = self.sm(S)

        styled_feat_flat = torch.bmm(style_flat, S.permute(0, 2, 1))


        styled_feat = styled_feat_flat.view(b, -1, h, w).contiguous()  # [b, c, h_c, w_c]


        O = self.out_conv(styled_feat)

        return O

class DualOutputFeatureNet_IN(nn.Module):


    def __init__(self,
                 in_channels,
                 out_channels1,
                 out_channels2,
                 hidden_channels):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels1, kernel_size=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels , kernel_size=3),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels2, kernel_size=1)
        )

    def forward(self, x):
        assert len(x.shape) == 4, f"输入必须是4维特征图，当前维度：{x.shape}"

        feat = self.backbone(x)
        style = self.branch1(feat)
        content = self.branch2(feat)

        return style, content

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d( 256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder_layer_3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder_layer_4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3))
        )

    def forward(self, cs):
        cs = self.decoder_layer_1(cs)
        cs = self.decoder_layer_2(cs)
        cs = self.decoder_layer_3(cs)
        cs = self.decoder_layer_4(cs)
        return cs

