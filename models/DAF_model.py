import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks


class DAFModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--image_encoder_path', required=True, help='path to pretrained image encoder')
        parser.add_argument('--shallow_layer', action='store_true',
                            help='if specified, also use features of shallow layers')
        if is_train:
            parser.add_argument('--lambda_content', type=float, default=1., help='weight for L2 content loss')
            parser.add_argument('--lambda_style', type=float, default=3., help='weight for L2 style loss')
            parser.add_argument('--lambda_decouple_c', type=float, default=1.0, help='weight for L2 decouple_c loss')
            parser.add_argument('--lambda_decouple_s', type=float, default=1.0, help='weight for L2 decouple_s loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load(opt.image_encoder_path))
        enc_layers = list(image_encoder.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(opt.gpu_ids[0]), opt.gpu_ids)
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(opt.gpu_ids[0]), opt.gpu_ids)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        if hasattr(opt, 'pair_style') and opt.pair_style:
            self.visual_names = ['c','s1', 's2','interp_outputs']
        else:
            self.visual_names = ['c', 'cs', 's']
        self.model_names = ['decoder','Attention_1','DualOutputFeatureNet_IN_1','DualOutputFeatureNet_IN_2','DualOutputFeatureNet_IN_3','DualOutputFeatureNet_IN_4','StyleOffsetPredictor_1']
        parameters = []
        DualOutputFeatureNet_IN_1 = networks.DualOutputFeatureNet_IN(
            in_channels=64,
            out_channels1=64,
            out_channels2=64,
            hidden_channels=128
        )
        DualOutputFeatureNet_IN_2 = networks.DualOutputFeatureNet_IN(
            in_channels=128,
            out_channels1=128,
            out_channels2=128,
            hidden_channels=256
        )
        DualOutputFeatureNet_IN_3 = networks.DualOutputFeatureNet_IN(
            in_channels=256,
            out_channels1=256,
            out_channels2=256,
            hidden_channels=512
        )
        DualOutputFeatureNet_IN_4 = networks.DualOutputFeatureNet_IN(
            in_channels=512,
            out_channels1=512,
            out_channels2=512,
            hidden_channels=1024
        )

        Attention_1 = networks.Attention(in_planes=512)
        self.net_Attention_1 = networks.init_net(Attention_1, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_DualOutputFeatureNet_IN_1 = networks.init_net(DualOutputFeatureNet_IN_1, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.net_DualOutputFeatureNet_IN_2 = networks.init_net(DualOutputFeatureNet_IN_2, opt.init_type, opt.init_gain,
                                                               opt.gpu_ids)
        self.net_DualOutputFeatureNet_IN_3 = networks.init_net(DualOutputFeatureNet_IN_3, opt.init_type, opt.init_gain,
                                                               opt.gpu_ids)
        self.net_DualOutputFeatureNet_IN_4 = networks.init_net(DualOutputFeatureNet_IN_4, opt.init_type, opt.init_gain,
                                                               opt.gpu_ids)
        StyleOffsetPredictor_1 = networks.StyleOffsetPredictor(
            style_in_channels = 64+128+256+512,
            content_in_channels =64+128+256+512,
            hidden_channels=128,
            out_channels=512
        )
        self.net_StyleOffsetPredictor_1 = networks.init_net(StyleOffsetPredictor_1,opt.init_type, opt.init_gain, opt.gpu_ids)
        self.max_sample = 64 * 64
        decoder = networks.Decoder()
        self.net_decoder = networks.init_net(decoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        parameters.append(self.net_decoder.parameters())
        parameters.append(self.net_DualOutputFeatureNet_IN_1.parameters())
        parameters.append(self.net_DualOutputFeatureNet_IN_2.parameters())
        parameters.append(self.net_DualOutputFeatureNet_IN_3.parameters())
        parameters.append(self.net_DualOutputFeatureNet_IN_4.parameters())
        parameters.append(self.net_StyleOffsetPredictor_1.parameters())
        parameters.append(self.net_Attention_1.parameters())
        self.c = None
        self.cs = None
        self.cs1 = None
        self.cs2 = None
        self.s = None
        self.s1 = None
        self.s2 = None
        self.s_feats = None
        self.c_feats = None
        self.s1_feats = None
        self.s2_feats = None
        self.seed = 6666
        self.ALPHA_RATIOS = [1.0, 0.75, 0.5, 0.25, 0.0]
        self.cc_dual_output_list = None
        self.cs_dual_output_list = None
        self.sc_dual_output_list = None
        self.ss_dual_output_list = None
        self.s1c_dual_output_list = None
        self.s1s_dual_output_list = None
        self.s2c_dual_output_list = None
        self.s2s_dual_output_list = None
        self.cs_feats = None
        self.cs_c_feats = None
        self.cs_s_feats = None
        self.interp_outputs = []
        if self.isTrain:
            self.loss_names = ['content', 'style','decouple_c', 'decouple_s']
            self.criterionMSE = torch.nn.MSELoss().to(self.device)
            self.optimizer_g = torch.optim.Adam(itertools.chain(*parameters), lr=opt.lr)
            self.optimizers.append(self.optimizer_g)
            self.loss_style = torch.tensor(0., device=self.device)
            self.loss_content = torch.tensor(0., device=self.device)
            self.loss_decouple_c = torch.tensor(0., device=self.device)
            self.loss_decouple_s = torch.tensor(0., device=self.device)

    def set_input(self, input_dict):
        if 's1' in input_dict and 's2' in input_dict:
            self.s = None
            self.s1 = input_dict['s1'].to(self.device)
            self.s2 = input_dict['s2'].to(self.device)
        else:
            self.s = input_dict['s'].to(self.device)
            self.s1 = self.s2 = None
        self.c = input_dict['c'].to(self.device)
        self.image_paths = input_dict['name']

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(4):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    @staticmethod
    def get_key(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(networks.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(networks.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return networks.mean_variance_norm(feats[last_layer_idx])

    @staticmethod
    def get_key_no_norm(feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(nn.functional.interpolate(feats[i], (h, w)))
            results.append(feats[last_layer_idx])
            return torch.cat(results, dim=1)
        else:
            return feats[last_layer_idx]
    def forward(self):
        if self.s is not None:
            self.c_feats = self.encode_with_intermediate(self.c)
            self.s_feats = self.encode_with_intermediate(self.s)
            c1s,c1c = self.net_DualOutputFeatureNet_IN_1(self.c_feats[0])
            s1s,s1c = self.net_DualOutputFeatureNet_IN_1(self.s_feats[0])
            c2s,c2c = self.net_DualOutputFeatureNet_IN_2(self.c_feats[1])
            s2s,s2c = self.net_DualOutputFeatureNet_IN_2(self.s_feats[1])
            c3s,c3c = self.net_DualOutputFeatureNet_IN_3(self.c_feats[2])
            s3s,s3c = self.net_DualOutputFeatureNet_IN_3(self.s_feats[2])
            c4s,c4c = self.net_DualOutputFeatureNet_IN_4(self.c_feats[3])
            s4s,s4c = self.net_DualOutputFeatureNet_IN_4(self.s_feats[3])
            self.cc_dual_output_list = [
                c1c,c2c,c3c,c4c
            ]
            self.cs_dual_output_list = [
                c1s,c2s,c3s,c4s
            ]
            self.sc_dual_output_list = [
                s1c,s2c,s3c,s4c
            ]
            self.ss_dual_output_list = [
                s1s,s2s,s3s,s4s
            ]
            p1 = self.net_StyleOffsetPredictor_1(self.get_key_no_norm(self.cc_dual_output_list, 3, self.opt.shallow_layer),
                                                     self.get_key_no_norm(self.ss_dual_output_list, 3, self.opt.shallow_layer))
            p2 = self.net_Attention_1(c4c,s4s,self.seed)
            p3 = p1+p2

            self.cs = self.net_decoder(p3)
            self.cs_feats = self.encode_with_intermediate(self.cs)
            cs1s,cs1c = self.net_DualOutputFeatureNet_IN_1(self.cs_feats[0])
            cs2s,cs2c = self.net_DualOutputFeatureNet_IN_2(self.cs_feats[1])
            cs3s,cs3c = self.net_DualOutputFeatureNet_IN_3(self.cs_feats[2])
            cs4s,cs4c = self.net_DualOutputFeatureNet_IN_4(self.cs_feats[3])
            self.cs_c_feats = [cs1c, cs2c, cs3c, cs4c]
            self.cs_s_feats = [cs1s, cs2s, cs3s, cs4s]
        else:
            self.interp_outputs.clear()
            self.c_feats = self.encode_with_intermediate(self.c)
            self.s1_feats = self.encode_with_intermediate(self.s1)
            c1s, c1c = self.net_DualOutputFeatureNet_IN_1(self.c_feats[0])
            s11s, s11c = self.net_DualOutputFeatureNet_IN_1(self.s1_feats[0])
            c2s, c2c = self.net_DualOutputFeatureNet_IN_2(self.c_feats[1])
            s12s, s12c = self.net_DualOutputFeatureNet_IN_2(self.s1_feats[1])
            c3s, c3c = self.net_DualOutputFeatureNet_IN_3(self.c_feats[2])
            s13s, s13c = self.net_DualOutputFeatureNet_IN_3(self.s1_feats[2])
            c4s, c4c = self.net_DualOutputFeatureNet_IN_4(self.c_feats[3])
            s14s, s14c = self.net_DualOutputFeatureNet_IN_4(self.s1_feats[3])
            self.cc_dual_output_list = [
                c1c, c2c, c3c, c4c
            ]
            self.cs_dual_output_list = [
                c1s, c2s, c3s, c4s
            ]
            self.s1c_dual_output_list = [
                s11c, s12c, s13c, s14c
            ]
            self.s1s_dual_output_list = [
                s11s, s12s, s13s, s14s
            ]
            p1 = self.net_StyleOffsetPredictor_1(
                self.get_key_no_norm(self.cc_dual_output_list, 3, self.opt.shallow_layer),
                self.get_key_no_norm(self.s1s_dual_output_list, 3, self.opt.shallow_layer))
            p2 = self.net_Attention_1(c4c, s14s, self.seed)
            p3 = p1 + p2

            self.cs1 = self.net_decoder(p3)
            self.s2_feats = self.encode_with_intermediate(self.s2)
            s21s, s21c = self.net_DualOutputFeatureNet_IN_1(self.s2_feats[0])
            s22s, s22c = self.net_DualOutputFeatureNet_IN_2(self.s2_feats[1])
            s23s, s23c = self.net_DualOutputFeatureNet_IN_3(self.s2_feats[2])
            s24s, s24c = self.net_DualOutputFeatureNet_IN_4(self.s2_feats[3])
            self.s2c_dual_output_list = [
                s21c, s22c, s23c, s24c
            ]
            self.s2s_dual_output_list = [
                s21s, s22s, s23s, s24s
            ]
            p12 = self.net_StyleOffsetPredictor_1(
                self.get_key_no_norm(self.cc_dual_output_list, 3, self.opt.shallow_layer),
                self.get_key_no_norm(self.s2s_dual_output_list, 3, self.opt.shallow_layer))
            p22 = self.net_Attention_1(c4c, s24s, self.seed)
            p32 = p12 + p22

            self.cs2 = self.net_decoder(p32)

            for a in self.ALPHA_RATIOS:
                fused = a * self.cs1 + (1 - a) * self.cs2
                self.interp_outputs.append(fused)


    def compute_decouple_loss(self):
        self.loss_decouple_c = torch.tensor(0., device=self.device)
        self.loss_decouple_s = torch.tensor(0., device=self.device)
        if self.opt.lambda_decouple_c > 0 and self.opt.lambda_decouple_s > 0:
            for cs_c, c_c in zip(self.cs_c_feats, self.cc_dual_output_list):
                self.loss_decouple_c += self.criterionMSE(cs_c, c_c)
            for cs_s, s_s in zip(self.cs_s_feats, self.ss_dual_output_list):
                self.loss_decouple_s += self.criterionMSE(cs_s, s_s)
    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[-1]),
                                                    networks.mean_variance_norm(self.c_feats[-1]))


    def compute_style_loss(self, stylized_feats):
        self.loss_style = torch.tensor(0., device=self.device)
        if self.opt.lambda_style > 0:
            for i in range(0, 4):
                s_feats_mean, s_feats_std = networks.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = networks.calc_mean_std(stylized_feats[i])
                self.loss_style += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)




    def compute_losses(self):
        stylized_feats = self.cs_feats
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.compute_decouple_loss()
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_style = self.loss_style * self.opt.lambda_style
        self.loss_decouple_c = self.loss_decouple_c * self.opt.lambda_decouple_c
        self.loss_decouple_s = self.loss_decouple_s * self.opt.lambda_decouple_s


    def optimize_parameters(self):
        self.seed = int(torch.randint(10000000, (1,))[0])
        self.forward()
        self.optimizer_g.zero_grad()
        self.compute_losses()
        loss = self.loss_content + self.loss_style + self.loss_decouple_c + self.loss_decouple_s
        loss.backward()
        self.optimizer_g.step()

