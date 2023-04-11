import torch
import torch.nn as nn
from utils.moments_loss import CMD_loss

vgg = nn.Sequential(
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


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def calc_CMD(input, target):
    cmd_weights = (1, 1, 1, 1)
    cmd_input = CMD_loss(input, k=len(cmd_weights), weights=cmd_weights)
    cmd = cmd_input(target)
    return cmd

class VGG_loss(nn.Module):
    def __init__(self, config, vgg):
        super(VGG_loss, self).__init__()
       
        self.config = config

        vgg_pretrained = config.vgg_model
        vgg.load_state_dict(torch.load(vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        vgg_enc_layers = list(vgg.children())
        
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:3])  # input -> relu1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[3:10])  # relu1_1 -> relu2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[10:17])  # relu2_1 -> relu3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[17:30])  # relu3_1 -> relu4_1
        self.vgg_enc_5 = nn.Sequential(*vgg_enc_layers[30:43])

        self.mse_loss = nn.MSELoss()

        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4', 'vgg_enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_vgg_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
   
    # extract relu3_1 from input image
    def encode_vgg_content(self, input):
        for i in range(4):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        
        loss = self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

        return loss

    def perceptual_loss(self, input_img, trs_img):
        # vgg content and style loss
        input_feats_vgg = self.encode_with_vgg_intermediate(input_img.repeat(1, 3, 1, 1))
        trs_feats_vgg = self.encode_with_vgg_intermediate(trs_img.repeat(1, 3, 1, 1))

        loss = 0.
        for i in range(0, 5):
            loss += self.calc_content_loss(trs_feats_vgg[i], input_feats_vgg[i])
        
        loss *= self.config.lambda_percept
        return loss

    def contrastive_loss(self, config, style_feats, res_style_feats, style_kernel, res_kernel, real_B, trs_AtoB, net_list):
        style_feats_E, style_feats_S = style_feats
        res_style_feats_E, res_style_feats_S = res_style_feats

        batch_size = style_feats_E[0].shape[0]
        loss_contrastive = 0.
        neg_idx_list = []
        for i in range(batch_size):
            pos_loss = 0.
            neg_loss = 0.
            min_neg = 1000000000.

            for j in range(batch_size):
                if j==i:
                    FeatMod_loss_E = self.calc_style_loss(res_style_feats_E[0][i].unsqueeze(0), style_feats_E[0][j].unsqueeze(0)) + \
                                    self.calc_style_loss(res_style_feats_E[1][i].unsqueeze(0), style_feats_E[1][j].unsqueeze(0))
                                    
                    FeatMod_loss_S = self.calc_content_loss(res_style_feats_S[0][i].unsqueeze(0), style_feats_S[0][j].unsqueeze(0)) + \
                                    self.calc_content_loss(res_style_feats_S[1][i].unsqueeze(0), style_feats_S[1][j].unsqueeze(0))

                    # depthwise & pointwise & bias from 3 layers in Generator
                    FilterMod_loss_h = 0
                    FilterMod_loss_l = 0
                    for k in range(len(res_kernel)):
                        FilterMod_loss_h += self.calc_content_loss(res_kernel[k][0][0][i], style_kernel[k][0][0][j]) + \
                                        self.calc_content_loss(res_kernel[k][0][1][i], style_kernel[k][0][1][j]) + \
                                        self.calc_content_loss(res_kernel[k][0][2][i], style_kernel[k][0][2][j]) 

                        FilterMod_loss_l += self.calc_content_loss(res_kernel[k][1][0][i], style_kernel[k][1][0][j]) + \
                                        self.calc_content_loss(res_kernel[k][1][1][i], style_kernel[k][1][1][j]) + \
                                        self.calc_content_loss(res_kernel[k][1][2][i], style_kernel[k][1][2][j])

                    pos_loss = FeatMod_loss_E * config.lambda_contr_E + FeatMod_loss_S * config.lambda_contr_S + \
                                (FilterMod_loss_h + FilterMod_loss_l) * config.lambda_contr_K

                else:
                    FeatMod_loss_E = self.calc_style_loss(res_style_feats_E[0][i].unsqueeze(0), style_feats_E[0][j].unsqueeze(0)) + \
                                    self.calc_style_loss(res_style_feats_E[1][i].unsqueeze(0), style_feats_E[1][j].unsqueeze(0))
                                    
                    FeatMod_loss_S = self.calc_content_loss(res_style_feats_S[0][i].unsqueeze(0), style_feats_S[0][j].unsqueeze(0)) + \
                                    self.calc_content_loss(res_style_feats_S[1][i].unsqueeze(0), style_feats_S[1][j].unsqueeze(0))

                    # depthwise & pointwise & bias
                    FilterMod_loss_h = 0
                    FilterMod_loss_l = 0
                    for k in range(len(res_kernel)):
                        FilterMod_loss_h += self.calc_content_loss(res_kernel[k][0][0][i], style_kernel[k][0][0][j]) + \
                                        self.calc_content_loss(res_kernel[k][0][1][i], style_kernel[k][0][1][j]) + \
                                        self.calc_content_loss(res_kernel[k][0][2][i], style_kernel[k][0][2][j])

                        FilterMod_loss_l += self.calc_content_loss(res_kernel[k][1][0][i], style_kernel[k][1][0][j]) + \
                                        self.calc_content_loss(res_kernel[k][1][1][i], style_kernel[k][1][1][j]) + \
                                        self.calc_content_loss(res_kernel[k][1][2][i], style_kernel[k][1][2][j])
                    
                    neg_each = FeatMod_loss_E * config.lambda_contr_E + FeatMod_loss_S * config.lambda_contr_S + \
                                (FilterMod_loss_h + FilterMod_loss_l) * config.lambda_contr_K
                    
                    if neg_each < min_neg:
                        neg_idx = j
                        min_neg = neg_each
                    neg_loss += neg_each

            neg_idx_list.append(neg_idx)
            loss_contrastive = loss_contrastive + pos_loss/neg_loss

        return loss_contrastive, neg_idx_list
