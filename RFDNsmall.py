import torch
import torch.nn as nn
import block as B

def make_model(args, parent=False):
    model = RFDNsmall()
    return model


class RFDNsmall(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        super(RFDNsmall, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDBsmall(in_channels=nf)
        self.B2 = B.RFDBsmall(in_channels=nf)
        self.B3 = B.RFDBsmall(in_channels=nf)
        self.B4 = B.RFDBsmall(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        self.skip_add = torch.nn.quantized.FloatFunctional()


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.skip_add.add(self.LR_conv(out_B), out_fea)

        output = self.upsampler(out_lr)

        return output
    
    def destill_forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.skip_add.add(self.LR_conv(out_B), out_fea)

        output = self.upsampler(out_lr)
        return out_B1, out_B2, out_B3, out_B4, out_B, output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
