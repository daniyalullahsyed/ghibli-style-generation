#Import PyTorch and define helper functions for weight initialization (weights_init) and choosing normalization layers (get_norm_layer).

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import random_split, DataLoader
import functools
from torch.autograd import Variable
import csv

# Iitializes the weights of the model
def weights_init(m):
    #print(type(m))
    classname = m.__class__.__name__
    #print(classname)
    #print(m)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        #print(m.weight)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# creates and returns specific normalization layers
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    else:
        raise ValueError(f"normalization layer {norm_type} is not found")

#Define two-layer residual block with padding -> Conv3×3 -> Norm -> ReLU -> An optional dropout -> Conv3×3 -> Norm.

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'padding [{padding_type}] is not implemented')

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim),
            activation
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

res_net_test = ResnetBlock(3, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True))
res_net_test

plt.imshow(rearrangeTorchArrayForPlt(ghibli_data.__getitem__(0)[0]))
plt.axis('off')
plt.show()

ghibli_data.__getitem__(0)[0].unsqueeze(dim=0).shape

out.detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out.detach().squeeze()))
plt.axis('off')
plt.show()

#Implement coarse global generator: 7×7 conv -> downsample (×N) -> N ResnetBlocks -> upsample (×N) -> 7×7 conv -> Tanh.

class GlobalGenerator(nn.Module):
    def __init__(
        self, input_nc, output_nc, ngf=64,
        n_downsampling=3, n_blocks=9,
        norm_layer=nn.BatchNorm2d, padding_type='reflect'
    ):
        super(GlobalGenerator, self).__init__()
        assert n_blocks >= 0, "n_blocks must be non-negative"
        activation = nn.ReLU(True)

        model = []
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation
        ]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation
            ]

        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, activation=activation)
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(int(ngf * mult / 2)),
                activation
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

global_gen_test = GlobalGenerator(3, 3, 32, 3, 9, get_norm_layer(norm_type='instance'))
global_gen_test

out = global_gen_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out.detach()

out.detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out.detach().squeeze()))
plt.axis('off')
plt.show()

#Build on the global generator’s output by stripping its last layers and adding one or more local down-up-sampling ResNet stages for high-res refinement.

class LocalEnhancer(nn.Module):
    def __init__(
        self, input_nc, output_nc, ngf=32,
        n_downsample_global=3, n_blocks_global=9,
        n_local_enhancers=1, n_blocks_local=3,
        norm_layer=nn.BatchNorm2d, padding_type='reflect'
    ):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ngf_global = ngf * (2 ** n_local_enhancers)
        global_net = GlobalGenerator(
            input_nc, output_nc, ngf_global,
            n_downsample_global, n_blocks_global, norm_layer
        ).model
        stripped_global = [global_net[i] for i in range(len(global_net) - 3)]
        self.model = nn.Sequential(*stripped_global)

        for n in range(1, n_local_enhancers + 1):
            ngf_local = ngf * (2 ** (n_local_enhancers - n))

            model_downsample = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_local, kernel_size=7, padding=0),
                norm_layer(ngf_local),
                nn.ReLU(True),
                nn.Conv2d(ngf_local, ngf_local * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf_local * 2),
                nn.ReLU(True)
            ]

            model_upsample = []
            for _ in range(n_blocks_local):
                model_upsample += [
                    ResnetBlock(ngf_local * 2, padding_type=padding_type, norm_layer=norm_layer, activation=nn.ReLU(True))
                ]

            model_upsample += [
                nn.ConvTranspose2d(ngf_local * 2, ngf_local,
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_local),
                nn.ReLU(True)
            ]

            if n == n_local_enhancers:
                model_upsample += [
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                    nn.Tanh()
                ]

            setattr(self, f'model{n}_1', nn.Sequential(*model_downsample))
            setattr(self, f'model{n}_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        input_downsampled = [input]
        for _ in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))
        output_prev = self.model(input_downsampled[-1])
        for n in range(1, self.n_local_enhancers + 1):
            model_down = getattr(self, f'model{n}_1')
            model_up   = getattr(self, f'model{n}_2')
            input_i = input_downsampled[self.n_local_enhancers - n]
            output_prev = model_up(model_down(input_i) + output_prev)
        return output_prev

local_enhance_test = LocalEnhancer(3, 3, 32, 3, 9, 1, 3, get_norm_layer(norm_type='instance'))
local_enhance_test

out = local_enhance_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out.detach()

out.detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out.detach().squeeze()))
plt.axis('off')
plt.show()

#Downsampling feature extractor: 7×7 conv -> norm -> ReLU -> repeated strided conv blocks -> final 7×7 conv.

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, n_downsampling=3, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        activation = nn.ReLU(True)
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation
        ]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                activation
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * (2 ** n_downsampling), output_nc, kernel_size=7, padding=0)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

encoder_test = Encoder(3, 3, 32, n_downsampling=3, norm_layer=nn.BatchNorm2d)
encoder_test

out = encoder_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out.detach()

out.detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out.detach().squeeze()))
plt.axis('off')
plt.show()

#Define single-scale PatchGAN (NLayerDiscriminator) and wrap multiple scales into MultiscaleDiscriminator, each judging real/fake at decreasing resolutions.

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        activation = nn.LeakyReLU(0.2, True)
        kw = 4
        padw = 1

        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    activation]
        nf = ndf

        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf_prev * 2, 512)
            sequence += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                activation
            ]

        nf_prev = nf
        nf = min(nf_prev * 2, 512)
        sequence += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            activation
        ]

        sequence += [nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        if getIntermFeat:
            self.model = nn.ModuleList([nn.Sequential(layer) for layer in sequence])
        else:
            self.model = nn.Sequential(*sequence)

    def forward(self, x):
        if self.getIntermFeat:
            res = [x]
            for m in self.model:
                res.append(m(res[-1]))
            return res[1:]
        else:
            return [self.model(x)]

n_discriminator_test = NLayerDiscriminator(3, 32, 3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False)
n_discriminator_test

out = n_discriminator_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out[0].detach()

out[0].detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out[0].detach().squeeze(dim=0)))
plt.axis('off')
plt.show()

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 num_D=1, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            setattr(self, f'd{i}', netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, x):
        if self.getIntermFeat:
            result = []
            input = x
            for m in model:
                result.append(m(input))
                input = m(input)
            return result
        else:
            return model(x)

    def forward(self, input):
        results = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, f'd{i}')
            if self.getIntermFeat:
                results.append(self.singleD_forward(model.model, input_downsampled))
            else:
                results.append(self.singleD_forward(model.model, input_downsampled))
            input_downsampled = self.downsample(input_downsampled)
        return results

mscale_discriminator_test = MultiscaleDiscriminator(3, ndf=32, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=1, getIntermFeat=False)
mscale_discriminator_test

out = mscale_discriminator_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out[0].detach()

out[0].detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out[0].detach().squeeze(dim=0)))
plt.axis('off')
plt.show()
