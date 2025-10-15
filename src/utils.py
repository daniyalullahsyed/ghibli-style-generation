#Provide helper functions (define_G/define_D) to instantiate G/D with proper init.

def define_G(
    input_nc, output_nc, ngf, netG,
    n_downsample_global=3, n_blocks_global=9,
    n_local_enhancers=1, n_blocks_local=3,
    norm='instance', gpu_ids=[]
):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        model = GlobalGenerator(input_nc, output_nc, ngf,
                                n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        model = LocalEnhancer(input_nc, output_nc, ngf,
                              n_downsample_global, n_blocks_global,
                              n_local_enhancers, n_blocks_local, norm_layer)
        #print("Model: ", model)
    elif netG == 'encoder':
        model = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ValueError('generator not implemented!')
    if gpu_ids:
        model.cuda(gpu_ids[0])
    model.apply(weights_init)
    return model

G_test = define_G(3, 3, ngf=32, netG="local", n_local_enhancers=1, gpu_ids=[])
G_test

out = G_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out.detach()

out.detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out.detach().squeeze(dim=0)))
plt.axis('off')
plt.show()

def define_D(
    input_nc, ndf, n_layers_D,
    norm='instance', use_sigmoid=False,
    num_D=1, getIntermFeat=False, gpu_ids=[]
):
    norm_layer = get_norm_layer(norm_type=norm)
    model = MultiscaleDiscriminator(
        input_nc, ndf, n_layers_D, norm_layer,
        use_sigmoid, num_D, getIntermFeat
    )
    if gpu_ids:
        model.cuda(gpu_ids[0])
    model.apply(weights_init)
    return model

D_test = define_D(3, ndf=64, n_layers_D=3, num_D=1, gpu_ids=[])
D_test

out = D_test(ghibli_data.__getitem__(0)[0].unsqueeze(dim=0))
out[0].shape

out[0].detach().shape

plt.imshow(rearrangeTorchArrayForPlt(out[0].detach().squeeze(dim=0)))
plt.axis('off')
plt.show()

#Implement GANLoss (LSGAN wrapper) and VGGLoss (perceptual loss) classes.

class GANLoss(nn.Module):
    def __init__(
        self, use_lsgan=True,
        target_real_label=1.0, target_fake_label=0.0,
        tensor=torch.FloatTensor
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_var is None or self.real_label_var.numel() != input.numel():
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False).to(input.device) # Move to input device
            return self.real_label_var.to(input.device) # Ensure it's on input device
        else:
            if self.fake_label_var is None or self.fake_label_var.numel() != input.numel():
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False).to(input.device) # Move to input device
            return self.fake_label_var.to(input.device) # Ensure it's on input device

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for pred in input:
                pred = pred[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        from torchvision import models
        vgg_model = models.vgg19(pretrained=True).features
        self.vgg = vgg_model
        if gpu_ids:
          self.vgg = vgg_model.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
