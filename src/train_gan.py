# creating a Trainer class to load models, data along with helper functions to train model
class Trainer:
    def __init__(self, G, D, train_loader, val_loader, num_epochs=50, start_epoch=1, lr=2e-4, device="cuda", checkpoint_dir="checkpoints/"):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.device = device
        self.lr = lr
        self.G = G
        self.D = D
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.FloatTensor).to(device)
        self.criterionL1 = nn.L1Loss().to(device)
        if device.type == "cuda":
          self.criterionVGG = VGGLoss(gpu_ids=[device]).to(device)
        else:
          self.criterionVGG = VGGLoss(gpu_ids=[])
        self.opt_G = Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_D = Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

        self.checkpoint_dir = checkpoint_dir
        self.f = open(f"{checkpoint_dir}/losses.csv", "a")
        self.csv_writer = csv.writer(self.f)
        pass

    def save_checkpoint(self, state, path):
        # saves checkpoints of model
        torch.save(state, path)

    def load_checkpoint(self, path):
        # loads specific checkpoint of model
        ckpt = torch.load(path, map_location='cpu')
        self.G.load_state_dict(ckpt['G_state'])
        self.D.load_state_dict(ckpt['D_state'])
        self.opt_G.load_state_dict(ckpt['opt_G'])
        self.opt_D.load_state_dict(ckpt['opt_D'])
        return ckpt['epoch'] + 1

    def validate(self, epoch, out_dir):
        # validates model  against validation set printing losses, and writing first batch of images to file.
        self.G.eval()
        os.makedirs(out_dir, exist_ok=True)
        total_val_G_loss = 0
        total_val_D_loss = 0
        with torch.no_grad():
            for idx, (real, ghibli) in enumerate(self.val_loader):
                #real, ghibli = next(iter(self.val_loader))
                real, ghibli = real.to(self.device), ghibli.to(self.device)
                fake = self.G(real)
                grid = torch.cat([real, ghibli, fake], dim=0)
                D_input_real = torch.cat([real, ghibli], dim=1)
                D_input_fake = torch.cat([real, fake], dim=1)
                loss_D = self.criterionGAN(self.D(D_input_real), True) + self.criterionGAN(self.D(D_input_fake), False)
                D_input_fake = torch.cat([real, fake], dim=1)
                loss_G = self.criterionGAN(self.D(D_input_fake), True) \
                        + 10.0 * self.criterionL1(fake, ghibli) \
                        + 5.0 * self.criterionVGG(fake, ghibli)
                total_val_G_loss += loss_G.item()
                total_val_D_loss += loss_D.item()

                if idx == 0:
                    save_image((grid * 0.5 + 0.5), os.path.join(out_dir, f"val_epoch_{epoch}.png"), nrow=real.size(0))
                    print(f"    Validation image saved for epoch {epoch}")
        print(f"    Validation G_Loss: {total_val_G_loss/len(self.val_loader)} D_Loss: {total_val_D_loss/len(self.val_loader)}")

    def train(self):
        # Entire training loop
        print("Starting Training")
        try:
          for epoch in range(start_epoch, self.num_epochs + 1):
              self.G.train(); self.D.train()
              epoch_G_loss = 0
              epoch_D_loss = 0
              img_cntr = 0
              print(f"  Epoch {epoch}")
              for real, ghibli in self.train_loader:
                  real, ghibli = real.to(self.device), ghibli.to(self.device)

                  # Train Discriminator
                  fake = self.G(real)
                  D_input_real = torch.cat([real, ghibli], dim=1)
                  D_input_fake = torch.cat([real, fake], dim=1)
                  #print(D_input_real)
                  #print(D_input_fake)
                  loss_D = self.criterionGAN(self.D(D_input_real), True) + self.criterionGAN(self.D(D_input_fake), False)
                  self.opt_D.zero_grad()
                  loss_D.backward(retain_graph=True)
                  self.opt_D.step()

                  # Train Generator
                  D_input_fake = torch.cat([real, fake], dim=1)
                  loss_G = self.criterionGAN(self.D(D_input_fake), True) \
                        + 10.0 * self.criterionL1(fake, ghibli) \
                        + 5.0 * self.criterionVGG(fake, ghibli)
                  self.opt_G.zero_grad()
                  loss_G.backward()
                  self.opt_G.step()

                  epoch_G_loss += loss_G.item()
                  epoch_D_loss += loss_D.item()

                  real.detach()
                  ghibli.detach()
                  fake.detach()

                  print(f"    Image {img_cntr}, G_loss: {loss_G.item():.4f} D_loss: {loss_D.item():.4f}")
                  img_cntr += 1

              print(f"  [Epoch {epoch}/{self.num_epochs}] G_loss: {epoch_G_loss/len(train_loader):.4f} "
                    f"D_loss: {epoch_D_loss/len(train_loader):.4f}")
              self.csv_writer.writerow([epoch, epoch_G_loss/len(train_loader), epoch_D_loss/len(train_loader)])

              self.save_checkpoint({
                  'epoch': epoch,
                  'G_state': G.state_dict(),
                  'D_state': D.state_dict(),
                  'opt_G': self.opt_G.state_dict(),
                  'opt_D': self.opt_D.state_dict()
              }, os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch}.pth"))

              if epoch % 1 == 0:
                self.validate(epoch, self.checkpoint_dir)

        except KeyboardInterrupt:
            print("Training Interrupted")

        self.f.close()

  data_path = ""
root_path = ""
if os.path.exists("/content/drive/MyDrive"):
    root_path = "/content/drive/MyDrive/"
    data_path = "/content/drive/MyDrive/"

data_path += r"Final_Ghibli_Dataset/ghibli-illustration-generated"

train_root = data_path
val_root = "data/val"
test_root = "data/test"

# Data loaders
train_ds = GhibliDataset(train_root)
#val_ds = GhibliDataset(val_root)
#test_ds = GhibliDataset(test_root)

total_size = len(train_ds)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    train_ds, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
  G = define_G(3, 3, ngf=32, netG="local", n_local_enhancers=1, gpu_ids=[device])
  D = define_D(6, ndf=64, n_layers_D=3, num_D=1, gpu_ids=[device])
else:
  G = define_G(3, 3, ngf=32, netG="local", n_local_enhancers=1)
  D = define_D(6, ndf=64, n_layers_D=3, num_D=1)

# Optional: Resume from checkpoint
start_epoch = 1
latest_ckpt = f"{root_path}/checkpoints/ckpt_epoch_20.pth"

trainer = Trainer(G, D, train_loader, val_loader, num_epochs=1000, start_epoch=start_epoch, device=device, checkpoint_dir=f"{root_path}/checkpoints/")

if os.path.isfile(latest_ckpt):
    start_epoch = trainer.load_checkpoint(latest_ckpt)

# Train
trainer.train()

#Plotting Curves
import pandas as pd
import matplotlib.pyplot as plt

losses = pd.read_csv("checkpoints/losses.csv", header=None)[:-1]
losses

plt.plot(losses[0], losses[1])
plt.title('G loss over epochs')
plt.xlabel('epochs')
plt.ylabel('G Loss')
plt.savefig('GLoss.png')
plt.show()

plt.plot(losses[0], losses[2])
plt.title('D loss over epochs')
plt.xlabel('epochs')
plt.ylabel('D Loss')
plt.savefig('DLoss.png')
plt.show()

def load_checkpoint(G, D, path):
        ckpt = torch.load(path, map_location='cpu')
        G.load_state_dict(ckpt['G_state'])
        D.load_state_dict(ckpt['D_state'])
        return G, D

def test(G, D, test_loader, out_dir, device):
    G.eval()
    os.makedirs(out_dir, exist_ok=True)
    criterionGAN = GANLoss(use_lsgan=True, tensor=torch.FloatTensor).to(device)
    criterionL1 = nn.L1Loss().to(device)
    if device.type == "cuda":
        criterionVGG = VGGLoss(gpu_ids=[device]).to(device)
    else:
        criterionVGG = VGGLoss(gpu_ids=[])
    total_test_G_loss = 0
    total_test_D_loss = 0
    with torch.no_grad():
        for idx, (real, ghibli) in enumerate(test_loader):
            #real, ghibli = next(iter(self.val_loader))
            real, ghibli = real.to(device), ghibli.to(device)
            fake = G(real)
            grid = torch.cat([real, ghibli, fake], dim=0)
            D_input_real = torch.cat([real, ghibli], dim=1)
            D_input_fake = torch.cat([real, fake], dim=1)
            loss_D = criterionGAN(D(D_input_real), True) + criterionGAN(D(D_input_fake), False)
            D_input_fake = torch.cat([real, fake], dim=1)
            loss_G = criterionGAN(D(D_input_fake), True) \
                    + 10.0 * criterionL1(fake, ghibli) \
                    + 5.0 * criterionVGG(fake, ghibli)
            total_test_G_loss += loss_G.item()
            total_test_D_loss += loss_D.item()

            #if idx == 0:
            save_image((grid * 0.5 + 0.5), os.path.join(out_dir, f"test_batch_{idx}.png"), nrow=real.size(0))
            print(f"    Test image saved for batch {idx}")
    print(f"    Test G_Loss: {total_test_G_loss/len(test_loader)} D_Loss: {total_test_D_loss/len(test_loader)}")

data_path = ""
root_path = ""
if os.path.exists("/content/drive/MyDrive"):
    root_path = "/content/drive/MyDrive/"
    data_path = "/content/drive/MyDrive/"

out_dir = f"{root_path}/test_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
  G = define_G(3, 3, ngf=32, netG="local", n_local_enhancers=1, gpu_ids=[device])
  D = define_D(6, ndf=64, n_layers_D=3, num_D=1, gpu_ids=[device])
else:
  G = define_G(3, 3, ngf=32, netG="local", n_local_enhancers=1)
  D = define_D(6, ndf=64, n_layers_D=3, num_D=1)

latest_ckpt = f"{root_path}/checkpoints/ckpt_epoch_110.pth"

G, D = load_checkpoint(G, D, latest_ckpt)
test(G, D, test_loader, out_dir, device)

