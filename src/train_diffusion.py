pip install diffusers==0.25.0 accelerate==0.30.1 torchvision

pip install scikit-image lpips

# =============================
# #### Library Imports and Setup
# =============================
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import Hugging Face diffusers and accelerate libraries for model loading and training optimization
from accelerate import Accelerator
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import LoRAAttnProcessor2_0, AttnProcessor, LoRAAttnProcessor

# =============================
# Dataset and DataLoader Setup
# =============================

class GhibliDataset(Dataset):
    """
    GhibliDataset is a custom PyTorch Dataset for loading images from specified directories.
    It expects each directory to contain pairs of images: one original (o.png) and one generated (g.png).
    The images are transformed using a series of augmentations and normalization before being returned.
    """
    def __init__(self, root_dirs):
        self.data = []
        # Allow single or multiple root directories
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        # Define image transformations including normalization and data augmentation
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Load image paths from each folder
        for root_dir in root_dirs:
            for folder in sorted(os.listdir(root_dir)):
                path = os.path.join(root_dir, folder)
                o_path = os.path.join(path, "o.png")
                g_path = os.path.join(path, "g.png")
                if os.path.exists(o_path) and os.path.exists(g_path):
                    self.data.append((o_path, g_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load original and stylized images, apply transforms, return as tensor pair
        o_img = Image.open(self.data[idx][0]).convert("RGB")
        g_img = Image.open(self.data[idx][1]).convert("RGB")
        return self.transform(o_img), self.transform(g_img)

## 2. Model Design and Implementation

We fine-tune a pretrained Stable Diffusion model using LoRA layers to guide image transformation in "Ghibli illustration" style. The model accepts an input image and a prompt and outputs a stylized image.

We inject **LoRA layers** into attention modules of the U-Net backbone to reduce training complexity and memory usage.

# =============================
# Accelerator Setup
# =============================
accelerator = Accelerator(mixed_precision="bf16") # Use bf16 for better performance on supported hardware
device = accelerator.device

# =============================
# Model Setup
# =============================

# Load the pretrained Stable Diffusion pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.bfloat16,  # or torch.float16
    safety_checker=None
)
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# Extract components for training
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder.to(device)
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# Load the U-Net and move to training device
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.bfloat16
).to(device)


# Replace attention processors in U-Net with LoRA processors
lora_attn_processors = {}
for name, module in unet.attn_processors.items():
    if isinstance(module, AttnProcessor):  # Only replace supported ones
        # Replace standard attention with LoRA version
        lora_attn_processors[name] = LoRAAttnProcessor2_0(base_processor=module)
    else:
        # Keep non-attention layers unchanged
        lora_attn_processors[name] = module  # keep original if not LoRA-compatible

# Sanity check: ensure all processors are replaced
if len(lora_attn_processors) != len(unet.attn_processors):
    raise ValueError(f"Expected {len(unet.attn_processors)} processors, got {len(lora_attn_processors)}")

# Apply LoRA processors to U-Net
unet.set_attn_processor(lora_attn_processors)

# Set model to training mode
unet.train()

from torch.utils.data import random_split

# Load the full dataset
full_dataset = GhibliDataset("ghibli-illustration-generated")

# Define split sizes (e.g., 80% train, 20% val)
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size

# Split the dataset
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# =============================
# Training Setup
# =============================

# Use AdamW optimizer for weight decay regularization
optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6)#1e-6)

# Prepare components using accelerator
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, train_loader)
vae.to(dtype=torch.bfloat16, device=device)

# Ensure only LoRA parameters are updated during training
for name, module in unet.named_modules():
    if isinstance(module, LoRAAttnProcessor2_0):
        for param in module.parameters():
            print(f"Setting requires_grad for {name}.{param.name} to True")
            param.requires_grad = True

for name, param in unet.named_parameters():
    if param.requires_grad:
        print(name)
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import lpips

# LPIPS metric (must install via: pip install lpips)
lpips_fn = lpips.LPIPS(net='alex').to(device)

# Store values for plotting
train_losses = []
ssim_scores = []
lpips_scores = []

num_epochs = 10 # 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for step, (o_img, g_img) in enumerate(tqdm(train_loader)):
        try:
            # Move input/output to device
            g_img = g_img.to(device, dtype=torch.bfloat16)
            o_img = o_img.to(device, dtype=torch.bfloat16)

            # Generate text embeddings from prompt
            prompt = ["ghibli illustration"]
            input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device)
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]
                text_embeddings = text_embeddings.repeat(g_img.size(0), 1, 1)

            # Encode target image to latent space
            with torch.no_grad():
                latents = vae.encode(g_img.to(device, dtype=torch.bfloat16)).latent_dist.sample()
                latents = latents * 0.18215
                latents = torch.clamp(latents, -1.0, 1.0)

            # Add noise for denoising training objective
            noise = torch.randn_like(latents)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, t)

            # Predict noise using U-Net
            output = unet(noisy_latents, t, encoder_hidden_states=text_embeddings)
            noise_pred = output.sample

            # Compute loss between predicted and actual noise
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + 1e-6

            # Skip invalid loss values
            if not torch.isfinite(loss):
                print(f"Step {step}: Skipped due to NaN loss ({loss.item()})")
                continue
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at step {step}. Skipping batch.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)
    
    

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")


pipe.unet = unet

# plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import lpips

def evaluate_on_validation(pipe, val_dataset, lpips_fn, device):
    ssim_scores = []
    lpips_scores = []

    for idx in range(len(val_dataset)):
        input_img, gt_img = val_dataset[idx]  # both are tensors from Dataset

        # Convert input tensor to PIL for the pipeline
        input_pil = transforms.ToPILImage()(input_img.cpu()).convert("RGB")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = pipe(
                prompt="ghibli illustration",
                image=input_pil,
                strength=0.4,
                guidance_scale=6.0
            ).images[0]

        # Convert both GT and prediction to tensors
        gt_tensor = gt_img.unsqueeze(0).to(device)  # already tensor from dataset
        pred_tensor = transforms.ToTensor()(pred).unsqueeze(0).to(device)  # pred is PIL

        # Resize if shapes mismatch
        if pred_tensor.shape[-1] != gt_tensor.shape[-1]:
            pred_tensor = torch.nn.functional.interpolate(pred_tensor, size=gt_tensor.shape[-2:], mode='bilinear')

        # Compute SSIM
        ssim_score = ssim(
            gt_tensor.squeeze().permute(1, 2, 0).cpu().numpy(),
            pred_tensor.squeeze().permute(1, 2, 0).cpu().numpy(),
            channel_axis=2,
            data_range=1.0
        )
        ssim_scores.append(ssim_score)

        # Compute LPIPS
        lp = lpips_fn(gt_tensor, pred_tensor)
        lpips_scores.append(lp.item())

    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    avg_lpips = sum(lpips_scores) / len(lpips_scores)

    return avg_ssim, avg_lpips


ssim_val, lpips_val = evaluate_on_validation(pipe, val_dataset, lpips_fn, device)
print(f"Epoch {epoch+1}: SSIM = {ssim_val:.4f}, LPIPS = {lpips_val:.4f}")

# =============================
# Save Fine-Tuned LoRA Weights
# =============================

# Save trained LoRA weights only if main process
if accelerator.is_main_process:
    try:
        unet.save_attn_procs("ghibli_lora_finetune")
        print("LoRA weights saved successfully")
    except Exception as e:
        print(f"\n[ERROR] Could not save LoRA weights: {e}")

  # Load and preprocess test image
sample_path = "ghibli-illustration-generated/1000/o.png"
image = Image.open(sample_path).convert("RGB").resize((512, 512))

# Generate image using fine-tuned pipeline
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = pipe(prompt="ghibli illustration", image=image, strength=0.6, guidance_scale=7.5)

# Plot the generated image
output_image = output.images[0]
output_image.save("ghibli_generated_image.png")

unet

sample_path = "ghibli-illustration-generated/1000/o.png"  # <- Update this to an actual image path
image = Image.open(sample_path).convert("RGB").resize((512, 512))
image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device, dtype=torch.bfloat16)

# Prompt for conditioning
prompt = "ghibli illustration"

# Generate image
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = pipe(prompt=prompt, image=image, strength=0.6, guidance_scale=7.5)

output_image = output.images[0]
output_image.save("output_ghibli_gen.png")
print("Prediction saved as output_ghibli.png")

import matplotlib.pyplot as plt
# =============================
# Compare original input, ground truth, and generated output
val_rooth_dir = "ghibli-illustration-generated/1010" #15
val_img_path = val_rooth_dir +"/o.png"  # Update this path
input_image = Image.open(val_img_path).convert("RGB").resize((512, 512))

# Generate image
with torch.autocast("cuda", dtype=torch.bfloat16):
    #output = pipe(prompt="ghibli style", image=input_image, strength=0.75, guidance_scale=7.5)
    #output = pipe(prompt="ghibli illustration", image=input_image, strength=0.5, guidance_scale=4.0)
    output = pipe(prompt="ghibli illustration", image=input_image, strength=0.4, guidance_scale=6.0)


# Show or save result
output_image = output.images[0]
output_image.save("output_ghibli_from_memory.png")

# Optional: Show 3-panel image comparison
gt_path = val_rooth_dir+"/g.png"  # Update with ground truth
gt_image = Image.open(gt_path).convert("RGB").resize((512, 512))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(input_image)
axs[0].set_title("Original")
axs[1].imshow(gt_image)
axs[1].set_title("Ground Truth")
axs[2].imshow(output_image)
axs[2].set_title("Predicted (Ghibli)")
for ax in axs: ax.axis("off")
plt.tight_layout()
plt.show()

def unnormalize(tensor):
    """Reverse [-1, 1] normalization back to [0, 1] for visualization."""
    return tensor * 0.5 + 0.5

def visualize_random_predictions(pipe, val_dataset, device, num_samples=5):
    import matplotlib.pyplot as plt
    import random
    from torchvision.transforms.functional import to_pil_image

    indices = random.sample(range(len(val_dataset)), num_samples)

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, idx in enumerate(indices):
        # Get input and GT, unnormalize to view properly
        input_tensor, gt_tensor = val_dataset[idx]
        input_img = unnormalize(input_tensor).clamp(0, 1)
        gt_img = unnormalize(gt_tensor).clamp(0, 1)

        input_pil = to_pil_image(input_img)
        gt_pil = to_pil_image(gt_img)

        # Inference
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(prompt="ghibli illustration", image=input_pil, strength=0.45, guidance_scale=6.0)
        pred_pil = output.images[0]

        # Plot the images
        axs[i, 0].imshow(input_pil)
        axs[i, 0].set_title(f"Original (Input) - {idx}")

        axs[i, 1].imshow(gt_pil)
        axs[i, 1].set_title("Ground Truth")

        axs[i, 2].imshow(pred_pil)
        axs[i, 2].set_title("Predicted (Ghibli)")

        for j in range(3):
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()

visualize_random_predictions(pipe, val_dataset, device, num_samples=5)
