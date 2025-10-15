from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Creates a custom torch dataset based on the folder paths provided
class GhibliDataset(Dataset):
    def __init__(self, data_path, image_size = (512, 512)):
        self._data_path = data_path
        self._folder_paths = os.listdir(data_path)
        self._transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __apply_transform(self, img):
        return self._transform(img)

    def __len__(self):
        return len(self._folder_paths)

    def __getitem__(self, idx):
        folder_name = self._folder_paths[idx]
        og_image = self.__apply_transform(Image.open(os.path.join(self._data_path, folder_name, "o.png")).convert("RGB"))
        target_image = self.__apply_transform(Image.open(os.path.join(self._data_path, folder_name, "g.png")).convert("RGB"))

        return og_image, target_image

# Converts a torch Tensor to Numpy Array
def convertPTTensorToNP(img_tensor):
    return img_tensor.numpy()

# Since Matplotlib uses a different channel order, rearranging Torch Tensor to PLt order
def rearrangeTorchArrayForPlt(img_arr):
    return np.transpose(convertPTTensorToNP(img_arr), (1, 2, 0))

ghibli_data = GhibliDataset(data_path)

# First input-output pair in the dataset
ghibli_data.__getitem__(0)

convertPTTensorToNP(ghibli_data.__getitem__(0)[0])

plt.imshow(rearrangeTorchArrayForPlt(ghibli_data.__getitem__(0)[0]))
plt.axis('off')
plt.show()
