import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


#数据预处理
tf = transforms.Compose(
    [transforms.Resize((224, 224)), 
    # [transforms.RandomCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ]
)


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        # print(self.classes)
        self.transform = tf
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label

    def _load_data(self):
        data = []

        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                data.append((img_path, class_idx))
        return data

# Example usage:
if __name__ == '__main__':
    train_dataset = MyDataset(root_dir='img/train', transform=tf)
    val_dataset = MyDataset(root_dir='img/val', transform=tf)
    for i in range(1):
        img,label = train_dataset[i]
        print(f"img:{img}",f"label:{label}")
        # print(type(img))
