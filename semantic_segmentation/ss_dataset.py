import os
import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt

VOC_DIR = Path(__file__).resolve().parents[1] / "data" / "VOCdevkit" / "VOC2012"

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype("int32")
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    return colormap2label[idx]


def read_voc_images(voc_dir, is_train=True):
    txt_frame = os.path.join(
        VOC_DIR, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt"
    )
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_frame, "r") as f:
        images = f.read().split()

    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "JPEGImages", f"{fname}.jpg")
            )
        )
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, "SegmentationClass", f"{fname}.png"), mode
            )
        )

    return features, labels


def show_images(features, labels, num_images=5):
    """显示特征图像和对应的标签图像"""
    fig, axes = plt.subplots(2, num_images, figsize=(10, 6))

    if num_images == 1:
        axes = axes.reshape(2, 1)

    for i in range(num_images):
        # 处理特征图像
        feature_img = features[i]
        # 检查第一维是否是3（C, H, W格式）
        if feature_img.shape[0] == 3:
            # 如果是tensor就转numpy，然后permute
            if isinstance(feature_img, torch.Tensor):
                feature_img = feature_img.permute(1, 2, 0).numpy()
            else:
                feature_img = feature_img.transpose(1, 2, 0)
        else:
            # 第一维不是3，直接转numpy
            if isinstance(feature_img, torch.Tensor):
                feature_img = feature_img.numpy()

        feature_img = feature_img.astype("uint8")
        axes[0, i].imshow(feature_img)
        axes[0, i].set_title(f"Feature {i+1}")
        axes[0, i].axis("off")

        # 处理标签图像
        label_img = labels[i]
        if label_img.shape[0] == 3:
            if isinstance(label_img, torch.Tensor):
                label_img = label_img.permute(1, 2, 0).numpy()
            else:
                label_img = label_img.transpose(1, 2, 0)
        else:
            if isinstance(label_img, torch.Tensor):
                label_img = label_img.numpy()

        label_img = label_img.astype("uint8")
        axes[1, i].imshow(label_img)
        axes[1, i].set_title(f"Label {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    # in feature(an image) generate a random (height, width) image and return (top, left, crop_height, crop_width)
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


class VOCSegDataset(torch.utils.data.Dataset):

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.crop_size = crop_size
        # Train or Test
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [
            self.normalize_image(feature) for feature in self.filter(features)
        ]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print("read " + str(len(self.features)) + " examples")

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [
            img
            for img in imgs
            if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])
        ]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(
            self.features[idx], self.labels[idx], *self.crop_size
        )
        # label: [batch, channel, w, h] -> [batch, w, h]
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, VOC_DIR),
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, VOC_DIR),
        batch_size,
        drop_last=True,
        num_workers=num_workers,
    )


def test():
    train_features, train_labels = read_voc_images(VOC_DIR, True)
    print(len(train_features))
    print(train_features[0].shape)
    print(train_features[0][0][:10][:10])

    show_images(train_features, train_labels, num_images=5)
    imgs = []
    for _ in range(5):
        imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

    imgs = [img.permute(1, 2, 0) for img in imgs]
    show_images(imgs[::2], imgs[1::2], 5)
    crop_size = (320, 480)
    voc_train = VOCSegDataset(True, crop_size, VOC_DIR)
    voc_test = VOCSegDataset(False, crop_size, VOC_DIR)
    batch_size = 64
    train_iter = torch.utils.data.DataLoader(
        voc_train,
        batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    for X, Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break
