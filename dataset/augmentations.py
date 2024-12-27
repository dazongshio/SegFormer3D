from monai.transforms import (
    AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd,
    ScaleIntensityRanged, Spacingd, ToTensord, RandAffined, RandSpatialCropSamplesd,
    RandCropByPosNegLabeld, RandShiftIntensityd, SpatialPadd, ResizeWithPadOrCropd, EnsureTyped, RandRotated, RandFlipd,
    RandCoarseDropoutd, GibbsNoised
)
import numpy as np


from train_scripts.utils import convert_to_tuple


def build_transforms(augmentations_config, mode="train", keys=["image", "label"]):
    """
    Build transformation pipeline based on dataset and mode.

    Args:
        config (dict): Configuration dictionary with keys like 'dataset', 'augmentations', etc.
        mode (str): Mode of transformation ('train', 'val', 'test').

    Returns:
        Compose: A MONAI Compose object with the transformations applied.
    """

    keys = augmentations_config['keys']
    # dataset = augmentations_config['dataset']
    crop_samples = augmentations_config['crop_sample']

    pixdim = augmentations_config['pixdim']
    pixdim = convert_to_tuple(pixdim)
    spatial_size = augmentations_config['spatial_size']
    spatial_size = convert_to_tuple(spatial_size)

    a_min = augmentations_config['a_min']
    a_max = augmentations_config['a_max']

    # Base transformations for all modes
    base_transforms = []

    # base_transforms.extend([
    #     # 根据给定的目标像素间距（pixdim）对图像进行重新采样
    #     Spacingd(keys=keys if mode != "test" else ["image"], pixdim=pixdim,
    #              mode=("bilinear", "nearest") if mode != "test" else ("bilinear")),
    #     # 该操作用于对图像的强度进行缩放（归一化），将图像的像素值调整到指定的范围
    #     ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    #     # 裁剪图像的前景部分，去掉图像中无关的背景区域。这在医学图像处理中非常常见，通常用于去除背景噪声或无关区域
    #     CropForegroundd(keys=keys if mode != "test" else ["image"], source_key="image"),
    # ])

    # Mode-specific augmentations
    if mode == "train":
        train_transforms = [
            # 从图像和标签中随机裁剪出多个指定大小的子区域
            RandSpatialCropSamplesd(keys=keys, roi_size=spatial_size, num_samples=crop_samples,
                                    random_center=True, random_size=False),
            # 从图像和标签中随机裁剪样本，裁剪区域包含指定数量的正负样本。
            RandFlipd(keys=keys, prob=0.30, spatial_axis=1),  # 对图像和标签进行随机翻转
            RandRotated(keys=keys, prob=0.50, range_x=0.36, range_y=0.0, range_z=0.0),  # 对图像和标签进行随机旋转
            RandCoarseDropoutd(keys=keys, holes=20, spatial_size=(-1, 7, 7), fill_value=0,
                               prob=0.5),  # 对图像和标签进行随机粗略的孔洞丢弃，模拟数据丢失或噪声
            GibbsNoised(keys=["image"]),  # 向图像中添加Gibbs噪声，通常用于模拟医疗图像中的噪声
        ]
        base_transforms.extend(train_transforms)
    elif mode == "val":
        val_transforms = [
            # 从图像和标签中随机裁剪出多个指定大小的子区域
            # RandSpatialCropSamplesd(keys=keys, roi_size=spatial_size, num_samples=crop_samples,
            #                         random_center=True, random_size=False),
            # RandCropByPosNegLabeld(
            #     keys=keys, label_key="label", spatial_size=spatial_size,
            #     pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0
            # ),
        ]
        base_transforms.extend(val_transforms)
    elif mode == "test":
        test_transforms = [
        ]
        base_transforms.extend(test_transforms)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    base_transforms.extend(
        # 调整图像和标签的大小到指定的空间大小
        [
            # ResizeWithPadOrCropd(keys=keys if mode != "test" else ["image"], spatial_size=spatial_size,
            #                      mode=("constant"), method="end"),
            EnsureTyped(keys=keys, track_meta=False), ])

    return Compose(base_transforms)


def build_augmentations(mode: str = 'train', augmentations_config=None):
    """
    Wrapper function to build training, validation, and testing transformations.

    Args:
        args: Namespace or dictionary with 'dataset', 'mode', and other configuration.

    Returns:
        Tuple[Compose, Compose] or Compose: Training and validation transforms, or test transform.
    """

    if mode == 'train':
        # print(f"Building training transforms for dataset: {augmentations_config['dataset']}")
        train_transforms = build_transforms(augmentations_config, mode="train")
        return train_transforms
    elif mode == 'val':
        # print(f"Building validation transforms for dataset: {augmentations_config['dataset']}")
        test_transforms = build_transforms(augmentations_config, mode="val")
        return test_transforms
    else:
        # print(f"Building training transforms for dataset: {augmentations_config['dataset']}")
        train_transforms = build_transforms(augmentations_config, mode="test")
        return train_transforms

# 从图像和标签中随机裁剪出多个指定大小的子区域
# RandSpatialCropSamplesd(keys=keys, roi_size=(96, 96, 96), num_samples=4,
#                         random_center=True, random_size=False),
# RandFlipd(keys=keys, prob=0.30, spatial_axis=1),  # 对图像和标签进行随机翻转
# RandRotated(keys=keys, prob=0.50, range_x=0.36, range_y=0.0, range_z=0.0),  # 对图像和标签进行随机旋转
# RandCoarseDropoutd(keys=keys, holes=20, spatial_size=(-1, 7, 7), fill_value=0,
#                    prob=0.5),  # 对图像和标签进行随机粗略的孔洞丢弃，模拟数据丢失或噪声
# GibbsNoised(keys=["image"]),  # 向图像中添加Gibbs噪声，通常用于模拟医疗图像中的噪声
# EnsureTyped(keys=keys, track_meta=False),  # 确保图像和标签数据是适当的类型（例如，将其转换为torch.Tensor）

# RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=spatial_size,
#     pos=1, neg=1, num_samples=crop_samples, image_key="image", image_threshold=0),
# 对图像的强度进行随机偏移。通过在指定的范围内调整像素值，可以模拟不同的成像条件和噪声。
# RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
# 对图像进行随机仿射变换。仿射变换包括旋转、缩放、平移等操作
# RandAffined(keys=keys, mode=("bilinear", "nearest"), prob=1.0,
#     spatial_size=spatial_size, rotate_range=(0, 0, np.pi / 15), scale_range=(0.1, 0.1, 0.1)),


# LoadImaged(keys=keys if mode != "test" else ["image"]), # 从文件系统加载图像
# AddChanneld(keys=keys if mode != "test" else ["image"]),# 添加一个新的维度（通常是通道维度）到图像数据中
# Orientationd(keys=keys if mode != "test" else ["image"], axcodes="RAS"), # 调整图像的空间方向（即对图像进行重定向或重排列），使其符合指定的空间坐标轴顺序。常见的空间坐标系有 RAS（Right-Anterior-Superior），LPS（Left-Posterior-Superior）等。
