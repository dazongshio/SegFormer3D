import monai.transforms as transforms

#######################################################################################
def build_augmentations(train: bool = True):
    if train:
        train_transform = [
            # 从图像和标签中随机裁剪出多个指定大小的子区域
            transforms.RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(96, 96, 96), num_samples=4,
                                               random_center=True, random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.30, spatial_axis=1), # 对图像和标签进行随机翻转
            transforms.RandRotated(keys=["image", "label"], prob=0.50, range_x=0.36, range_y=0.0, range_z=0.0), # 对图像和标签进行随机旋转
            transforms.RandCoarseDropoutd(keys=["image", "label"], holes=20, spatial_size=(-1, 7, 7), fill_value=0,
                                          prob=0.5), #对图像和标签进行随机粗略的孔洞丢弃，模拟数据丢失或噪声
            transforms.GibbsNoised(keys=["image"]),# 向图像中添加Gibbs噪声，通常用于模拟医疗图像中的噪声
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False), # 确保图像和标签数据是适当的类型（例如，将其转换为torch.Tensor）
        ]
        return transforms.Compose(train_transform)
    else:
        val_transform = [
            # FeTA2021_TransBTS 验证时有问题，将验证集改成统一大小了
            transforms.RandSpatialCropSamplesd(keys=["image", "label"], roi_size=(96, 96, 96), random_center=True,
                                               num_samples=1,
                                               random_size=False),
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        ]
        return transforms.Compose(val_transform)