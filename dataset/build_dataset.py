from monai.data import pad_list_data_collate
from typing import Dict
from monai.data import DataLoader
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from dataset.augmentations import build_augmentations


######################################################################
def build_dataset(mode:str,dataset: Dict):
    all_dataset = {'BraTS2021', 'AMOS2022', 'FeTA2021', 'FLARE2022', 'BraTS2017'}
    dataset_name = dataset['name']

    # 检查输入的 dataset_type 是否以支持的前缀开头（不区分大小写）
    if not any(dataset_name.startswith(prefix) for prefix in all_dataset):
        raise ValueError(f"Only datasets starting with one of the following: {', '.join(all_dataset)} are supported. "
                         f"Your input '{dataset_name}' is not supported.")

    dataset = SegDataset(
        dataset_name = dataset_name,
        root_dir=dataset[dataset_name][mode]["root"],
        mode=mode,
        transform=build_augmentations(augmentations_config=dataset[dataset_name]["augmentations"], mode=mode),
        fold_id=dataset[dataset_name][mode]["fold_id"],
    )
    return dataset

######################################################################
def build_dataloader(
        dataset, dataloader_args: Dict, config: Dict = None, train: bool = True) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )
    return dataloader


class SegDataset(Dataset):
    def __init__(
            self, dataset_name: str, root_dir: str, mode: str = 'train', transform=None, fold_id: int = None
    ):
        """
        root_dir: path to (BraTS_Training_Data) folder
        is_train: whether or nor it is train or validation
        transform: composition of the pytorch transforms
        fold_id: fold index in kfold dataheld out
        """
        super().__init__()
        if fold_id is not None:
            csv_name = (
                f"train_fold_{fold_id}{dataset_name}.csv"
                if mode!='train'
                else f"validation_fold_{fold_id}{dataset_name}.csv"
            )
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)
        else:
            csv_name = f"train{dataset_name}.csv" if mode=='train' else f"validation{dataset_name}.csv"
            csv_fp = os.path.join(root_dir, csv_name)
            assert os.path.exists(csv_fp)

        self.csv = pd.read_csv(csv_fp)
        self.transform = transform

    def __len__(self):
        return self.csv.__len__()

    def __getitem__(self, idx):
        data_path = self.csv["data_path"][idx]
        case_name = self.csv["case_name"][idx]
        # e.g, BRATS_001_modalities.pt
        # e.g, BRATS_001_label.pt
        volume_fp = os.path.join(data_path, f"{case_name}_modality.pt")
        label_fp = os.path.join(data_path, f"{case_name}_label.pt")

        try:
            volume = torch.load(volume_fp)
        except FileNotFoundError:
            # 捕获文件未找到错误
            # print(f"File {volume_fp} not found, trying with a different name...")
            # 如果找不到文件，使用不同的文件名
            volume_fp = os.path.join(data_path, f"{case_name}_modalities.pt")
            try:
                volume = torch.load(volume_fp)
            except FileNotFoundError:
                # 如果第二个文件也找不到，给出错误提示
                print(f"Both {f'{case_name}_modality.pt'} and {f'{case_name}_modalities.pt'} are not found.")
        # load the preprocessed tensors
        # volume = torch.load(volume_fp)
        label = torch.load(label_fp)
        # print(volume.shape, label.shape)
        # volume = torch.load(volume_fp,weights_only=True)
        # label = torch.load(label_fp,weights_only=True)
        data = {"image": torch.from_numpy(volume).float(), "label": torch.from_numpy(label).float()}

        if self.transform:
            data = self.transform(data)

        return data