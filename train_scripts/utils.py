import random
import numpy as np
import torch
import wandb
import cv2
import os
from termcolor import colored
import sys
import yaml
from typing import Dict
import torch.distributed as dist
import ast

sys.path.append("../../../")

"""
Utils File Used for Training/Validation/Testing
"""


##################################################################################################
def log_metrics(**kwargs) -> None:
    # data to be logged
    log_data = {}
    log_data.update(kwargs)

    # log the data
    wandb.log(log_data)


##################################################################################################
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar") -> None:
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


##################################################################################################
def load_checkpoint(config, model, optimizer, load_optimizer=True):
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.checkpoint_file_name, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate

    return model, optimizer


##################################################################################################
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################################################################################
def random_translate(images, dataset="cifar10"):
    """
    This function takes multiple images, and translates each image randomly by at most quarter of the image.
    """

    (N, C, H, W) = images.shape

    min_pixel = torch.min(images).item()

    new_images = []
    for i in range(images.shape[0]):
        img = images[i].numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        dx = random.randrange(-8, 9, 1)
        dy = random.randrange(-8, 9, 1)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image_trans = cv2.warpAffine(img, M, (H, W)).reshape(H, W, C)

        image_trans = np.transpose(image_trans, (2, 0, 1))  # [C,H,W]
        new_images.append(image_trans)

    new_images = torch.tensor(np.stack(new_images, axis=0), dtype=torch.float32)

    return new_images


##################################################################################################
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


##################################################################################################
def save_and_print(
        config,
        model,
        optimizer,
        epoch,
        train_loss,
        val_loss,
        accuracy,
        best_val_acc,
        save_acc: bool = True,
) -> None:
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        epoch (_type_): _description_
        train_loss (_type_): _description_
        val_loss (_type_): _description_
        accuracy (_type_): _description_
        best_val_acc (_type_): _description_
    """

    if save_acc:
        if accuracy > best_val_acc:
            # change path name based on cutoff epoch
            if epoch <= config.cutoff_epoch:
                save_path = os.path.join(
                    config.checkpoint_save_dir, "best_acc_model.pth"
                )
            else:
                save_path = os.path.join(
                    config.checkpoint_save_dir, "best_acc_model_post_cutoff.pth"
                )

            # save checkpoint and log
            save_checkpoint(model, optimizer, save_path)
            print(
                f"=> epoch -- {epoch} || train loss -- {train_loss:.4f} || val loss -- {val_loss:.4f} || val acc -- {accuracy:.4f} -- saved"
            )
        else:
            save_path = os.path.join(config.checkpoint_save_dir, "checkpoint.pth")
            save_checkpoint(model, optimizer, save_path)
            print(
                f"=> epoch -- {epoch} || train loss -- {train_loss:.4f} || val loss -- {val_loss:.4f} || val acc -- {accuracy:.4f}"
            )
    else:
        # change path name based on cutoff epoch
        if epoch <= config.cutoff_epoch:
            save_path = os.path.join(config.checkpoint_save_dir, "best_loss_model.pth")
        else:
            save_path = os.path.join(
                config.checkpoint_save_dir, "best_loss_model_post_cutoff.pth"
            )

        # save checkpoint and log
        save_checkpoint(model, optimizer, save_path)
        print(
            f"=> epoch -- {epoch} || train loss -- {train_loss:.4f} || val loss -- {val_loss:.4f}"
        )


##################################################################################################
def display_info(config, accelerator, trainset, valset, model):
    model_name = config['model']['name']
    # print experiment info
    accelerator.print(f"-------------------------------------------------------")
    accelerator.print(f"[info]: Experiment Info")
    accelerator.print(
        f"[info] ----- Project: {colored(config['project'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Group: {colored(config['wandb_parameters']['group'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Name: {colored(config['wandb_parameters']['name'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Batch Size: {colored(config['dataset']['train_dataloader_args']['batch_size'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Epochs: {colored(config['training_parameters']['num_epochs'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Loss: {colored(config['loss_fn']['loss_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Optimizer: {colored(config['optimizer']['optimizer_type'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Train Dataset Size: {colored(len(trainset), color='red')}"
    )
    accelerator.print(
        f"[info] ----- Test Dataset Size: {colored(len(valset), color='red')}"
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"[info] ----- Distributed Training: {colored('True' if torch.cuda.device_count() > 1 else 'False', color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Clases: {colored(config['model'][model_name]['num_classes'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- EMA: {colored(config['ema']['enabled'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Load From Checkpoint: {colored(config['training_parameters']['load_checkpoint']['load_full_checkpoint'], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Params: {colored(pytorch_total_params, color='red')}"
    )
    accelerator.print(f"-------------------------------------------------------")


##################################################################################################

def load_config(config_path: str) -> Dict:
    """loads the yaml config file

    Args:
        config_path (str): _description_

    Returns:
        Dict: _description_
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


##################################################################################################
# 设置随机数种子以保证程序在多次运行时得到一致的结果
def seed_everything(config) -> None:
    seed = config["training_parameters"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################################################################################
# 创建训练过程所需的检查点目录

# Todo 修改一下checkpoint
def build_directories(config: Dict) -> str:
    from datetime import datetime
    import os

    checkpoint_base_dir = config["training_parameters"]["checkpoint_save_dir"]
    checkpoint_dirs = os.path.join(checkpoint_base_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))

    # 创建必要目录
    if not os.path.exists(checkpoint_dirs):
        os.makedirs(checkpoint_dirs, exist_ok=True)
    elif os.listdir(checkpoint_dirs):  # 如果目录存在且非空
        raise ValueError(f"Checkpoint directory '{checkpoint_dirs}' already exists and is not empty.")

    return checkpoint_dirs
#
#
# def build_directories(config: Dict) -> None:
#     # create necessary directories
#     if not os.path.exists(config["training_parameters"]["checkpoint_save_dir"]):
#         os.makedirs(config["training_parameters"]["checkpoint_save_dir"])
#
#     if os.listdir(config["training_parameters"]["checkpoint_save_dir"]):
#         raise ValueError("checkpoint exits -- preventing file override -- rename file")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def convert_to_tuple(value):
    """
    将字符串形式的元组转换为实际的元组类型。

    参数:
        value (str or tuple): 输入的值，可以是字符串形式的元组，或已经是元组的值。

    返回:
        tuple: 转换后的元组。

    异常:
        ValueError: 如果转换失败或转换结果不是元组，会抛出错误。
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
            if not isinstance(value, tuple):
                raise ValueError("Converted value is not a tuple.")
        except Exception as e:
            raise ValueError(f"Error converting value: {e}")
    return value