import sys
import warnings
import torch
warnings.filterwarnings("ignore")

sys.path.append("../../")
from train_scripts.utils import load_config, display_info, seed_everything, build_directories, cleanup

import argparse
from typing import Dict
from accelerate import Accelerator
from losses.losses import build_loss_fn
from optimizers.optimizers import build_optimizer
from optimizers.schedulers import build_scheduler
from train_scripts.trainer_ddp import Segmentation_Trainer
from architectures.build_architecture import build_seg_model
# from dataloaders.build_dataset import build_dataset, build_dataloader
from dataset.build_dataset import build_dataset, build_dataloader


##################################################################################################
def launch_experiment(config_path) -> Dict:
    """
    Builds Experiment
    Args:
        config (Dict): configuration file
    Returns:
        Dict: _description_
    """
    # load config
    config = load_config(config_path)

    # set seed
    seed_everything(config)

    # build directories set checkpoint_save_dir
    config["training_parameters"]["checkpoint_save_dir"] = build_directories(config)
    # build_directories(config)

    # build training dataset & training data loader
    train_set = build_dataset(
        # dataset_type=config["dataset_parameters"]["dataset_type"],
        # augmentations=config["dataset_parameters"]["augmentations"],
        mode='train',
        dataset=config["dataset"],
    )
    train_loader = build_dataloader(
        dataset=train_set,
        dataloader_args=config["dataset"]["train_dataloader_args"],
        config=config,
        train=True,
    )

    # build validation dataset & validataion data loader
    val_set = build_dataset(
        # dataset_type=config["dataset_parameters"]["dataset_type"],
        # augmentations=config["dataset_parameters"]["augmentations"],
        # dataset_args=config["dataset_parameters"]["val_dataset_args"],
        mode='val',
        dataset=config["dataset"],
    )
    val_loader = build_dataloader(
        dataset=val_set,
        dataloader_args=config["dataset"]["val_dataloader_args"],
        config=config,
        train=False,
    )

    # build the Model
    model = build_seg_model(config)

    # set up the loss function
    criterion = build_loss_fn(
        loss_type=config["loss_fn"]["loss_type"],
        loss_args=config["loss_fn"]["loss_args"],
    )

    # set up the optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_type=config["optimizer"]["optimizer_type"],
        optimizer_args=config["optimizer"]["optimizer_args"],
    )

    # set up schedulers
    warmup_scheduler = build_scheduler(
        optimizer=optimizer, scheduler_type="warmup_scheduler", config=config
    )
    training_scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="training_scheduler",
        config=config,
    )

    # use accelarate
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=config["training_parameters"]["grad_accumulate_steps"],
    )
    accelerator.init_trackers(
        project_name=config["project"],
        config=config,
        init_kwargs={"wandb": config["wandb_parameters"]},
    )

    # display experiment info
    display_info(config, accelerator, train_set, val_set, model)

    # convert all components to accelerate
    model = accelerator.prepare_model(model=model)
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    trainloader = accelerator.prepare_data_loader(data_loader=train_loader)
    valloader = accelerator.prepare_data_loader(data_loader=val_loader)
    warmup_scheduler = accelerator.prepare_scheduler(scheduler=warmup_scheduler)
    training_scheduler = accelerator.prepare_scheduler(scheduler=training_scheduler)

    # create a single dict to hold all parameters
    storage = {
        "model": model,
        "trainloader": trainloader,
        "valloader": valloader,
        "criterion": criterion,
        "optimizer": optimizer,
        "warmup_scheduler": warmup_scheduler,
        "training_scheduler": training_scheduler,
    }

    # set up trainer
    trainer = Segmentation_Trainer(
        config=config,
        model=storage["model"],
        optimizer=storage["optimizer"],
        criterion=storage["criterion"],
        train_dataloader=storage["trainloader"],
        val_dataloader=storage["valloader"],
        warmup_scheduler=storage["warmup_scheduler"],
        training_scheduler=storage["training_scheduler"],
        accelerator=accelerator,
    )

    try:
        # 训练逻辑
        trainer.train()
    except Exception as e:
        import traceback
        print(f"Rank {torch.distributed.get_rank()} encountered an error: {e}")
        print(traceback.format_exc())
        raise
    finally:
        cleanup()


##################################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="path to yaml config file"
    )
    args = parser.parse_args()
    launch_experiment(args.config)
