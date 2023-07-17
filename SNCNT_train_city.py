import warnings

warnings.filterwarnings("ignore")
import torch.optim as optim
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning import loggers as pl_loggers
from models import (
    convnext_tiny_,
    PyramidSemSeg,
)
from experiments import BoundaryAwareSemseg
from data import (
    load_city_train,
    load_city_val,
    create_jitter_random_crop_BA_transform,
    create_basic_BA_transform,
    cityscapes_mean,
    cityscapes_std,
)
from config import CITYSCAPES_ROOT


def main():
    # pl.seed_everything(123, workers=True)
    # fire on all cylinders
    torch.backends.cudnn.enabled = False
    config = {
        "backbone": convnext_tiny_(pretrained=True, high_res=True),
        "batch_size_per_gpu": 1,
        "crop_size": (1024, 1024),
        "jitter_range": (0.5, 2),
        "max_epochs": 50,
        "ignore_id": 255,
        "num_classes": 19,
        "loss": "FL",
        "lr": 4e-4,
        "lr_min": 1e-7,
        "precision": 16,
        "GPUs": [0, 1],
    }

    config[
        "output_dir"
    ] = f"logs/city/Convnext_SN_tiny_epochs={config['max_epochs']}_precision={config['precision']}_crop_size={config['crop_size']}_jitter={config['jitter_range']}_bspg={config['batch_size_per_gpu']}_gpus={config['GPUs']}"

    model = PyramidSemSeg(
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=256,
    )
    backbone_params, upsample_params = model.prepare_optim_params()
    lr_backbone = config["lr"] / 4.0
    optimizer = optim.Adam(
        [{"params": backbone_params, "lr": lr_backbone}, {"params": upsample_params}],
        lr=config["lr"],
        weight_decay=1e-4,
        eps=1e-7,
    )

    config["optimizer"] = optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"], eta_min=config["lr_min"]
    )
    config["scheduler"] = scheduler
    config["model"] = model
    pl_exp = BoundaryAwareSemseg(
        model,
        optimizer,
        scheduler,
        config["ignore_id"],
        config["num_classes"],
        loss_type=config["loss"],
    )

    train_transform = create_jitter_random_crop_BA_transform(
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        input_mean=(0.0),
        num_classes=config["num_classes"],
        ignore_id=config["ignore_id"],
        mean=cityscapes_mean,
        std=cityscapes_std,
    )

    train_loader = load_city_train(
        CITYSCAPES_ROOT,
        train_transforms=train_transform,
        bs=config["batch_size_per_gpu"],
    )
    val_loader = load_city_val(
        CITYSCAPES_ROOT,
        create_basic_BA_transform(
            cityscapes_mean, cityscapes_std, config["num_classes"], config["ignore_id"]
        ),
    )

    output_dir = config["output_dir"]
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU",
        dirpath=config["output_dir"],
        filename="model-{epoch:02d}-{val_mIoU:.2f}",
        save_top_k=5,
        mode="max",
        save_last=True,
    )

    csv_logger = pl_loggers.CSVLogger(output_dir)

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        default_root_dir=output_dir,
        devices=config["GPUs"],
        callbacks=[lr_monitor, checkpoint_callback],
        precision=config["precision"],
        logger=csv_logger,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(pl_exp, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
