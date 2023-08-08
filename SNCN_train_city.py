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
    convnext_base_,
    convnext_large_,
    PyramidSemSeg,
    SingleScaleSemSeg,
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
from argparse import ArgumentParser
from datetime import datetime

backbone_versions = {
    "tiny": convnext_tiny_,
    "base": convnext_base_,
    "large": convnext_large_,
}

swiftnet_versions = {
    "ss": SingleScaleSemSeg,
    "pyr": PyramidSemSeg,
}


def main(args):
    # example if you want specific seed
    # pl.seed_everything(123, workers=True)
    # fire on all cylinders
    torch.backends.cudnn.enabled = False
    config = {
        "backbone": backbone_versions[args.backbone_version](
            pretrained=True, high_res=True
        ),
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "crop_size": (args.crop_height, args.crop_width),
        "jitter_range": (args.sjl, args.sju),
        "max_epochs": args.max_epochs,
        "ignore_id": args.ignore_id,
        "num_classes": args.num_classes,
        "loss": args.loss_type,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "precision": args.precision,
        "GPUs": ",".join(args.gpus),
        "upsample_dims": args.upsample_dims,
        "output_dir": f"logs/city/CN_SN_{args.swiftnet_version}_{args.backbone_version}_{args.precision}_{args.loss_type}_{args.batch_size_per_gpu}_{(args.crop_height, args.crop_width)}_{(args.sjl, args.sju)}_{args.upsample_dims}_{args.max_epochs}_{args.gpus}_{str(datetime.now())}",
    }

    if "," not in config["GPUs"]:
        config["GPUs"] = [int(config["GPUs"])]

    model = swiftnet_versions[args.swiftnet_version](
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=config["upsample_dims"],
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
        save_top_k=2,
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
    parser = ArgumentParser()
    parser.add_argument(
        "-bv",
        "--backbone_version",
        type=str,
        default="tiny",
        choices=["tiny", "base", "large"],
    )
    parser.add_argument(
        "-sv",
        "--swiftnet_version",
        type=str,
        default="pyr",
        choices=["pyr", "ss"],
        help="Wheather to use singlescale or pyramid swiftnet",
    )
    parser.add_argument("--upsample_dims", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--ignore_id", type=int, default=255)
    # focal loss or cross entropy loss
    parser.add_argument("--loss_type", type=str, choices=["FL", "CE"], default="FL")
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=1e-7)
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32])
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--gpus", nargs="+", help="Gpu indices", required=True)
    parser.add_argument("--crop_height", type=int, default=1024)
    parser.add_argument("--crop_width", type=int, default=1024)
    parser.add_argument(
        "--sjl", help="scale jitter lower bound", default=0.5, type=float
    )
    parser.add_argument(
        "--sju", help="scale jitter upper bound", default=2.0, type=float
    )

    args = parser.parse_args()
    main(args)
