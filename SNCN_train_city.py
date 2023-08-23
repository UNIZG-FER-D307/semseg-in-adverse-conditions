import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import torch.optim as optim
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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

# Dictionary mapping backbone versions to functions
backbone_versions = {
    "tiny": convnext_tiny_,
    "base": convnext_base_,
    "large": convnext_large_,
}

# Dictionary mapping SwiftNet versions to classes
swiftnet_versions = {
    "ss": SingleScaleSemSeg,
    "pyr": PyramidSemSeg,
}


def main(args):
    """
    Main function for training a semantic segmentation model with boundary-aware enhancement.

    Args:
        args: Command-line arguments parsed using argparse.

    Returns:
        None
    """
    # Disable cuDNN
    torch.backends.cudnn.enabled = False

    # Configuration dictionary
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
        "upsample_dims": args.upsample_dims,
        "GPUs": [int(gpu) for gpu in args.gpus],
        "output_dir": f"logs/city/CN_SN_{args.swiftnet_version}_{args.backbone_version}_{args.precision}_{args.loss_type}_{args.batch_size_per_gpu}_{(args.crop_height, args.crop_width)}_{(args.sjl, args.sju)}_{args.upsample_dims}_{args.max_epochs}_{args.gpus}_{str(datetime.now())}",
    }

    # Instantiate the selected SwiftNet model
    model = swiftnet_versions[args.swiftnet_version](
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=config["upsample_dims"],
    )
    backbone_params, upsample_params = model.prepare_optim_params()
    lr_backbone = config["lr"] / 4.0

    # Define optimizer
    optimizer = optim.Adam(
        [{"params": backbone_params, "lr": lr_backbone}, {"params": upsample_params}],
        lr=config["lr"],
        weight_decay=1e-4,
        eps=1e-7,
    )

    config["optimizer"] = optimizer

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"], eta_min=config["lr_min"]
    )
    config["scheduler"] = scheduler
    config["model"] = model

    # Initialize the BoundaryAwareSemseg experiment
    pl_exp = BoundaryAwareSemseg(
        model,
        optimizer,
        scheduler,
        config["ignore_id"],
        config["num_classes"],
        loss_type=config["loss"],
    )

    # Create training and validation transforms
    train_transform = create_jitter_random_crop_BA_transform(
        crop_size=config["crop_size"],
        scale=config["jitter_range"],
        input_mean=(0.0),
        num_classes=config["num_classes"],
        ignore_id=config["ignore_id"],
        mean=cityscapes_mean,
        std=cityscapes_std,
    )

    # Load training and validation datasets
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

    # Define ModelCheckpoint callback for saving best models
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mIoU",
        dirpath=config["output_dir"],
        filename="model-{epoch:02d}-{val_mIoU:.2f}",
        save_top_k=2,
        mode="max",
        save_last=True,
    )

    # Create CSV logger for logging metrics
    csv_logger = pl_loggers.CSVLogger(output_dir)

    # Configure PyTorch Lightning Trainer
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

    # Start model training
    trainer.fit(pl_exp, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train a semantic segmentation model with boundary-aware enhancement."
    )

    # Define command-line arguments
    parser.add_argument(
        "-bv",
        "--backbone_version",
        type=str,
        default="tiny",
        choices=["tiny", "base", "large"],
        help="Select the ConvNeXT backbone version: 'tiny', 'base', or 'large'.",
    )
    parser.add_argument(
        "-sv",
        "--swiftnet_version",
        type=str,
        default="pyr",
        choices=["pyr", "ss"],
        help="Select the SwiftNet version: 'pyr' (pyramid) or 'ss' (single scale).",
    )
    parser.add_argument(
        "--upsample_dims",
        type=int,
        default=256,
        help="Dimension of the upsampling path in the network.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        help="Number of classes in the dataset. For the cityscapes taxonomy, the class count is 19.",
    )
    parser.add_argument(
        "--ignore_id",
        type=int,
        default=255,
        help="Class ID to be ignored during processing.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["FL", "CE"],
        default="FL",
        help="Loss type for training: 'FL' (focal loss) or 'CE' (cross entropy loss).",
    )
    parser.add_argument(
        "--lr", type=float, default=4e-4, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-7,
        help="Minimum learning rate for the learning rate scheduler.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        choices=[16, 32],
        default=16,
        help="Floating-point precision for training: 16 or 32 bits.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=2,
        help="Batch size per GPU for training.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="Indices of GPUs to use for computation. Example: --gpus 0 1",
    )
    parser.add_argument(
        "--crop_height",
        type=int,
        default=1024,
        help="Height of the cropped input image.",
    )
    parser.add_argument(
        "--crop_width", type=int, default=1024, help="Width of the cropped input image."
    )
    parser.add_argument(
        "--sjl",
        default=0.5,
        type=float,
        help="Lower bound of scale jitter applied to input images during training.",
    )
    parser.add_argument(
        "--sju",
        default=2.0,
        type=float,
        help="Upper bound of scale jitter applied to input images during training.",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
