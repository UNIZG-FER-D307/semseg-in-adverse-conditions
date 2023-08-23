import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
from models import (
    PyramidSemSeg,
    convnext_tiny_,
    convnext_base_,
    convnext_large_,
    SingleScaleSemSeg,
)
from evaluations import PseudolabelGenerator
from data import load_datasets_for_pseudolabeling, create_raw_val_transform
from config import DATAROOTS
from argparse import ArgumentParser

# Disable cuDNN
torch.backends.cudnn.enabled = False

# Dictionary mapping backbone convnext versions to functions
backbone_versions = {
    "tiny": convnext_tiny_,
    "large": convnext_large_,
    "base": convnext_base_,
}

# Dictionary mapping SwiftNet versions to classes
swiftnet_versions = {
    "ss": SingleScaleSemSeg,
    "pyr": PyramidSemSeg,
}


def main(args):
    """
    Main function for generating pseudolabels using a pre-trained semantic segmentation model.

    Args:
        args: Command-line arguments parsed using argparse.

    Returns:
        None
    """
    # Configuration dictionary
    config = {
        "backbone": backbone_versions[args.backbone_version](
            pretrained=True, high_res=True
        ),
        "ignore_id": args.ignore_id,
        "num_classes": args.num_classes,
        "output_dir": args.output_dir,
        "output_pseudo": args.output_pseudo,
        "GPUs": [int(gpu) for gpu in args.gpus],
    }

    # Instantiate the selected SwiftNet model
    model = swiftnet_versions[args.swiftnet_version](
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=args.upsample_dims,
    )

    # Load model weights from the checkpoint
    state_dict = torch.load(args.ckpt_path)["state_dict"]
    new_state = {
        k.replace("model.", ""): v for k, v in state_dict.items() if "loss" not in k
    }
    model.load_state_dict(new_state)

    # Configure the model for evaluation
    config["model"] = model
    model.eval()

    # Initialize the PseudolabelGenerator for generating pseudolabels
    pl_exp = PseudolabelGenerator(
        model,
        ignore_id=config["ignore_id"],
        num_classes=config["num_classes"],
        output_dir=config["output_pseudo"],
        min_conf=args.minimal_confidence,
    )

    # Load validation dataset for pseudolabeling
    val_loader = load_datasets_for_pseudolabeling(DATAROOTS, create_raw_val_transform())

    # Configure PyTorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=config["output_dir"],
        devices=config["GPUs"],
        accelerator="gpu",
    )

    # Perform testing using the PseudolabelGenerator
    trainer.test(pl_exp, dataloaders=val_loader)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate pseudolabels using a pre-trained semantic segmentation model."
    )

    # Define command-line arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint used for generating pseudolabels.",
    )
    parser.add_argument(
        "-bv",
        "--backbone_version",
        type=str,
        required=True,
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
        "--gpus",
        nargs="+",
        required=True,
        help="GPU indices for computation. Example: --gpus 0 1",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/pseudolabeling_logs",
        help="Directory for storing log outputs.",
    )
    parser.add_argument(
        "--output_pseudo",
        type=str,
        default="./output_pseudo_test",
        help="Directory for saving pseudolabel outputs.",
    )
    parser.add_argument(
        "--minimal_confidence",
        type=float,
        default=0.99,
        help="The hard pseudolabel will incorporate all pixels whose softmax values surpass the specified minimal confidence threshold.",
    )

    # Parse command-line arguments
    args = parser.parse_args()
    main(args)
