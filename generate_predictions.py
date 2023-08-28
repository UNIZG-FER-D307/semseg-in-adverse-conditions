import warnings

warnings.filterwarnings("ignore")
import copy
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn

from config import DATAROOTS
from data import create_raw_val_transform, load_own_dataset
from evaluations import PseudolabelGenerator
from models import (
    PyramidSemSeg,
    SingleScaleSemSeg,
    convnext_base_,
    convnext_large_,
    convnext_tiny_,
)

# Disable cuDNN
torch.backends.cudnn.enabled = False

# Dictionary mapping convnext backbone versions to functions
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


class SimpleEnsemble(nn.Module):
    def __init__(self, template_model, weights):
        super(SimpleEnsemble, self).__init__()

        self.models = nn.ModuleList([])
        for w in weights:
            m = copy.deepcopy(template_model)
            m.load_state_dict(w, strict=True)
            m.eval()
            self.models.append(m)

    def forward(self, x):
        with torch.no_grad():
            return torch.cat([m(x).unsqueeze(0) for m in self.models], 0).mean(0)


def load_model(args, config):
    template_model = swiftnet_versions[args.swiftnet_version](
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=args.upsample_dims,
    )
    # single model
    if len(args.ckpt_path) == 1:
        print("Loading single model...")
        # Load model weights from the checkpoint
        state_dict = torch.load(args.ckpt_path[0])["state_dict"]
        new_state = {
            k.replace("model.", ""): v for k, v in state_dict.items() if "loss" not in k
        }
        template_model.load_state_dict(new_state)

        # Configure the model for evaluation
        config["model"] = template_model
        template_model.eval()
    # ensemble
    else:
        print(f"Loading ensemble of {len(args.ckpt_path)} models...")

        def remap_keys(d):
            new_state = dict()
            for k, v in d.items():
                if "loss" in k:
                    continue
                new_state[k.replace("model.", "")] = v
            return new_state

        weights = [
            remap_keys(torch.load(file, map_location="cpu")["state_dict"])
            for file in args.ckpt_path
        ]
        model = SimpleEnsemble(template_model=template_model, weights=weights)
        config["model"] = model

    return config["model"]


def main(args):
    """
    Main function for generating predictions using a pre-trained semantic segmentation model.

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
        "output_pred": args.output_pred,
        "GPUs": [int(gpu) for gpu in args.gpus],
    }

    # Print the selected GPUs
    print("Selected GPUs:", config["GPUs"])

    # Instantiate the selected SwiftNet model
    model = load_model(args=args, config=config)

    # Initialize the PseudolabelGenerator for generating predictions
    pl_exp = PseudolabelGenerator(
        model,
        ignore_id=config["ignore_id"],
        num_classes=config["num_classes"],
        output_dir=config["output_pred"],
        min_conf=0,
        multiscale=args.multiscale,
    )

    # Load validation dataset
    val_loader = load_own_dataset(
        args.img_dir, create_raw_val_transform(), ext=args.ext
    )

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
        description="Generate predictions using a pre-trained semantic segmentation model."
    )

    # Define command-line arguments
    parser.add_argument(
        "--ckpt_path",
        nargs="+",
        required=True,
        help="Specifies the path to the checkpoint utilized for prediction generation.",
    )

    parser.add_argument(
        "-bv",
        "--backbone_version",
        type=str,
        required=True,
        choices=["tiny", "base", "large"],
        help="Specifies the ConvNeXT backbone version to utilize, offering choices of 'tiny', 'base', or 'large'. This selection must align with the version of saved checkpoint.",
    )

    parser.add_argument(
        "-sv",
        "--swiftnet_version",
        type=str,
        default="pyr",
        choices=["pyr", "ss"],
        help="Determines the SwiftNet version to utilize, with options being 'pyr' (pyramid) or 'ss' (singlescale), influencing the choice of architecture for processing. This selection must align with the version of saved checkpoint.",
    )

    parser.add_argument(
        "--upsample_dims",
        type=int,
        default=256,
        help="Specifies the dimension of the upsampling path in the network.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=19,
        help="Defines the total number of classes. For the cityscapes taxonomy, the class count is 19.",
    )
    parser.add_argument(
        "--ignore_id",
        type=int,
        default=255,
        help="Indicates the class ID to be ignored during processing.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="Specifies GPU indices for computation. For example, to use GPUs 0 and 1, provide --gpus 0 1.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/pseudolabeling_logs",
        help="Specifies the directory for log outputs.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        help="Specifies the path to a directory containing images or to a single image file.",
    )
    parser.add_argument(
        "--output_pred",
        type=str,
        default="./output_pred_test",
        help="Designates the directory for saving prediction outputs.",
    )
    parser.add_argument(
        "--ext",
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Specifies the extension/format of images used for inference.",
    )
    parser.add_argument(
        "--multiscale", action="store_true", help="Multi scale inference"
    )

    args = parser.parse_args()
    main(args)
