import warnings

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
from data import load_own_dataset, create_raw_val_transform
from config import DATAROOTS
from argparse import ArgumentParser

torch.backends.cudnn.enabled = False

backbone_versions = {
    "tiny": convnext_tiny_,
    "large": convnext_large_,
    "base": convnext_base_,
}

swiftnet_versions = {
    "ss": SingleScaleSemSeg,
    "pyr": PyramidSemSeg,
}


def main(args):
    config = {
        "backbone": backbone_versions[args.backbone_version](
            pretrained=True, high_res=True
        ),
        "ignore_id": args.ignore_id,
        "num_classes": args.num_classes,
        "GPUs": ",".join(args.gpus),
        "output_dir": args.output_dir,
        "output_pred": args.output_pred,
    }

    if "," not in config["GPUs"]:
        config["GPUs"] = [int(config["GPUs"])]

    print(config["GPUs"])

    model = swiftnet_versions[args.swiftnet_version](
        backbone=config["backbone"],
        num_classes=config["num_classes"],
        upsample_dims=args.upsample_dims,
    )

    state_dict = torch.load(args.ckpt_path)["state_dict"]
    new_state = dict()
    for k, v in state_dict.items():
        if "loss" in k:
            continue
        new_state[k.replace("model.", "")] = v
    model.load_state_dict(new_state)
    config["model"] = model
    model.eval()
    pl_exp = PseudolabelGenerator(
        model,
        ignore_id=config["ignore_id"],
        num_classes=config["num_classes"],
        output_dir=config["output_pred"],
        min_conf=0,
    )
    val_loader = load_own_dataset(
        args.img_dir, create_raw_val_transform(), ext=args.ext
    )

    trainer = pl.Trainer(
        default_root_dir=config["output_dir"],
        devices=config["GPUs"],
        accelerator="gpu",
    )

    trainer.test(pl_exp, dataloaders=val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Model checkpoint for pseudolabels generation",
    )
    parser.add_argument(
        "-bv",
        "--backbone_version",
        type=str,
        required=True,
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
    parser.add_argument("--gpus", nargs="+", help="Gpu indices", required=True)
    parser.add_argument("--output_dir", type=str, default="logs/pseudolabeling_logs")
    parser.add_argument(
        "--img_dir",
        type=str,
        help="Path to directory with images or path to the single image.",
    )
    parser.add_argument("--output_pred", type=str, default="./output_pred_test")
    parser.add_argument(
        "--ext",
        help="Extension of images for prediction generation.",
        choices=["png", "jpg", "jpeg"],
        default="png",
    )
    args = parser.parse_args()
    main(args)
