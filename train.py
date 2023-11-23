import argparse
import collections
import warnings

import numpy as np
import torch

import lib.loss as module_loss
import lib.metric as module_metric
import lib.model as module_arch
import lib.postprocessing as module_postprocessing
import lib.spectrogram_decoder as module_spectrogram_decoder
import lib.text_encoder as module_text_encoder
from lib.config_processing.parse_config import ConfigParser
from lib.text_encoder.base_encoder import BaseTextEncoder
from lib.trainer import Trainer
from lib.utils import prepare_device
from lib.utils.object_loading import get_dataloaders


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    sampling_rate = config["common"]["sr"]

    # text encoder
    text_encoder: BaseTextEncoder = config.init_obj(config['text_encoder'], module_text_encoder)

    # setup data_loader instances
    dataloaders = get_dataloaders(config, encoder=text_encoder)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config.config.get("metrics", [])
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    if "lr_scheduler" in config.config:
        lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    if "postprocessor" in config.config:
        postprocessor = config.init_obj(config["postprocessor"], module_postprocessing)
    else:
        postprocessor = None

    if "spectrogram_decoder" in config.config:
        spectrogram_decoder = config.init_obj(config["spectrogram_decoder"], module_spectrogram_decoder,
                                              device=device, sampling_rate=sampling_rate)
    else:
        spectrogram_decoder = None

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        spectrogram_decoder=spectrogram_decoder,
        postprocessor=postprocessor,
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="TSS trainer")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
