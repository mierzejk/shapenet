# author: Justus Schock (justus.schock@rwth-aachen.de)


def train_shapenet():
    """
    Trains a single shapenet with config file from comandline arguments

    See Also
    --------
    :class:`delira.training.PyTorchNetworkTrainer`
    
    """

    import collections
    import logging
    import numpy as np
    import torch
    import torch.nn.functional as F
    from shapedata.single_shape import SingleShapeDataset
    from delira.training import PyTorchNetworkTrainer
    from ..utils import Config, L1Loss_IOD, RMSELoss
    from ..layer import HomogeneousShapeLayer
    from ..networks import SingleShapeNetwork
    from delira.logging import TrixiHandler
    from trixi.logger import PytorchVisdomLogger
    from delira.training.callbacks import ReduceLROnPlateauCallbackPyTorch
    from delira.data_loading import BaseDataManager, RandomSampler, \
        SequentialSampler
    from delira.training.train_utils import convert_batch_to_numpy_identity
    import os
    import argparse
    from sklearn.metrics import mean_squared_error
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    config = Config()

    config_dict = config(os.path.abspath(args.config))

    shapes = np.load(os.path.abspath(config_dict["layer"].pop("pca_path"))
                     )["shapes"][:config_dict["layer"].pop("num_shape_params") + 1]

# layer_cls = HomogeneousShapeLayer

    net = SingleShapeNetwork(
        HomogeneousShapeLayer, {"shapes": shapes,
                                **config_dict["layer"]},
        img_size=config_dict["data"]["img_size"],
        **config_dict["network"])

    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    if args.verbose:
        print("Number of Parameters: %d" % num_params)

    # criterions = {"L1": torch.nn.L1Loss()}
    criterions = {"L1": L1Loss_IOD()} # IOD normalization
    metrics = {"RMSE": RMSELoss()}

    mixed_prec = config_dict["training"].pop("mixed_prec", False)

    config_dict["training"]["save_path"] = os.path.abspath(
        config_dict["training"]["save_path"])

    def validation_metrics(*args):
        inpt, target = map(torch.from_numpy, args)
        mse_loss = F.mse_loss(inpt.float(), target.float(), reduction='mean')
        return torch.sqrt(mse_loss)

    def batch_to_numpy(*args, **kwargs):
        args = [_arg.detach().cpu().numpy() for _arg in args
                if isinstance(_arg, torch.Tensor)]
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.detach().cpu().numpy()
            elif isinstance(v, collections.abc.Collection) and all(map(lambda i: isinstance(i, torch.Tensor), v)):
                kwargs[k] = np.array([t.detach().cpu() if t.is_cuda else t for t in v])

        return convert_batch_to_numpy_identity(*args, **kwargs)

    trainer = PyTorchNetworkTrainer(
        net, losses=criterions, train_metrics=metrics,
        val_metrics={"RMSE": validation_metrics},
        lr_scheduler_cls=ReduceLROnPlateauCallbackPyTorch,
        lr_scheduler_params=config_dict["scheduler"],
        optimizer_cls=torch.optim.Adam,
        optimizer_params=config_dict["optimizer"],
        mixed_precision=mixed_prec,
        key_mapping={"input_images": "data"},
        **config_dict["training"],
        convert_batch_to_npy_fn = batch_to_numpy)

    if args.verbose:
        print(trainer.input_device)

        print("Load Data")
    dset_train = SingleShapeDataset(
        os.path.abspath(config_dict["data"]["train_path"]),
        config_dict["data"]["img_size"], config_dict["data"]["crop"],
        config_dict["data"]["landmark_extension_train"],
        cached=config_dict["data"]["cached"],
        rotate=config_dict["data"]["rotate_train"],
        random_offset=config_dict["data"]["offset_train"]
    )

    if config_dict["data"]["test_path"]:
        dset_val = SingleShapeDataset(
            os.path.abspath(config_dict["data"]["test_path"]),
            config_dict["data"]["img_size"], config_dict["data"]["crop"],
            config_dict["data"]["landmark_extension_test"],
            cached=config_dict["data"]["cached"],
            rotate=config_dict["data"]["rotate_test"],
            random_offset=config_dict["data"]["offset_test"]
        )

    else:
        dset_val = None

    mgr_train = BaseDataManager(
        dset_train,
        batch_size=config_dict["data"]["batch_size"],
        n_process_augmentation=config_dict["data"]["num_workers"],
        transforms=None,
        sampler_cls=RandomSampler
    )
    mgr_val = BaseDataManager(
        dset_val,
        batch_size=config_dict["data"]["batch_size"],
        n_process_augmentation=config_dict["data"]["num_workers"],
        transforms=None,
        sampler_cls=SequentialSampler
    )

    if args.verbose:
        print("Data loaded")
    if config_dict["logging"].pop("enable", False):
        logger_cls = PytorchVisdomLogger

        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                TrixiHandler(
                                    logger_cls, **config_dict["logging"])
                            ])

    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=[logging.NullHandler()])

    logger = logging.getLogger("Test Logger")
    logger.info("Start Training")

    if args.verbose:
        print("Start Training")

    trainer.train(config_dict["training"]["num_epochs"], mgr_train, mgr_val,
                  config_dict["training"]["val_score_key"],
                  val_score_mode='lowest')


if __name__ == '__main__':
    train_shapenet()
