import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from shared.collator import zero_pad_collator
from pose_to_segments.args import args
from pose_to_segments.data import get_dataset, PoseSegmentsDataset
from pose_to_segments.model import PoseTaggingModel


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    LOGGER = None
    if not args.no_wandb:
        LOGGER = WandbLogger(project="tangram", log_model=False, offline=False)
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)

    dataset = get_dataset(components=args.pose_components)
    split = round(0.9 * len(dataset))

    train_dataset = PoseSegmentsDataset(dataset.data[:split])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=zero_pad_collator)

    validation_dataset = PoseSegmentsDataset(dataset.data[split:])
    validation_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    # Model Arguments
    model_args = dict(
        class_weights=dataset.inverse_classes_ratio(),
        pose_dims=(num_pose_joints, num_pose_dims),
        hidden_dim=args.hidden_dim,
        encoder_depth=args.encoder_depth)

    if args.checkpoint is not None:
        model = PoseTaggingModel.load_from_checkpoint(args.checkpoint, **model_args)
    else:
        model = PoseTaggingModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs("models", exist_ok=True)

        callbacks.append(ModelCheckpoint(
            dirpath="models/" + LOGGER.experiment.id,
            filename="model",
            verbose=True,
            save_top_k=1,
            monitor='validation_loss',
            mode='min'
        ))

    trainer = pl.Trainer(
        max_epochs=100,
        logger=LOGGER,
        callbacks=callbacks,
        log_every_n_steps=1,
        gpus=args.gpus)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)
