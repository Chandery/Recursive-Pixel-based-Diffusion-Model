import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from dataloader.dataloader import GetDataset
from torch.utils.data import Dataset, DataLoader
from model.FDM import FDM

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    config=config["config"]
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.hydra_path,
        filename="FDM-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=False,
    )
    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=config.hydra_path,
        filename="latest",
    )
    
    
    data_set = GetDataset(config, split="train")
    train_dataset = data_set.train_dataset
    val_dataset = data_set.val_dataset
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    
    model = FDM(config)
    
    trainer = L.Trainer(
        **config["trainer"],
        callbacks=[checkpoint_callback, checkpoint_callback_latest],
        default_root_dir=config.hydra_path
    )
    
    trainer.fit(model, train_loader, val_loader)
    
if __name__=="__main__":
    main()