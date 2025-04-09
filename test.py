import os
import sys
import json
import utils
import torch
import models
import tester
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cpu")
num_workers = 0

configs_paths = [
]

states_paths = [
]

BT = False

def main():

    AUG_DATA_ROOT = ''
    BATCH_SIZE = 1
    VAL_FILES = []

    print("Device:", device)

    val_dataset = utils.SunriseDataset(AUG_DATA_ROOT, VAL_FILES)
    
    for cfg_path, state_path in tqdm(zip(configs_paths, states_paths)):
    
        with open(cfg_path, "r", encoding="utf-8") as file:
            cfg = json.load(file)
    
        MODEL_STR = cfg["model"]
        CHANNELS = cfg["channels"]

        if MODEL_STR == "unet":
            if BT:
                encoder = models.UNetEncoder(n_channels=1, channels=CHANNELS).to(device)
                model = models.PreTrainedUNet(encoder=encoder, channels=CHANNELS).to(device)
            else:
                model = models.UNet(1, output_ch=5, channels=CHANNELS)
        elif MODEL_STR == "unetpp":
            model = models.UNetPP(
                n_channels=1,
                output_ch=5,
                starting_ch=CHANNELS[0],
                level=len(CHANNELS)
            ).to(device)
        else:
            raise ValueError("wrong model in config")
        
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
        )

        output_path = os.path.dirname(state_path)

        tester.run(model, val_loader, device, output_path)


if __name__ == '__main__':
    main()
