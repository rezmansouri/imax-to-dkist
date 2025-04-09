import os
import sys
import json
import utils
import torch
import shutil
import losses
import models
import trainer
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 0


def main():

    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = json.load(file)

    if len(sys.argv) > 2:
        BARLOW_TWINS = True
        BARLOW_TWINS_ENCODER_PATH = sys.argv[2]
    else:
        BARLOW_TWINS = False

    AUG_DATA_ROOT = cfg["aug_data_root"]
    ORG_DATA_ROOT = cfg["org_data_root"]
    TRAIN_FILES = cfg["train_files"]
    VAL_FILES = cfg["val_files"]
    BATCH_SIZE = cfg["batch_size"]
    LOSS_STR = cfg["loss"]
    N_EPOCHS = cfg["n_epochs"]
    MODEL_STR = cfg["model"]
    OUTPUT_PATH = cfg["output_path"]
    CHANNELS = cfg["channels"]

    if not os.path.exists(OUTPUT_PATH):
        raise ValueError("output path does not exist")

    print("Device:", device)

    train_dataset = utils.SunriseDataset(AUG_DATA_ROOT, TRAIN_FILES)

    val_dataset = utils.SunriseDataset(AUG_DATA_ROOT, VAL_FILES)

    if BARLOW_TWINS:
        print("Pre-trained")

        encoder = models.UNetEncoder(n_channels=1, channels=CHANNELS).to(device)
        best_encoder_state = torch.load(BARLOW_TWINS_ENCODER_PATH)

        encoder.load_state_dict(best_encoder_state)

    if MODEL_STR == "unet":
        if BARLOW_TWINS:
            model = models.PreTrainedUNet(encoder=encoder, channels=CHANNELS).to(device)
        else:
            model = models.UNet(n_channels=1, output_ch=5, channels=CHANNELS).to(device)
    elif MODEL_STR == "unetpp":
        model = models.UNetPP(
            n_channels=1,
            output_ch=5,
            starting_ch=CHANNELS[0],
            level=len(CHANNELS)
        ).to(device)
    else:
        raise ValueError("wrong model in config")

    print("Training")

    train_dataset.barlow = False
    val_dataset.barlow = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
    )

    loss_weights = losses.get_weights().to(device)

    print("weights:", loss_weights)

    if LOSS_STR == "focal":
        criterion = losses.FocalLoss(alpha=loss_weights).to(device)
    elif LOSS_STR == "lovasz":
        criterion = losses.lovasz_softmax
    elif LOSS_STR == "mIoU":
        criterion = losses.mIoULoss(n_classes=5, weight=loss_weights).to(device)
    else:
        raise ValueError("wrong loss in config")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    experiment_dir_name = (
        ("bt_" if BARLOW_TWINS else "")
        + MODEL_STR
        + "_"
        + cfg_path.split("/")[-1]
        + "_"
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, experiment_dir_name)

    os.mkdir(OUTPUT_PATH)
    shutil.copy2(cfg_path, OUTPUT_PATH)

    best_model_path = trainer.run(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=N_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_path=OUTPUT_PATH,
        save_model=True,
        model_summary=False,
    )

    print("Predicting using", best_model_path)

    best_state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_state)

    image_path = os.path.join(ORG_DATA_ROOT, "images", VAL_FILES[-1] + ".png")
    mask_path = os.path.join(ORG_DATA_ROOT, "masks", VAL_FILES[-1] + ".npy")

    image = np.array(Image.open(image_path), dtype=np.float32)
    mask = np.load(mask_path)
    x = image
    x /= x.max()
    y = mask.astype(np.float32)

    model.eval()

    xx = torch.unsqueeze(torch.Tensor(x), 0)
    xx = torch.unsqueeze(xx, 1).to(device)

    summed = np.zeros((5, 768, 768), dtype=np.float32)
    window_w, window_h = 128, 128
    stride = 64
    for i in range(0, summed.shape[1] - window_h + 1, stride):
        for j in range(0, summed.shape[2] - window_w + 1, stride):
            with torch.no_grad():
                out = model(xx[:, :, i : i + window_w, j : j + window_h])
                summed[:, i : i + window_w, j : j + window_h] += (
                    out.cpu().detach().numpy()[0]
                )

    yhat = np.argmax(summed, axis=0)

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.title("Input", fontsize=30)
    plt.imshow(x, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Prediction", fontsize=30)
    plt.imshow(yhat, cmap=plt.get_cmap("PiYG", 5))
    plt.subplot(1, 3, 3)
    plt.title("Ground truth", fontsize=30)
    plt.imshow(y, cmap=plt.get_cmap("PiYG", 5))
    plt.savefig(str(best_model_path)[:-3] + f"-{stride}-stride.png")

    summed = np.zeros((5, 768, 768), dtype=np.float32)
    window_w, window_h = 128, 128
    stride = 128
    for i in range(0, summed.shape[1] - window_h + 1, stride):
        for j in range(0, summed.shape[2] - window_w + 1, stride):
            with torch.no_grad():
                out = model(xx[:, :, i : i + window_w, j : j + window_h])
                summed[:, i : i + window_w, j : j + window_h] += (
                    out.cpu().detach().numpy()[0]
                )

    yhat = np.argmax(summed, axis=0)

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.title("Input", fontsize=30)
    plt.imshow(x, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Prediction", fontsize=30)
    plt.imshow(yhat, cmap=plt.get_cmap("PiYG", 5))
    plt.subplot(1, 3, 3)
    plt.title("Ground truth", fontsize=30)
    plt.imshow(y, cmap=plt.get_cmap("PiYG", 5))
    plt.savefig(str(best_model_path)[:-3] + f"-{stride}-stride.png")


if __name__ == "__main__":
    main()
