import os
import sys
import json
import utils
import torch
import models
import pretrainer
import barlow_twins

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_workers = 0


def main():

    cfg_path = sys.argv[1]
    with open(cfg_path, 'r', encoding='utf-8') as file:
        cfg = json.load(file)

    AUG_DATA_ROOT = cfg['aug_data_root']
    TRAIN_FILES = cfg['train_files']
    BATCH_SIZE = cfg['batch_size']
    VAL_FILES = cfg['val_files']
    N_EPOCHS = cfg['n_epochs']
    OUTPUT_PATH = cfg['output_path']
    CHANNELS = cfg['channels']
    MODEL_STR = 'unet'

    if not os.path.exists(OUTPUT_PATH):
        raise ValueError('output path does not exist')

    print('Device:', device)

    train_dataset = utils.SunriseDataset(
        AUG_DATA_ROOT, TRAIN_FILES, barlow=True)

    val_dataset = utils.SunriseDataset(
        AUG_DATA_ROOT, VAL_FILES, barlow=True)

    encoder = models.UNetEncoder(n_channels=1, channels=CHANNELS).to(device)

    projector = models.UNetProjector(
        CHANNELS[-1], CHANNELS[-1] // 2).to(device)

    criterion = barlow_twins.bt_loss

    params = list(encoder.parameters()) + list(projector.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-3)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=barlow_twins.collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=barlow_twins.collate)

    best_encoder_state = pretrainer.run(
        encoder, projector, train_loader, val_loader, N_EPOCHS, criterion, optimizer, device)

    OUTPUT_PATH = os.path.join(
        OUTPUT_PATH,  MODEL_STR + '_' + cfg_path.split(
            '/')[-1][:-5])

    os.mkdir(OUTPUT_PATH)

    torch.save(best_encoder_state, os.path.join(
        OUTPUT_PATH, 'encoder.pt'))


if __name__ == '__main__':
    main()
