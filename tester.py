import os
import sys
import torch
import metrics
import numpy as np
import pandas as pd


def run(model, val_loader, device, output_path):

    model.eval()
    pred_masks = []
    ys = []

    for batch_i, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            pred_mask = model(x.to(device))
        pred_mask_class = torch.argmax(pred_mask, axis=1)
        pred_masks.append(pred_mask_class)
        ys.append(y)

        sys.stdout.write(
            "\r[Batch %d/%d]"
            % (
                batch_i,
                len(val_loader),
            )
        )

    pred_masks = torch.cat(pred_masks, dim=0)
    ys = torch.cat(ys, dim=0)

    scores, confusion_mat = metrics.test_metrics(
        ys.to(device), pred_masks.to(device), 5, device)

    df = pd.DataFrame(scores)
    df.to_csv(os.path.join(output_path, 'scores.csv'))

    np.save(os.path.join(output_path, 'confusion_mat.npy'), confusion_mat)
