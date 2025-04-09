import os
import sys
import torch
import metrics
import numpy as np
from torchsummary import summary


def run(model, train_loader, val_loader, n_epochs, criterion, optimizer, device, output_path, save_model=False, model_summary=False):

    if model_summary:
        summary(model, (1, 1, 128, 128))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.9)
    min_loss = torch.tensor(float('inf'))

    save_losses = []

    save_h_train_losses = []
    save_h_val_losses = []
    scheduler_counter = 0
    best_path = None

    for epoch in range(1, n_epochs+1):

        model.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_loader):

            pred_mask = model(x.to(device))
            loss = criterion(pred_mask, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(metrics.acc(y, pred_mask).numpy())

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    n_epochs,
                    batch_i,
                    len(train_loader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1

        model.eval()
        val_loss_list = []
        val_acc_list = []
        val_overall_pa_list = []
        val_per_class_pa_list = []
        val_jaccard_index_list = []
        val_dice_index_list = []

        for batch_i, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                pred_mask = model(x.to(device))
            val_loss = criterion(pred_mask, y.to(device))
            pred_mask_class = torch.argmax(pred_mask, axis=1)

            val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, _, _, _ = metrics.eval_metrics_sem(
                y.to(device), pred_mask_class.to(device), 5, device)
            val_overall_pa_list.append(val_overall_pa.cpu().detach().numpy())
            val_per_class_pa_list.append(
                val_per_class_pa.cpu().detach().numpy())
            val_jaccard_index_list.append(
                val_jaccard_index.cpu().detach().numpy())
            val_dice_index_list.append(val_dice_index.cpu().detach().numpy())
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(metrics.acc(y, pred_mask).numpy())

        print('\n epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val per class acc : {:.2f}'.format(epoch,
                                                                                                                   np.mean(
                                                                                                                       loss_list),
                                                                                                                   np.mean(
                                                                                                                       acc_list),
                                                                                                                   np.mean(
                                                                                                                       val_loss_list),
                                                                                                                   np.mean(val_per_class_pa_list)))
        if epoch % 20 == 0:
            save_h_train_losses.append([loss_list, acc_list])
            save_h_val_losses.append([val_loss_list, val_acc_list])

        save_losses.append([epoch, np.mean(loss_list), np.mean(acc_list), np.mean(val_loss_list),  np.mean(val_acc_list),
                            np.mean(val_overall_pa_list), np.mean(
                                val_per_class_pa_list),
                            np.mean(val_jaccard_index_list), np.mean(val_dice_index_list)])

        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        print('\n', min_loss, compare_loss)
        if is_best and save_model:
            best_path = os.path.join(
                output_path, 'epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list)))
            print("Best_model")
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model.state_dict(
            ), best_path)

        if scheduler_counter > 5:
            lr_scheduler.step()
            print(
                f"\nlowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0

        if epoch == n_epochs:
            print("Final Model")
            torch.save(model.state_dict(
            ), os.path.join(output_path, 'epoch_{}_{:.5f}.pt'.format(epoch, np.mean(val_loss_list))))

    if save_model:
        with open(os.path.join(output_path, 'stats.npy'), 'wb') as f:
            np.save(f, save_losses)
            np.save(f, save_h_train_losses)
            np.save(f, save_h_val_losses)

    return best_path
