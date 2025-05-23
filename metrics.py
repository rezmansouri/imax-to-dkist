import torch

EPS = 1e-10


def acc(label, predicted):
    seg_acc = (label.cpu() == torch.argmax(predicted, axis=1).cpu()
               ).sum() / torch.numel(label.cpu())
    return seg_acc


def nanmean(x):
    return torch.mean(x[x == x])


def _fast_hist(true, pred, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    return hist


def overall_pixel_accuracy(hist):
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + EPS)
    return overall_acc


def per_class_pixel_accuracy(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc


def jaccard_index(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    avg_dice = nanmean(dice)
    return avg_dice


def per_class_OPA(hist):
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    return per_class_acc


def per_class_jaccard(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + EPS)
    return jaccard


def per_class_dice(hist):
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + EPS)
    return dice


def eval_metrics_sem(true, pred, num_classes, device):
    hist = torch.zeros((num_classes, num_classes)).to(device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    pc_opa = per_class_OPA(hist)
    pc_j = per_class_jaccard(hist)
    pc_d = per_class_dice(hist)
    return overall_acc, avg_per_class_acc, avg_jacc, avg_dice, pc_opa, pc_j, pc_d


def test_metrics(true, pred, num_classes, device):
    hist = torch.zeros((num_classes, num_classes)).to(device)
    for t, p in zip(true, pred):
        hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
    overall_acc = overall_pixel_accuracy(hist)
    avg_per_class_acc = per_class_pixel_accuracy(hist)
    avg_jacc = jaccard_index(hist)
    avg_dice = dice_coefficient(hist)
    pc_opa = per_class_OPA(hist)
    pc_j = per_class_jaccard(hist)
    pc_d = per_class_dice(hist)

    hist = hist.cpu().numpy()

    return {
        'overall_acc': [float(overall_acc.cpu()), None, None, None, None],
        'avg_per_class_acc': [float(avg_per_class_acc.cpu()), None, None, None, None],
        'avg_jacc': [float(avg_jacc.cpu().float()), None, None, None, None],
        'avg_dice': [float(avg_dice.cpu().float()), None, None, None, None],
        'pc_opa': list(pc_opa.cpu().numpy()),
        'pc_j': list(pc_j.cpu().numpy()),
        'pc_d': list(pc_d.cpu().numpy()),
    }, hist
