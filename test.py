import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import csv


def plot_roc_curves(y_true, y_score, save_path,
                    micro_filename='roc_micro.png',
                    per_class_filename='roc_per_class.png',
                    csv_filename='roc_data.csv'):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # ========== Save CSV ==========
    with open(os.path.join(save_path, csv_filename), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'FPR', 'TPR', 'AUC'])

        # Per-class ROC
        fpr_dict = {}
        tpr_dict = {}
        auc_dict = {}

        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            for f, t in zip(fpr_dict[i], tpr_dict[i]):
                writer.writerow([i, f, t, auc_dict[i]])
            writer.writerow([])

    # ========== Plot micro-average ROC ==========
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.figure()
    plt.plot(fpr_micro, tpr_micro, color='darkorange',
             lw=2, label=f'Micro-average ROC (AUC = {auc_micro:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, micro_filename))
    plt.close()

    # ========== Plot per-class ROC in one figure ==========
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        plt.plot(fpr_dict[i], tpr_dict[i],
                 lw=2, label=f'Class {i} (AUC = {auc_dict[i]:.4f})',
                 color=colors(i))

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Per-Class ROC Curves')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, per_class_filename))
    plt.close()


def evaluate(epoch, model, loader, args=None, set='val', plot_roc=False):
    model.eval()
    loss_meter = Meter()
    acc_meter = Meter()
    label = torch.arange(args.way).repeat(args.query).cuda()
    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm_gen, 1):
            data = data.cuda()
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'ca'
            logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

            if plot_roc:
                all_logits.append(F.softmax(logits, dim=1).cpu())
                all_labels.append(label.cpu())

    if plot_roc:
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        plot_roc_curves(all_labels, all_logits, args.save_path)

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=4, pin_memory=True)

    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test', plot_roc=True)
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')
    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)
    test_main(model, args)
