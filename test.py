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
from sklearn.metrics import precision_recall_fscore_support


def plot_roc_curve(y_true, y_score, save_path, filename='roc_curve_test.png'):
    y_score = np.array(y_score)
    y_true = np.array(y_true)
    n_classes = y_score.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='Micro-average ROC (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-Averaged ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, filename))
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
    all_preds = []

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
                all_preds.append(torch.argmax(logits, dim=1).cpu())

    # Compute ROC curve + metrics
    if plot_roc:
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels_np = torch.cat(all_labels, dim=0).numpy()
        all_preds_np = torch.cat(all_preds, dim=0).numpy()
        plot_roc_curve(all_labels_np, all_logits, args.save_path)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average='macro', zero_division=0)
        print(f'[metrics] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    else:
        precision = recall = f1 = None

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval(), precision, recall, f1


def test_main(model, args):
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))
    Dataset = dataset_builder(args)
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.test_episode, args.way, args.shot + args.query)
    test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=4, pin_memory=True)

    _, test_acc, test_ci, precision, recall, f1 = evaluate("best", model, test_loader, args, set='test', plot_roc=True)
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')
    print(f'[final] Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)
    test_main(model, args)
