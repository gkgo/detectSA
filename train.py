import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common.meter import Meter
from common.utils import  compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from test import test_main, evaluate


from ptflops import get_model_complexity_info  # ✅ 新增

def train_main(args):
    Dataset = dataset_builder(args)

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    model = RENet(args).cuda()

    # ✅ 模型复杂度统计（注意：不要用 DataParallel 包裹）
    print("===== Model Complexity Summary =====")
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 84, 84), as_strings=True, print_per_layer_stat=False)
        print(f'FLOPs: {flops}')
        print(f'Params: {params}')
        if not args.no_wandb:
            wandb.log({'model/FLOPs': flops, 'model/Params': params})

    # ✅ 再包裹 DataParallel（用于实际训练）
    model = nn.DataParallel(model, device_ids=args.device_ids)

    if not args.no_wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)
        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc, 'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
