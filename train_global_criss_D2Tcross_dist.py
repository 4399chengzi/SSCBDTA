#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Traning scripts for regression tasks
"""

import torch
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# torch.optim.lr_scheduler.CosineAnnealingLR
import numpy as np
import pandas as pd
import json
from datetime import datetime
from argparse import ArgumentParser
from model.model_global_criss_D2Tcross import MolTransModel
from data.data_todo1 import DataEncoder, concordance_index1, get_rm2
from _utils.util import fix_random_seed_for_reproduce
from torch.utils.data import DataLoader
from data.label_util import encode
from _utils.util_function import (load_DAVIS, load_KIBA, load_ChEMBL_kd, load_ChEMBL_pkd,
                                  load_BindingDB_kd, load_davis_dataset, load_kiba_dataset)
from adan import Adan

# Set seed for reproduction
fix_random_seed_for_reproduce(torch_seed=2, np_seed=3)

# Whether to use GPU
# USE_GPU = True

# Set device as $export CUDA_VISIBLE_DEVICES='your device number'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# use_cuda = torch.cuda.is_available()
# device = 'cuda:1' if use_cuda else 'cpu'
# device = torch.cuda.set_device(device)

# Set loss function
reg_loss_fn = torch.nn.MSELoss()
Accumulation_Steps = 8


def get_reg_db(db_name):
    """
    Get benchmark dataset for regression
    """
    if db_name.lower() == 'benchmark_davis':
        dataset = load_davis_dataset()
    elif db_name.lower() == 'benchmark_kiba':
        dataset = load_kiba_dataset()
    else:
        raise ValueError('%s not supported' % db_name)
    return dataset


def get_raw_db(db_name):
    """
    Get raw dataset for regression
    """
    # Load Davis data
    if db_name.lower() == 'raw_davis':
        X_drugs, X_targets, y = load_DAVIS(convert_to_log=True)
    # Load Kiba data
    elif db_name.lower() == 'raw_kiba':
        X_drugs, X_targets, y = load_KIBA()
    # Load ChEMBL Kd data
    elif db_name.lower() == 'raw_chembl_kd':
        X_drugs, X_targets, y = load_ChEMBL_kd()
    # Load ChEMBL pKd data
    elif db_name.lower() == 'raw_chembl_pkd':
        X_drugs, X_targets, y = load_ChEMBL_pkd()
    # Load BindingDB Kd data
    elif db_name.lower() == 'raw_bindingdb_kd':
        X_drugs, X_targets, y = load_BindingDB_kd()
    else:
        raise ValueError('%s not supported! ' % db_name)
    return X_drugs, X_targets, y


def reg_test(data_generator, model, device):
    """
    Test for regression task
    """
    y_pred = []
    y_label = []

    model.eval()
    for _, data in enumerate(data_generator):
        d_out, mask_d_out, t_out, mask_t_out, label, label_drugSeqs, label_targetSeqs = \
            data['d_out'], data['mask_d_out'], data['t_out'], data['mask_t_out'], data['label'], data['label_drugSeqs'], \
            data['label_targetSeqs']

        pad_drug, drug_mask = encode(label_drugSeqs, -1)  # padding and obtain pad mask
        pad_tar, tar_mask = encode(label_targetSeqs, -1)  # padding and obtain pad mask

        temp = model(torch.as_tensor(np.array(d_out)).long().to(device),
                     torch.as_tensor(np.array(t_out)).long().to(device),
                     torch.as_tensor(np.array(mask_d_out)).long().to(device),
                     torch.as_tensor(np.array(mask_t_out)).long().to(device),
                     pad_drug.to(device), pad_tar.to(device), drug_mask.to(device), tar_mask.to(device))

        label = torch.as_tensor(np.array(label)).float().to(device)

        predicts = torch.squeeze(temp, axis=1)

        loss = reg_loss_fn(predicts, label)
        predict_id = torch.squeeze(temp).detach().cpu().numpy()
        label_id = label.to('cpu').numpy()

        y_label = y_label + label_id.flatten().tolist()
        y_pred = y_pred + predict_id.flatten().tolist()

        total_label = np.array(y_label)
        total_pred = np.array(y_pred)

        mse = ((total_label - total_pred) ** 2).mean(axis=0)
    return (mse, concordance_index1(np.array(y_label), np.array(y_pred)), get_rm2(y_label, y_pred), loss.item())


class DistCollateFn(object):
    '''
    fix bug when len(data) do not be divided by batch_size, on condition of distributed validation
    avoid error when some gpu assigned zero samples
    '''

    def __call__(self, batch):
        # if batch_size == 0:
        #     return dict(batch_size = batch_size, images = None, labels = None)

        # if self.training:
        d_out, mask_d_out, t_out, mask_t_out, label, label_drugSeqs, label_targetSeqs = zip(*batch)

        # return (d_out, mask_d_out, t_out, mask_t_out, label, label_drugSeqs, label_targetSeqs)

        return dict(
            d_out=d_out, mask_d_out=mask_d_out,
            t_out=t_out, mask_t_out=mask_t_out,
            label=label,
            label_drugSeqs=label_drugSeqs,
            label_targetSeqs=label_targetSeqs
        )


def main(args, local_master: bool):
    """
    Main function
    """
    if args.distributed:
        local_master = (args.local_rank == 0)
        global_master = (dist.get_rank() == 0)
    else:
        local_master = True
        global_master = True

    # Initialize LogWriter
    if local_master:
        log_writer = SummaryWriter(log_dir="./log")

    device, device_ids = _prepare_device(args.distributed, args.local_rank, args.local_world_size, local_master)
    # Basic setting
    optimal_mse = 10000
    log_iter = 50
    log_step = 0
    optimal_CI = 0
    optimal_RM2 = 0

    # Load model config
    model_config = json.load(open(args.model_config, 'r'))
    model = MolTransModel(model_config)
    model = model.to(device)

    # Load pretrained model
    # params_dict= paddle.load('./pretrained_model/pdb2016_single_tower_1')
    # model.set_dict(params_dict)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=device_ids, output_device=device_ids[0],
                    find_unused_parameters=False)

    # Optimizer

    # optim = torch.optim.Adam(params=model.parameters(), lr=args.lr) # Adam
    optim = torch.optim.Adamax(params=model.parameters(), lr=args.lr)  # Adamax
    # optim = torch.optimizer.AdamW(learning_rate=scheduler, parameters=model.parameters(), weight_decay=0.01) # AdamW
    # optim = Adan(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=args.opt_betas,
    #                  eps=args.opt_eps, max_grad_norm=args.max_grad_norm, no_prox=args.no_prox)
    # optim = Adan(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [100, 200], gamma=0.5)

    # Data Preparation
    # Regression Task - Raw Dataset
    # X_drugs, X_targets, y = get_raw_db(args.dataset)
    # reg_training_set, reg_validation_set, reg_testing_set = data_process(X_drugs, X_targets, y, 
    #                     frac=[0.8,0.1,0.1], drug_encoding='Transformer', target_encoding='Transformer', 
    #                     split_method='random_split', random_seed=1)
    # reg_training_data = DataEncoder(reg_training_set.index.values, 
    #                     reg_training_set.Label.values, reg_training_set)
    # reg_train_loader = utils.BaseDataLoader(reg_training_data, batch_size=args.batchsize, 
    #                     shuffle=True, drop_last=False, num_workers=args.workers)
    # reg_validation_data = DataEncoder(reg_validation_set.index.values, 
    #                     reg_validation_set.Label.values, reg_validation_set)
    # reg_validation_loader = utils.BaseDataLoader(reg_validation_data, batch_size=args.batchsize, 
    #                     shuffle=False, drop_last=False, num_workers=args.workers)
    # reg_testing_data = DataEncoder(reg_testing_set.index.values, 
    #                     reg_testing_set.Label.values, reg_testing_set)
    # reg_testing_loader = utils.BaseDataLoader(reg_testing_data, batch_size=args.batchsize, 
    #                     shuffle=False, drop_last=False, num_workers=args.workers)

    # Regression Task - Benchmark Dataset
    trainset, testset = get_reg_db(args.dataset)
    trainset_smiles = [d['smiles'] for d in trainset]
    trainset_protein = [d['protein'] for d in trainset]
    trainset_aff = [d['aff'] for d in trainset]

    testset_smiles = [d['smiles'] for d in testset]
    testset_protein = [d['protein'] for d in testset]
    testset_aff = [d['aff'] for d in testset]

    df_data_t = pd.DataFrame(zip(trainset_smiles, trainset_protein, trainset_aff))
    df_data_t.rename(columns={0: 'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)
    df_data_tt = pd.DataFrame(zip(testset_smiles, testset_protein, testset_aff))
    df_data_tt.rename(columns={0: 'SMILES', 1: 'Target Sequence', 2: 'Label'}, inplace=True)

    reg_training_data = DataEncoder(df_data_t.index.values, df_data_t.Label.values, df_data_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(reg_training_data) \
        if args.distributed else None
    is_shuffle = False if args.distributed else True
    reg_train_loader = DataLoader(reg_training_data,
                                  batch_size=args.batchsize,
                                  sampler=train_sampler,
                                  shuffle=is_shuffle,
                                  drop_last=True,
                                  num_workers=args.workers,
                                  collate_fn=DistCollateFn())

    reg_validation_data = DataEncoder(df_data_tt.index.values, df_data_tt.Label.values, df_data_tt)
    reg_validation_loader = DataLoader(reg_validation_data, batch_size=args.batchsize,
                                       shuffle=False, drop_last=False, num_workers=args.workers,
                                       collate_fn=DistCollateFn())

    # Initial Testing    
    print("=====Go for Initial Testing=====") if local_master else None
    with torch.no_grad():
        mse, CI, RM2, reg_loss = reg_test(reg_validation_loader, model, device)
        print("Testing result: MSE: {}, CI: {}, RM2: {}"
              .format(mse, CI, RM2)) if local_master else None

    if args.distributed:
        dist.barrier()  # Syncing machines before training

    # Training
    for epoch in range(args.epochs):
        print("=====Go for Training=====") if local_master else None

        if args.distributed:
            reg_train_loader.sampler.set_epoch(epoch)

        model.train()

        # Regression Task
        for batch_id, data in enumerate(reg_train_loader):
            d_out, mask_d_out, t_out, mask_t_out, label, label_drugSeqs, label_targetSeqs = \
                data['d_out'], data['mask_d_out'], data['t_out'], data['mask_t_out'], data['label'], data[
                    'label_drugSeqs'], data['label_targetSeqs']

            pad_drug, drug_mask = encode(label_drugSeqs, -1)  # padding and obtain pad mask
            pad_tar, tar_mask = encode(label_targetSeqs, -1)  # padding and obtain pad mask

            # temp = model(d_out.long().cuda(), t_out.long().cuda(), mask_d_out.long().cuda(), mask_t_out.long().cuda(),
            # pad_drug.cuda(), pad_tar.cuda(), drug_mask.cuda(), tar_mask.cuda())
            temp = model(torch.as_tensor(np.array(d_out)).long().to(device),
                         torch.as_tensor(np.array(t_out)).long().to(device),
                         torch.as_tensor(np.array(mask_d_out)).long().to(device),
                         torch.as_tensor(np.array(mask_t_out)).long().to(device),
                         pad_drug.to(device), pad_tar.to(device), drug_mask.to(device), tar_mask.to(device))

            # label = paddle.cast(label, "float32")
            # label = torch.FloatTensor(label)
            label = torch.as_tensor(np.array(label)).float().to(device)

            predicts = torch.squeeze(temp)
            loss = reg_loss_fn(predicts, label)
            loss = loss / Accumulation_Steps
            loss.backward()

            if (batch_id + 1) % Accumulation_Steps == 0:  # grad accumulate
                optim.step()
                optim.zero_grad()

            # Use a barrier() to make sure that all process have finished forward and backward
            if args.distributed:
                dist.barrier()
                #  obtain the sum of all total_loss at all processes
                dist.all_reduce(loss, op=dist.reduce_op.SUM)
                size = dist.get_world_size()
            else:
                size = 1
            loss /= size  # averages gl_loss across the whole world

            # scheduler.step()

            if local_master and batch_id % log_iter == 0:
                print("Training at epoch: {}, step: {}, loss is: {}"
                      .format(epoch, batch_id, loss.cpu().detach().numpy()))
                log_writer.add_scalar(tag="train/loss", global_step=log_step, scalar_value=loss.cpu().detach().numpy())
                log_step += 1

        scheduler.step()

        # Validation
        print("=====Go for Validation=====") if local_master else None
        with torch.no_grad():
            mse, CI, RM2, reg_loss = reg_test(reg_validation_loader, model, device)
            if local_master:
                print(
                    "Validation at epoch: {}, MSE: {}, CI: {}, RM2: {}, loss is: {} Best MSE: {} Best CI: {} Best RM2: {}"
                    .format(epoch, mse, CI, RM2, reg_loss, optimal_mse, optimal_CI, optimal_RM2))
                log_writer.add_scalar(tag="dev/loss", global_step=log_step, scalar_value=reg_loss)
                log_writer.add_scalar(tag="dev/mse", global_step=log_step, scalar_value=mse)
                log_writer.add_scalar(tag="dev/CI", global_step=log_step, scalar_value=CI)

                # Save best model
                if mse < optimal_mse:
                    optimal_mse = mse
                    print("Saving the best_model with best MSE...")
                    print("Best MSE: {}".format(optimal_mse))
                    torch.save(model.state_dict(), os.path.join('ckpts_global_criss_D2Tcross_kiba',
                                                                f'{args.dataset}_bestMSE_model_reg1_before_step32_kiba'))
                if CI > optimal_CI:
                    optimal_CI = CI
                    print("Saving the best_model with best CI...")
                    print("Best CI: {}".format(optimal_CI))
                    torch.save(model.state_dict(), os.path.join('ckpts_global_criss_D2Tcross_kiba',
                                                                f'{args.dataset}_bestCI_model_reg1_before_step32_kiba'))
                if RM2 > optimal_RM2:
                    optimal_RM2 = RM2
                    print("Saving the best_model with best RM2...")
                    print("Best RM2: {}".format(optimal_RM2))
                    torch.save(model.state_dict(), os.path.join('ckpts_global_criss_D2Tcross_kiba',
                                                                f'{args.dataset}_bestRM2_model_reg1_before_step32_kiba'))
    if local_master:
        # Print final result
        print("Best MSE: {}".format(optimal_mse))
        print("Best CI: {}".format(optimal_CI))
        print("Best RM2: {}".format(optimal_RM2))
        torch.save(model.state_dict(), f'{args.dataset}_final_model_reg1')

    # Testing
    # print("=====Go for Testing=====")
    # Regression Task
    # with paddle.no_grad():
    #     try:
    #         mse, CI, reg_loss = reg_test(reg_testing_loader, model)
    #         print("Testing result: MSE: {}, CI: {}".format(mse, CI))
    #     except:
    #         print("Testing failed...")


def logger_info(msg, local_master):
    print(msg) if local_master else None


def logger_warning(msg, local_master):
    logger_info(msg, local_master)


def _prepare_device(distributed, local_rank, local_world_size, local_master):
    '''
     setup GPU device if available, move model into configured device
    :param local_rank:
    :param local_world_size:
    :return:
    '''
    if distributed:
        ngpu_per_process = torch.cuda.device_count() // local_world_size
        device_ids = list(range(local_rank * ngpu_per_process, (local_rank + 1) * ngpu_per_process))

        if torch.cuda.is_available() and local_rank != -1:
            torch.cuda.set_device(device_ids[0])  # device_ids[0] =local_rank if local_world_size = n_gpu per node
            device = 'cuda'
            logger_info(
                f"[Process {os.getpid()}] world_size = {dist.get_world_size()}, "
                + f"rank = {dist.get_rank()}, n_gpu/process = {ngpu_per_process}, device_ids = {device_ids}",
                local_master
            )
        else:
            logger_warning('Training will be using CPU!', local_master)
            device = 'cpu'
        device = torch.device(device)
        return device, device_ids
    else:
        n_gpu = torch.cuda.device_count()
        n_gpu_use = local_world_size
        if n_gpu_use > 0 and n_gpu == 0:
            logger_warning("Warning: There\'s no GPU available on this machine,"
                           "training will be performed on CPU.", local_master)
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logger_warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                           "on this machine.".format(n_gpu_use, n_gpu), local_master)
            n_gpu_use = n_gpu

        list_ids = list(range(n_gpu_use))
        if n_gpu_use > 0:
            torch.cuda.set_device(list_ids[0])  # only use first available gpu as devices
            logger_warning(f'Training is using GPU {list_ids[0]}!', local_master)
            device = 'cuda'
        else:
            logger_warning('Training is using CPU!', local_master)
            device = 'cpu'
        device = torch.device(device)
        return device, list_ids


def entry_point(config):
    '''
    entry-point function for a single worker, distributed training
    '''

    local_world_size = config.local_world_size

    # check distributed environment cfgs
    if config.distributed:  # distributed gpu mode
        # check gpu available
        if torch.cuda.is_available():
            if torch.cuda.device_count() < local_world_size:
                raise RuntimeError(f'the number of GPU ({torch.cuda.device_count()}) is less than '
                                   f'the number of processes ({local_world_size}) running on each node')
            local_master = (config.local_rank == 0)
        else:
            raise RuntimeError('CUDA is not available, Distributed training is not supported.')
    else:  # one gpu or cpu mode
        if config.local_world_size != 1:
            raise RuntimeError('local_world_size must set be to 1, if distributed is set to false.')
        config.local_rank = 0
        local_master = True
        config.global_rank = 0

    # logger = config.get_logger('train') if local_master else None
    if config.distributed:
        print('Distributed GPU training model start...') if local_master else None
    else:
        print('One GPU or CPU training mode start...') if local_master else None

    if config.distributed:
        # these are the parameters used to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
        }
        print(f'[Process {os.getpid()}] Initializing process group with: {env_dict}') if local_master else None

        # init process group
        dist.init_process_group(backend='nccl', init_method='env://')
        config.global_rank = dist.get_rank()
        # info distributed training cfg
        print(
            f'[Process {os.getpid()}] world_size = {dist.get_world_size()}, '
            + f'rank = {dist.get_rank()}, backend={dist.get_backend()}'
        ) if local_master else None

    # start train
    beginT = datetime.now()
    print("Starting Time: {}".format(beginT.strftime(r'%m_%d_%H:%M:%S'))) if local_master else None

    main(config, local_master)

    endT = datetime.now()
    print("Ending Time: {}".format(endT.strftime(r'%m_%d_%H:%M:%S'))) if local_master else None
    difference = endT - beginT
    m, s = divmod(difference.total_seconds(), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Duration is {}d {}h {}m {}s".format(d, h, m, s)) if local_master else None

    # tear down the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(description='Start Training...')
    parser.add_argument('-b', '--batchsize', default=32, type=int, metavar='N', help='Batch size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='Number of workers')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='Number of total epochs')
    parser.add_argument('--dataset', choices=['raw_chembl_pkd', 'raw_chembl_kd', 'raw_bindingdb_kd', 'raw_davis',
                                              'raw_kiba', 'benchmark_davis', 'benchmark_kiba'],
                        default='benchmark_davis', type=str,
                        metavar='DATASET', help='Select specific dataset for your task')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='Initial learning rate', dest='lr')
    parser.add_argument('--model_config', default='./config.json', type=str)

    # parser.add_argument('--distributed', default=True, type=bool,
    #                     help='run distributed training. (true or false, default: true)')
    parser.add_argument('--distributed', action='store_true',
                        help='run distributed training. (true or false, default: true)')
    parser.add_argument('--local_world_size', default=1, type=int,
                        help='the number of processes running on each node, this is passed in explicitly '
                             'and is typically either $1$ or the number of GPUs per node. (default: 1)'),
    parser.add_argument('--local_rank', default=0, type=int,
                        help='this is automatically passed in via torch.distributed.launch.py, '
                             'process will be assigned a local rank ID in [0, local_world_size-1]. (default: 0)')
    parser.add_argument('--global_rank', default=0, type=int)

    # # Adan args
    # parser.add_argument('--max-grad-norm', type=float, default=0.0,
    #                     help='if the l2 norm is large than this hyper-parameter, then we clip the gradient  (default: 0.0, no gradient clip)')
    # parser.add_argument('--weight-decay', type=float, default=0.02,
    #                     help='weight decay, similar one used in AdamW (default: 0.02)')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
    #                     help='optimizer epsilon to avoid the bad case where second-order moment is zero (default: None, use opt default 1e-8 in adan)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
    #                     help='optimizer betas in Adan (default: None, use opt default [0.98, 0.92, 0.99] in Adan)')
    # parser.add_argument('--no-prox', action='store_true', default=False,
    #                     help='whether perform weight decay like AdamW (default=False)')

    args = parser.parse_args()

    entry_point(args)
    # main(args)
