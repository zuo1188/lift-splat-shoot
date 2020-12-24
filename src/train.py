"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import datetime
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info, plot_data,plot_vehicle_roadsegment


def train(version,
            dataroot='/data/nuscenes',
            nepochs=10000,
            gpuid=1,

            H=900, W=1600,
            resize_lim=(0.386,0.45),#(0.193, 0.225),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',
            xbound=[-50.0, 50.0, 0.25],
            ybound=[-50.0, 50.0, 0.25],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 0.5],
            # xbound=[-50.0, 50.0, 0.5],
            # ybound=[-50.0, 50.0, 0.5],
            # zbound=[-10.0, 10.0, 20.0],
            # dbound=[4.0, 45.0, 1.0],

            bsz=4,
            nworkers=10,
            lr=1e-3,
            weight_decay=1e-7,
            modelf="",
            map_folder='/data/nuscenes/mini',
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata',map_folder=map_folder)
    n_train = len(trainloader.dataset)
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    
    if modelf:
        print('loading', modelf)
        model.load_state_dict(torch.load(modelf))
    
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    dir_logs = f'./output/{now}/'
    os.mkdir(dir_logs)

    writer = SummaryWriter(comment=f'{now}')
    val_step = 1000 if version == 'mini' else 10000
    
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=((fW*3)*val, (2*fW+ 2*fH)*val))
    gs = mpl.gridspec.GridSpec(4, 3, width_ratios=(fW,fW,fW),height_ratios=(fW,fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    counter = 0
    for epoch in range(nepochs):
    
        np.random.seed()
        model.train()
        opt.zero_grad()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{nepochs}', unit='img') as pbar: 
            for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs,meta) in enumerate(trainloader):
                
                
                preds = model(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
                binimgs = binimgs.to(device)
                # loss_vehicle = loss_fn(preds[:,0], binimgs[:,0])
                # loss_road_segment = loss_fn(preds[:,1], binimgs[:,1])
                # loss_lane_divider = loss_fn(preds[:,2], binimgs[:,2])
                # loss = loss_vehicle + loss_road_segment + loss_lane_divider
                loss = loss_fn(preds, binimgs)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                if counter % 2==0:
                    opt.step()
                    opt.zero_grad()
                
                

                if counter % 10 == 0:
                    # print(counter, loss.item())
                    writer.add_scalar('train/loss', loss, counter)
                    # writer.add_scalar('train/loss_vehicle', loss_vehicle, counter)
                    # writer.add_scalar('train/loss_lane_divider', loss_lane_divider, counter)
                    # writer.add_scalar('train/loss_road_segment', loss_road_segment, counter)
                    pbar.set_postfix(**{'loss_batch': loss.item(),'counter':counter})
                if counter % 50 == 0:
                    _, _, iou = get_batch_iou(preds, binimgs)
                    writer.add_scalar('train/iou', iou, counter)
                if counter % 1000 == 0:
                    output = f"{dir_logs}{counter:08}"
                    plot_data(imgs,binimgs,preds,meta['cams'],grid_conf,gs,output)
                    # output_gt = f"{dir_logs}gt_{batchi:06}"
                    # plot_ground_truth(imgs,binimgs,meta['cams'],grid_conf,gs,output_gt)
                    # output_pred = f"{dir_logs}pred_{batchi:06}"
                    # plot_ground_truth(imgs,preds,meta['cams'],grid_conf,gs,output_pred)
                    
                counter += 1
                pbar.update(imgs.shape[0])
            
        #eval after 1 epochs 
        if epoch % 1 ==0:
            val_info = get_val_info(model, valloader, loss_fn, device,True)
            print('VAL', val_info)
            writer.add_scalar('val/loss', val_info['loss'], counter)
            writer.add_scalar('val/iou', val_info['iou'], counter)
            model.eval()
            mname = os.path.join(logdir, "{}model{}_lr{}.pt".format(now,counter,lr))
            print('saving', mname)
            torch.save(model.state_dict(), mname)
    
    writer.close()
                
                
def infer(version,
            dataroot='/data/nuscenes',
            gpuid=1,
            H=900, W=1600,
            resize_lim=(0.386,0.45),#(0.193, 0.225),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=True,
            ncams=5,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',
            xbound=[-50.0, 50.0, 0.25],
            ybound=[-50.0, 50.0, 0.25],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 0.5],

            bsz=4,
            nworkers=10,
            modelf_vehicle="",
            modelf_roadsegment="",
            map_folder='/data/nuscenes/mini',
            ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata',map_folder=map_folder)
    loader = valloader
    n_train = len(loader.dataset)
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model_vehicle = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf_vehicle)
    model_vehicle.load_state_dict(torch.load(modelf_vehicle))
    model_vehicle.to(device)
    model_roadsegment = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf_roadsegment)
    model_roadsegment.load_state_dict(torch.load(modelf_roadsegment))
    model_roadsegment.to(device)

    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    dir_logs = f'./output/infer_{now}/'
    os.mkdir(dir_logs)
    
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=((fW*3)*val, (2*fW+ 2*fH)*val))
    gs = mpl.gridspec.GridSpec(4, 3, width_ratios=(fW,fW,fW),height_ratios=(fW,fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    counter = 0
    
    model_vehicle.eval()
    model_roadsegment.eval()
    with torch.no_grad():
        with tqdm(total=n_train,  unit='img') as pbar: 
            for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs,meta) in enumerate(loader):
                
                
                preds_vehicle = model_vehicle(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
                preds_roadsegment = model_roadsegment(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
                preds_vehicle = preds_vehicle.sigmoid().cpu()
                preds_roadsegment = preds_roadsegment.sigmoid().cpu()
                #binimgs = binimgs.to(device)
                #loss = loss_fn(preds, binimgs)

                output = f"{dir_logs}{counter:08}"
                plot_vehicle_roadsegment(imgs,binimgs,preds_vehicle,preds_roadsegment,meta['cams'],grid_conf,gs,output)

                counter += 1
                pbar.update(imgs.shape[0])
        