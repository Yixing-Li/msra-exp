# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
from tensorboardX import SummaryWriter


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    last_loss = None
    for ith_sample, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # if ith_sample <= 2000:
        #     metric_logger.update(loss=0)
        #     metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        #     continue

        # if ith_sample in range(2240, 2243+1):
        #     for name, par in model.named_parameters():
        #         par_have_nan = par_have_nan or torch.isnan(par).any()
        #     print(f'{"*"*10}\n{ith_sample}, before, par_have_nan:{par_have_nan}\n{"*"*10}')

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        if args.use_mix:
            with torch.cuda.amp.autocast(): # Yixing: this would cause loss -> nan
                outputs = model(samples)
                if not args.cosub:
                    # yixing: goes here.
                    loss = criterion(samples, outputs, targets)
                else:
                    outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                    loss = 0.25 * criterion(outputs[0], targets) 
                    loss = loss + 0.25 * criterion(outputs[1], targets) 
                    loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                    loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 
        else: #if True:
            outputs = model(samples)
            if not args.cosub:
                # yixing: goes here.
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f'ith_sample:{ith_sample}') # 2243
            print("Loss is {}, stopping training".format(loss_value))
            # begin: yixing-test
            print(f'torch.isnan(samples).any():{torch.isnan(samples).any()}') # false
            print(f'torch.isnan(outputs[0]).any():{torch.isnan(outputs[0]).any()}') # true
            print(f'torch.isnan(outputs[1]).any():{torch.isnan(outputs[1]).any()}') # true
            print(f'torch.isnan(targets).any():{torch.isnan(targets).any()}')
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                outputs = model(samples, observe = True)
            ### end
            sys.exit(1)
        # end: yixing-test

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if args.get_histo and (ith_sample % 200 == 0):
            writer = None
            if utils.is_main_process():
                writer = SummaryWriter(f"{args.output_dir}/tmp/tb_grad_histo/v2_4k")  # for tensorboardX

                # gradient histogram in tensorboard
                is_infinite, is_finite = False, None
                for name, par in model.named_parameters():
                    is_infinite = is_infinite or torch.isnan(par).any()
                is_finite = not is_infinite

                if writer is not None and is_finite: #torch.isfinite(grad_norm).all():
                    for n, p in model.named_parameters():
                        if (p.requires_grad) and ("bias" not in n):
                            # self.writer.add_histogram("grad/{}".format(name), p.grad.float() * (float(args.update_freq[0]) / _acc_norm / _cur_scale), global_step)
                            if n.startswith('decoder.layers.'):
                                layer_idx, module_name = n[len(
                                    'decoder.layers.'):].split('.', 1)
                                writer.add_histogram(
                                    "grad_layer/{}".format(module_name), p.grad.float() / float(optimizer.scaler.loss_scale), int(layer_idx))
                    writer.flush()
                    print('gradient histogram in tensorboard')
                else:
                    print('isfinite', is_finite) # torch.isfinite(grad_norm).all())

        # for name, param in model.named_parameters():
        #     print(f'**{ith_sample} grad**:{name}, {param.grad}', end = '\n\n')
        
        ### yixing-test:begin
        # check the model grads.
        # if ith_sample == 2242 or ith_sample == 2241: 
        #     for name, param in model.named_parameters():
        #         print(f'**{ith_sample} grad**:{name}, {param.grad}', end = '\n\n') 
        ### end

        torch.cuda.synchronize()

        if model_ema is not None:
            # yixing: goes to here
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
    
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
        
#     last_loss = None
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
            
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
            
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
         
#         # with torch.cuda.amp.autocast(): # Yixing: this would cause loss -> nan
#         if True:
#             outputs = model(samples)
#             if not args.cosub:
#                 # yixing: goes here.
#                 loss = criterion(samples, outputs, targets)
#             else:
#                 outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
#                 loss = 0.25 * criterion(outputs[0], targets) 
#                 loss = loss + 0.25 * criterion(outputs[1], targets) 
#                 loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
#                 loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             # begin: yixing-test
#             # loss = last_loss
#             # loss_value = loss.item()
#             print(f'torch.isnan(samples).any():{torch.isnan(samples).any()}') # false
#             print(f'torch.isnan(outputs[0]).any():{torch.isnan(outputs[0]).any()}') # true
#             print(f'torch.isnan(outputs[1]).any():{torch.isnan(outputs[1]).any()}') # true
#             print(f'torch.isnan(targets).any():{torch.isnan(targets).any()}')
#             with torch.no_grad():
#                 # with torch.cuda.amp.autocast():
#                 outputs = model(samples, observe = True)

#             ### end
#             sys.exit(1)
#         # last_loss = loss
#         # end: yixing-test

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, args):
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for ith_images, (images, target) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # start: yixing
        get_res = args.get_spectrum or args.get_batch_simi
        get_res_args = {'get_spectrum': args.get_spectrum, 'get_batch_simi': args.get_batch_simi}
        if args.get_spectrum and args.get_spectrum_num is not None:
            if (ith_images not in range(args.get_spectrum_num)):
                get_res_args['get_spectrum'] = False 
        if args.get_batch_simi and args.get_batch_simi_num is not None:
            if (ith_images not in (range( args.get_batch_simi_num))):
                get_res_args['get_batch_simi'] = False 
        if get_res and (not get_res_args['get_spectrum']) and (not get_res_args['get_batch_simi']):
            print(f'get result mode, results got but test set not finished yet.')
            # break
            pass
            
        # end: yixing
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if args.use_mix:
            with torch.cuda.amp.autocast():
                output = model(images, ith_images = ith_images, get_res_args = get_res_args)
                loss = criterion(output, target)
        else: #if True:
            output = model(images, ith_images = ith_images, get_res_args = get_res_args)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
#                     set_training_mode=True, args = None):
#     model.train(set_training_mode)
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10
    
#     if args.cosub:
#         criterion = torch.nn.BCEWithLogitsLoss()
        
#     last_loss = None
#     for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
#         samples = samples.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         if mixup_fn is not None:
#             samples, targets = mixup_fn(samples, targets)
            
#         if args.cosub:
#             samples = torch.cat((samples,samples),dim=0)
            
#         if args.bce_loss:
#             targets = targets.gt(0.0).type(targets.dtype)
         
#         # with torch.cuda.amp.autocast(): # Yixing: this would cause loss -> nan
#         if True:
#             outputs = model(samples)
#             if not args.cosub:
#                 # yixing: goes here.
#                 loss = criterion(samples, outputs, targets)
#             else:
#                 outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
#                 loss = 0.25 * criterion(outputs[0], targets) 
#                 loss = loss + 0.25 * criterion(outputs[1], targets) 
#                 loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
#                 loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

#         loss_value = loss.item()

#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             # begin: yixing-test
#             # loss = last_loss
#             # loss_value = loss.item()
#             print(f'torch.isnan(samples).any():{torch.isnan(samples).any()}') # false
#             print(f'torch.isnan(outputs[0]).any():{torch.isnan(outputs[0]).any()}') # true
#             print(f'torch.isnan(outputs[1]).any():{torch.isnan(outputs[1]).any()}') # true
#             print(f'torch.isnan(targets).any():{torch.isnan(targets).any()}')
#             with torch.no_grad():
#                 # with torch.cuda.amp.autocast():
#                 outputs = model(samples, observe = True)

#             ### end
#             sys.exit(1)
#         # last_loss = loss
#         # end: yixing-test

#         optimizer.zero_grad()

#         # this attribute is added by timm on one optimizer (adahessian)
#         is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
#         loss_scaler(loss, optimizer, clip_grad=max_norm,
#                     parameters=model.parameters(), create_graph=is_second_order)

#         torch.cuda.synchronize()
#         if model_ema is not None:
#             model_ema.update(model)

#         metric_logger.update(loss=loss_value)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

