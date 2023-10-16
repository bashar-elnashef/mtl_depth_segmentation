import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils.util import inf_loop, MetricTracker
from model.metric import AverageMeter, MeanIoU, RMSE
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, show_log_on_screen=True):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.masks = self.config['masks']
        self.metrics = self.config['metrics']
        self.metric_ftns = metric_ftns
        self.show_log_on_screen = show_log_on_screen

        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        ##
        self.do_validation = False
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', 
            *[f'loss_{msk}' for msk in self.masks])

        self.valid_metrics = MetricTracker(
            'loss',
            *[f'loss_{msk}' for msk in self.masks],
            *list(self.metrics),
            writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        loss_meter = AverageMeter()
        self.train_metrics.reset()


        pbar = tqdm(self.data_loader)

        loss_coeffs = (1., 1.)
        grad_norm = 0.
        batch_idx = 0

        for sample in tqdm(self.data_loader):
            loss = 0.0
            data = sample["image"].float().to(self.device)
            targets = [sample[k].to(self.device) for k in self.data_loader.dataset.masks]
            outputs = self.model(data) # Forward

            losses = [crit(F.interpolate(out, size=target.size()[1:], mode="bilinear", 
                            align_corners=False).squeeze(dim=1),target.squeeze(dim=1))
                for out, target, crit, loss_coeff in zip(outputs, targets, self.criterion, loss_coeffs)] 

            loss = sum(c * l for c, l in zip(loss_coeffs, losses))

            # Backward
            for opt in self.optimizer: opt.zero_grad()
            loss.backward()

            if grad_norm > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)

            for opt in self.optimizer: opt.step()

            loss_meter.update(loss.item())
            pbar.set_description(
                f'Loss {loss.item():.3f} | Avg. Loss {loss_meter.avg:.3f}'
            )

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for idx, msk in enumerate(self.masks):
                self.train_metrics.update(f'loss_{msk}', losses[idx].item())

            if batch_idx % self.log_step == 0:
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

            batch_idx += 1

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{f'val_{k}': v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            for sch in self.lr_scheduler: sch.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        loss_meter = AverageMeter()

        pbar = tqdm(self.valid_data_loader)

        loss_coeffs = (0.5, 0.5)
        grad_norm = 0.
        batch_idx = 0


        for metric in self.metric_ftns:
            metric.reset()

        def get_val(metrics):
            results = [(m.name, m.val()) for m in metrics]
            names, vals = list(zip(*results))
            out = ["{} : {:4f}".format(name, val) for name, val in results]
            return vals, " | ".join(out)

        with torch.no_grad():

            for sample in pbar:
                data = sample["image"].float().to(self.device)
                targets = [sample[k].to(self.device) for k in self.valid_data_loader.dataset.masks]

                outputs = self.model(data) # Forward

                losses_lst = []
                for out, target, crit, loss_coeff in zip(outputs, targets, self.criterion, loss_coeffs):
                    losses_lst.append(loss_coeff * crit(
                        F.interpolate(
                            out, size=target.size()[1:], mode="bilinear", align_corners=False
                        ).squeeze(dim=1),
                        target.squeeze(dim=1),
                    ))

                targets = [target.squeeze(dim=1).cpu().numpy() for target in targets]


                for out, target, metric in zip(outputs, targets, self.metric_ftns):
                    metric.update(
                        F.interpolate(out, size=target.shape[1:], mode="bilinear", align_corners=False)
                        .squeeze(dim=1)
                        .cpu()
                        .numpy(),
                        target,
                )
                vals = get_val(self.metric_ftns)
                pbar.set_description(vals[1])
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                self.valid_metrics.update('loss', sum(l.item() for l in losses_lst))

                for idx, msk in enumerate(self.masks):
                    self.valid_metrics.update(f'loss_{msk}', losses_lst[idx].item())

                for idx, met in enumerate(self.metrics):
                    self.valid_metrics.update(met, vals[0][idx])

                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            # add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

            return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
