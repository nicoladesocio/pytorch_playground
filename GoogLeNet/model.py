"""GoogLeNet (Inception v1) implementation with auxiliary classifiers."""

import torch
import torch.nn as nn

import lightning as pl

try:
    from inception import Inception, InceptionOutClassifier
except ImportError:  # imported as part of the package / from the repo root
    from GoogLeNet.inception import Inception, InceptionOutClassifier

from utils.metrics import PercentageAccuracy


class GoogLeNet(torch.nn.Module):
    def __init__(self, num_classes=1000, image_size=227, aux_loss_weight=0.3):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.aux_loss_weight = aux_loss_weight

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192, [64, [96, 128], [16, 32], 32])
        self.inception3b = Inception(256, [128, [128, 192], [32, 96], 64])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, [192, [96, 208], [16, 48], 64])
        # extract aux1 after inception4a
        self.inception4b = Inception(512, [160, [112, 224], [24, 64], 64])
        self.inception4c = Inception(512, [128, [128, 256], [24, 64], 64])
        self.inception4d = Inception(512, [112, [144, 288], [32, 64], 64])
        # extract aux2 after inception4d
        self.inception4e = Inception(528, [256, [160, 320], [32, 128], 128])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, [256, [160, 320], [32, 128], 128])
        self.inception5b = Inception(832, [384, [192, 384], [48, 128], 128])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        # The auxiliary heads used a hardcoded 128*4*4 flatten, which only matched a
        # 227px input. Trace one dummy image through the stem to learn the real
        # spatial size at the aux tap points, for whatever --image_size was asked for.
        spatial_4a, spatial_4d = self._aux_spatial_sizes(image_size)
        self.aux1 = InceptionOutClassifier(512, num_classes, in_spatial=spatial_4a)
        self.aux2 = InceptionOutClassifier(528, num_classes, in_spatial=spatial_4d)

    def _aux_spatial_sizes(self, image_size):
        with torch.no_grad():
            x = torch.zeros(1, 3, image_size, image_size)
            x = self.maxpool1(self.conv1(x))
            x = self.maxpool2(self.conv2(x))
            x = self.inception3b(self.inception3a(x))
            x = self.maxpool3(x)
            x = self.inception4a(x)
            spatial_4a = int(x.shape[-1])
            x = self.inception4d(self.inception4c(self.inception4b(x)))
            spatial_4d = int(x.shape[-1])
        return spatial_4a, spatial_4d

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)

        # aux1 after inception4a (input channels 512)
        aux1 = self.aux1(x) if self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        # aux2 after inception4d (input channels 528)
        aux2 = self.aux2(x) if self.training else None

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, aux2, aux1


class GoogLeNetLightning(pl.LightningModule):
    def __init__(self, num_classes=1000, target_classes=None, image_size=227,
                 aux_loss_weight=0.3, lr=1e-4, weight_decay=1e-4, eta_min=1e-6,
                 optimizer="adam", momentum=0.9, scheduler="cosine"):
        super(GoogLeNetLightning, self).__init__()
        # save_hyperparameters() (not the restricted two-key form) so the optimizer /
        # aux-loss settings survive a checkpoint round-trip and a resumed run cannot
        # silently differ from the original one.
        self.save_hyperparameters()
        self.model = GoogLeNet(
            num_classes=num_classes,
            image_size=image_size,
            aux_loss_weight=aux_loss_weight,
        )
        self.criterion = nn.CrossEntropyLoss()
        # Accumulated (not per-batch-averaged) accuracy, reported as a percentage so it
        # can be read directly against the chance threshold in goal.md.
        self.train_acc_metric = PercentageAccuracy(num_classes=num_classes)
        self.val_acc_metric = PercentageAccuracy(num_classes=num_classes)
        self.test_acc_metric = PercentageAccuracy(num_classes=num_classes)
        if target_classes is not None:
            self.idx_to_class = {idx: cls for idx, cls in enumerate(target_classes)}
        else:
            self.idx_to_class = None

    def forward(self, x):
        return self.model(x)

    def predict_classes(self, outputs):
        pred_idx = outputs.argmax(dim=1).tolist()
        if self.idx_to_class is not None:
            return [self.idx_to_class[i] for i in pred_idx]
        return pred_idx

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs, aux2, aux1 = self.model(images)
        loss_main = self.criterion(outputs, labels)
        loss_aux2 = self.criterion(aux2, labels)
        loss_aux1 = self.criterion(aux1, labels)
        w = self.model.aux_loss_weight
        loss = loss_main + w * loss_aux2 + w * loss_aux1

        # NaN/Inf detection used to print unconditionally on every batch; log a scalar
        # instead so it is visible in TensorBoard without flooding stdout.
        if not torch.isfinite(loss):
            self.log("nonfinite_batches", 1.0, on_step=True, on_epoch=True, reduce_fx="sum")

        self.train_acc_metric.update(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss_main', loss_main, on_step=False, on_epoch=True)
        self.log('train_loss_aux1', loss_aux1, on_step=False, on_epoch=True)
        self.log('train_loss_aux2', loss_aux2, on_step=False, on_epoch=True)
        # Log the Metric object, not its return value: Lightning mean-reduces a
        # tensor per batch but compute()s a Metric once at epoch end. Verified in
        # _check_metric.py -- the tensor form reported 6000.0 where the true global
        # accuracy was 60.0.
        self.log('train_acc', self.train_acc_metric, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs, _, _ = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_acc_metric.update(outputs, labels)
        self.log('val_acc', self.val_acc_metric, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs, _, _ = self.model(images)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.test_acc_metric.update(outputs, labels)
        self.log('test_acc', self.test_acc_metric, prog_bar=True)
        return loss

    def on_after_backward(self):
        # Log gradient L2-norm at representative layers across the depth of the network.
        # Averaged per epoch so TensorBoard shows a smooth trend per epoch.
        layers_to_monitor = {
            'conv1':        self.model.conv1,        # earliest layer
            'inception3a':  self.model.inception3a,  # early inception
            'inception4a':  self.model.inception4a,  # mid (feeds aux1)
            'inception4d':  self.model.inception4d,  # mid-late (feeds aux2)
            'inception5b':  self.model.inception5b,  # deep
            'fc':           self.model.fc,           # classifier head
        }
        for name, layer in layers_to_monitor.items():
            grads = [p.grad.norm(2) for p in layer.parameters() if p.grad is not None]
            if grads:
                self.log(f'grad_norm/{name}', torch.stack(grads).mean(),
                         on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr,
                momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay)

        kind = (self.hparams.scheduler or "cosine").lower()
        if kind == "none":
            return {"optimizer": optimizer}

        # T_max must not come from self.trainer.max_epochs: configure_optimizers can
        # run before the trainer is attached (and does when a checkpoint is loaded),
        # which raised AttributeError. Fall back to a sane horizon when unavailable.
        try:
            horizon = self.trainer.max_epochs
        except (AttributeError, RuntimeError, KeyError):
            horizon = None
        if not horizon or horizon < 1:
            horizon = 60

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=self.hparams.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
