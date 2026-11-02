"""AlexNet model implementation."""
import torch
import torch.nn as nn

import lightning as pl

# Import from another folder
from utils.helpers import print_info
from utils.metrics import PercentageAccuracy

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, image_size=227):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),  # No padding like original
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Re-enable Dropout
            nn.Linear(self._flatten_features(), 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        # Initialize like the original paper (Krizhevsky et al., 2012):
        # small fixed-std normal weights, and bias=1 on conv2/conv4/conv5 + both
        # hidden FC layers to keep ReLUs alive early in training (Kaiming init +
        # bias=0 everywhere was causing dead ReLUs / vanishing gradients).
        conv_layers = [layer for layer in self.features if isinstance(layer, nn.Conv2d)]
        for idx, layer in enumerate(conv_layers):
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            if layer.bias is not None:
                # conv_layers order: conv1, conv2, conv3, conv4, conv5
                bias_value = 1 if idx in (1, 3, 4) else 0
                nn.init.constant_(layer.bias, bias_value)
        
        # Initialize classifier: small std normal weights; bias=1 on the hidden
        # FC layers (fc1, fc2), bias=0 on the final output layer (no ReLU follows it).
        linear_layers = [layer for layer in self.classifier if isinstance(layer, nn.Linear)]
        for idx, layer in enumerate(linear_layers):
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            is_output_layer = idx == len(linear_layers) - 1
            nn.init.constant_(layer.bias, 0 if is_output_layer else 1)
    
    def _flatten_features(self):
        """Number of features reaching the classifier, derived from the real stack.

        Hardcoding ``256 * 6 * 6`` silently required an input that lands on exactly
        6x6: 227/231/256 work, but 224 yields 6400 and crashes inside ``nn.Linear``
        with a confusing shape error. Tracing one dummy image through ``features``
        makes any ``--image_size`` valid.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            out = self.features(dummy)
        return int(out.numel())

    def debug_features(self, x):
        for idx, layer in enumerate(self.features):
            x = layer(x)
            print_info(f"After features[{idx}] ({layer.__class__.__name__}): shape {x.shape}, mean {x.mean().item():.4f}, std {x.std().item():.4f}")
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetLightning(pl.LightningModule):
    def __init__(self, num_classes=1000, target_classes=None, image_size=227,
                 lr=0.01, momentum=0.9, weight_decay=5e-4,
                 lr_step_size=30, lr_gamma=0.1, log_grad_norms=True,
                 optimizer="sgd", scheduler="step"):
        super(AlexNetLightning, self).__init__()
        self.save_hyperparameters()  # saves num_classes + target_classes into checkpoint
        self.model = AlexNet(num_classes=num_classes, image_size=image_size)
        # Build label-mapping dicts from target_classes (list of original class ids)
        if target_classes is not None:
            print(f"Using target_classes: {target_classes}")
            self.idx_to_class = {idx: cls for idx, cls in enumerate(target_classes)}
        else:
            print("No target_classes provided; model output indices will match original class labels")
            self.idx_to_class = None
        # One loss module for the whole module (was reconstructed per step).
        self.criterion = nn.CrossEntropyLoss()
        # torchmetrics accumulates counts across the whole epoch, so the epoch value
        # is a true global accuracy. The previous code averaged the per-batch
        # percentage returned by utils.helpers.calculate_accuracy, which is wrong for
        # unequal batch sizes and -- because that helper returns 0..100 -- produced
        # values such as val_acc=1.22 that looked like a broken metric. Logged as a
        # percentage to keep the convention used by goal.md and train_final.py.
        self.train_acc_metric = PercentageAccuracy(num_classes=num_classes)
        self.val_acc_metric = PercentageAccuracy(num_classes=num_classes)
        self.test_acc_metric = PercentageAccuracy(num_classes=num_classes)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Validate checkpoint compatibility before weights are restored."""
        hparams = checkpoint.get("hyper_parameters", {})
        # Distinguish "key absent" from "key present and None". Passing
        # target_classes=None is a *supported* configuration (output indices then match
        # the original class labels), so a checkpoint saved that way by the current code
        # is perfectly resumable. Only a checkpoint predating the remapping feature --
        # where the key was never saved at all -- is genuinely incompatible. A blanket
        # `is None` test here made every such run unresumable.
        if "target_classes" not in hparams:
            raise ValueError(
                "Checkpoint was saved without a 'target_classes' hyperparameter "
                "(pre-remapping). This checkpoint is incompatible with the current "
                "code. Please delete it and start a fresh training run."
            )
        saved_num_classes = hparams.get("num_classes")
        if saved_num_classes != self.hparams.num_classes:
            raise ValueError(
                f"Checkpoint num_classes={saved_num_classes} does not match "
                f"current num_classes={self.hparams.num_classes}."
            )

    def forward(self, x):
        return self.model(x)

    def predict_classes(self, outputs):
        """Map model output indices back to original class labels."""
        pred_idx = outputs.argmax(dim=1).tolist()
        if self.idx_to_class is not None:
            return [self.idx_to_class[i] for i in pred_idx]
        return pred_idx
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        # Guard rails must not spam stdout: a bad batch is logged once per epoch to
        # the progress bar / logger instead of printing on every step.
        if (torch.isnan(images).any() or labels.min() < 0
                or labels.max() >= self.model.num_classes):
            self.log("bad_batches", 1.0, on_step=True, on_epoch=True, reduce_fx="sum")
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        # Update then log the Metric *object*. Verified in-container: logging the
        # tensor returned by the metric makes Lightning mean-reduce it per batch
        # (6000.0 where the true global accuracy is 60.0), while logging the Metric
        # itself is compute()d once at epoch end and gives the exact global value.
        self.train_acc_metric.update(outputs, labels)
        self.log("train_acc", self.train_acc_metric, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self):
        # Gradient health is useful for diagnosing vanishing gradients, but printing
        # every parameter for the first 50 steps produced ~50 * 25 lines of stdout and
        # dominated wall-clock time. Aggregate into a few TensorBoard scalars instead.
        if not self.hparams.log_grad_norms or self.global_step % 50 != 0:
            return
        total_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_sq += float(param.grad.detach().norm(2)) ** 2
        self.log("grad_norm/total", total_sq ** 0.5, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.val_acc_metric.update(outputs, labels)
        self.log("val_acc", self.val_acc_metric, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long()
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.test_acc_metric.update(outputs, labels)
        self.log("test_acc", self.test_acc_metric, prog_bar=True)
        return loss

    def configure_optimizers(self):
        name = (self.hparams.optimizer or "sgd").lower()
        if name == "adam":
            # Adam with the paper's lr=0.01 diverges; callers pair it with ~1e-4.
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay)

        kind = (self.hparams.scheduler or "step").lower()
        if kind == "none":
            return {"optimizer": optimizer}
        if kind == "cosine":
            try:
                horizon = self.trainer.max_epochs
            except (AttributeError, RuntimeError, KeyError):
                horizon = None
            if not horizon or horizon < 1:
                horizon = 60
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=horizon, eta_min=0.0)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
