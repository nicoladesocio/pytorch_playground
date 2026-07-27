"""AlexNet model implementation."""
import torch
import torch.nn as nn

import lightning as pl

# Import from another folder
from utils.helpers import print_info, calculate_accuracy

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        print_info(f"Initializing AlexNet with {num_classes} classes")
        
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
            nn.Linear(256 * 6 * 6, 4096),
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
    def __init__(self, num_classes=1000, target_classes=None):
        super(AlexNetLightning, self).__init__()
        self.save_hyperparameters()  # saves num_classes + target_classes into checkpoint
        self.model = AlexNet(num_classes=num_classes)
        # Build label-mapping dicts from target_classes (list of original class ids)
        if target_classes is not None:
            print(f"Using target_classes: {target_classes}")
            self.idx_to_class = {idx: cls for idx, cls in enumerate(target_classes)}
        else:
            print("No target_classes provided; model output indices will match original class labels")
            self.idx_to_class = None

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Validate checkpoint compatibility before weights are restored."""
        saved_classes = checkpoint.get("hyper_parameters", {}).get("target_classes")
        if saved_classes is None:
            raise ValueError(
                "Checkpoint was saved without 'target_classes' (pre-remapping). "
                "This checkpoint is incompatible with the current code. "
                "Please delete it and start a fresh training run."
            )
        saved_num_classes = checkpoint.get("hyper_parameters", {}).get("num_classes")
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
        if (torch.isnan(images).any() or labels.min() < 0 or labels.max() >= self.model.num_classes):
            print_info(f"Invalid data at batch {batch_idx}: NaN in images: {torch.isnan(images).any()}, label range: [{labels.min()}, {labels.max()}]")
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = calculate_accuracy(outputs.argmax(dim=1), labels)
        print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    def on_after_backward(self):
        if self.global_step > 50:
            return
        for name, param in self.named_parameters():
            if param.grad is not None:
                g = param.grad
                print(f"Grad {name}: std={g.std().item():.6f}, mean={g.mean().item():.6f}, min={g.min().item():.6f}, max={g.max().item():.6f}")
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = calculate_accuracy(outputs.argmax(dim=1), labels)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        acc = calculate_accuracy(outputs.argmax(dim=1), labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        #optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
        optimizer = torch.optim.SGD(
            params=self.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
