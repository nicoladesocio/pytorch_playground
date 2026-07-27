import torch

import pytorch_lightning as pl
from inception import Inception, InceptionOutClassifier

class GoogLeNet(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogLeNet, self).__init__()
        # Define the layers of the GoogLeNet architecture here

        # Auxiliary classifiers (input channels: 512 and 528 for aux1 and aux2)
        self.aux1 = InceptionOutClassifier(512, num_classes)
        self.aux2 = InceptionOutClassifier(528, num_classes)


        ## image dimension 227
        ## (227 - 7 + 2*x) / 2 + 1 = 112 => 
        ## x = 112 - 1 + 7 / 2 + - 227 / 2 = 112 - 1 + 3.5 - 113.5 = 111 - 110 = 

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1) 
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = Inception(192, [64, [96, 128], [16, 32], 32])
        self.inception3b = Inception(256, [128, [128,192], [32, 96], 64])
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = Inception(480, [192, [96, 208], [16, 48], 64])
        # extract aux1 after inception4a
        self.inception4b = Inception(512, [160, [112, 224], [24, 64], 64])
        self.inception4c = Inception(512, [128, [128, 256], [24, 64], 64])
        self.inception4d = Inception(512, [112, [144, 288], [32, 64], 64])
        # extract aux2 after inception4d
        self.inception4e = Inception(528, [256, [160, 320], [32, 128], 128])
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(832, [256, [160, 320], [32, 128], 128])
        self.inception5b = Inception(832, [384, [192, 384], [48, 128], 128])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(1024, num_classes)
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        #print('Input to GoogLeNet min:', x.min().item(), 'max:', x.max().item(), 'mean:', x.mean().item())
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
        #print('Shape before fc:', x.shape)
        x = torch.flatten(x, 1)
        #print('Shape after flatten:', x.shape)
        x = self.fc(x)
        #print('Shape after fc:', x.shape)
        #x = self.softmax(x)
        return x, aux2, aux1


class GoogLeNetLightning(pl.LightningModule):
    def __init__(self, num_classes=1000, target_classes=None):
        super(GoogLeNetLightning, self).__init__()
        self.save_hyperparameters("num_classes", "target_classes")
        self.model = GoogLeNet(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs, aux2, aux1 = self.model(images)
        # Check for NaNs/Infs in outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"NaN or Inf detected in outputs at batch {batch_idx}")
        if aux1 is not None and (torch.isnan(aux1).any() or torch.isinf(aux1).any()):
            print(f"NaN or Inf detected in aux1 at batch {batch_idx}")
        if aux2 is not None and (torch.isnan(aux2).any() or torch.isinf(aux2).any()):
            print(f"NaN or Inf detected in aux2 at batch {batch_idx}")
        loss_main = torch.nn.CrossEntropyLoss()(outputs, labels)
        loss_aux2 = torch.nn.CrossEntropyLoss()(aux2, labels)
        loss_aux1 = torch.nn.CrossEntropyLoss()(aux1, labels)
        loss = loss_main + 0.3 * loss_aux2 + 0.3 * loss_aux1
        # Check for NaNs/Infs in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at batch {batch_idx}")
        print(f"Batch {batch_idx}: loss_main={loss_main.item():.4f}, loss_aux2={loss_aux2.item():.4f}, loss_aux1={loss_aux1.item():.4f}, total_loss={loss.item():.4f}")
        for name, param in self.model.named_parameters():
            gradient = param.grad
            #if gradient is not None:
            #    print(f"Gradient for {name}: mean {gradient.mean().item():.4f}, std {gradient.std().item():.4f}")
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_loss_main', loss_main, on_step=False, on_epoch=True)
        self.log('train_loss_aux1', loss_aux1, on_step=False, on_epoch=True)
        self.log('train_loss_aux2', loss_aux2, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs, aux2, aux1 = self.model(images)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        print(f"Batch {batch_idx}: val_loss={loss.item():.4f}")
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
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
        # as from the paper
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96)
        # with Adam optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
