import zipfile
import io
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


LATENT_SIZE = 50


def load_archive():
    archive = zipfile.ZipFile("train.zip", "r")
    num_images = 0
    image_list = []
    filenames = []
    for name in archive.filelist:
        if "__MACOSX" in name.filename: continue
        image_bytes = archive.read(name)
        if len(image_bytes) == 0: continue
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        image_list.append(image)
        filenames.append(name.filename)
        num_images += 1
        if False and num_images < 10:
            plt.imshow(image, cmap='gray')
            plt.show()
    return image_list, filenames


class NormalizedOp(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op
        self.bn = nn.BatchNorm2d(op.out_channels)

    def forward(self, x):
        x = self.op(x)
        x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.conv1 = NormalizedOp(nn.Conv2d(1, 32, 3, padding=1, stride=2))
        self.conv2 = NormalizedOp(nn.Conv2d(32, 64, 3, padding=1, stride=2))
        self.conv3 = NormalizedOp(nn.Conv2d(64, 128, 3, padding=1, stride=2))
        self.conv4 = NormalizedOp(nn.Conv2d(128, 256, 3, padding=0, stride=3))

        self.mu_fc = nn.Linear(256, latent_size)
        self.log_var_fc = nn.Linear(256, latent_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.shape[0], -1)

        mu = self.mu_fc(x)
        log_var = self.log_var_fc(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.dec_fc = nn.Linear(latent_size, 256)

        self.tconv1 = nn.ConvTranspose2d(256, 128, 3, padding=0, stride=3)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 3, output_padding=0, stride=2)
        self.tconv3 = nn.ConvTranspose2d(64, 32, 3, output_padding=0, stride=2)
        self.tconv4 = nn.ConvTranspose2d(32, 1, 3, output_padding=0, stride=2)

    def forward(self, z):
        x = self.dec_fc(z)

        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.relu(self.tconv1(x))
        x = torch.relu(self.tconv2(x))[:, :, :-1, :-1]
        x = torch.relu(self.tconv3(x))[:, :, :-1, :-1]
        x = torch.sigmoid(self.tconv4(x))[:, :, :-1, :-1]

        return x


class VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x = self.decoder(z)

        return x, z


class VaeTrainer:
    def __init__(self, latent_size):
        self.vae = VAE(latent_size=latent_size)
        self.vae.cuda()

    def train(self):
        self.vae.train()

        image_list, _ = load_archive()

        tensor_data = (1 / 255 * torch.tensor(image_list).float()).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(tensor_data)

        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-2)

        num_iters = 10_000
        i_iter = 0
        while i_iter < num_iters:
            for i_batch, (batch,) in enumerate(loader):
                batch = batch.cuda()
                pred_batch, z_batch = self.vae(batch)
                optimizer.zero_grad()
                loss_recon = F.mse_loss(batch, pred_batch)
                z_norm_loss = torch.norm(z_batch)
                loss = loss_recon + 0.5e-4 * z_norm_loss
                loss.backward()
                optimizer.step()
                if i_iter % 100 == 0:
                    # print(z_batch[0, ...])
                    print(i_iter, loss_recon.item(), z_norm_loss.item(), loss.item())
                    if False:
                        fig, axs = plt.subplots(1, 2)
                        axs[0].imshow(batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                        axs[1].imshow(pred_batch[0, 0, ...].detach().cpu().numpy(), cmap='gray')
                        plt.show()
                    torch.save(self.vae.state_dict(), "vae.pth")
                i_iter += 1


class Classifier(nn.Module):
    def __init__(self, latent_size, pretrained_model_path="vae.pth"):
        super().__init__()
        self.encoder = Encoder(latent_size)
        if True:
            state_dict = torch.load(pretrained_model_path)
            state_dict = {k[len("encoder."):]: v
                          for k, v in state_dict.items() if "encoder." in k}
            self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.class_fc = nn.Linear(latent_size, 1)

    def forward(self, x):
        with torch.no_grad():
            mu, log_var = self.encoder(x)
        logits = self.class_fc(mu)
        x = torch.sigmoid(logits)
        x = x.squeeze(-1)
        return x

    def train(self, mode=True):
        self.encoder.train(False)
        self.class_fc.train(mode)


class ClassifierTrainer:
    def __init__(self, latent_size):
        self.model = Classifier(latent_size)
        self.model.cuda()

    def _load_dataset(self, is_train):
        with open("annotation.json", "r") as f:
            dataset = json.load(f)
        flat_list = []
        for label, lst in dataset.items():
            val_div = 3
            if is_train:
                lst = lst[len(lst) // val_div:]
            else:
                lst = lst[:len(lst) // val_div]
            if label == "open":
                for name in lst:
                    flat_list.append(("train/" + name, 1))
            else:
                for name in lst:
                    flat_list.append(("train/" + name, 0))
        image_list, filenames = load_archive()
        name_to_image = dict(zip(filenames, image_list))
        anno_images = []
        for anno_name, anno_label in flat_list:
            if anno_name in name_to_image:
                anno_images.append(name_to_image[anno_name])
            else:
                print("Warning: file not found")
        images_np = np.array(anno_images)
        images_np = 1 / 255 * np.expand_dims(images_np, 1)
        anno_np = np.array([v[1] for v in flat_list])
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(images_np, dtype=torch.float32),
            torch.tensor(anno_np, dtype=torch.float32))
        return dataset

    def train(self):

        train_dataset = self._load_dataset(True)
        val_dataset = self._load_dataset(False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset),
                                                   shuffle=True, num_workers=0, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset),
                                                 shuffle=False, num_workers=0, drop_last=True)

        all_params = list(self.model.parameters())
        print("len(all_params)", len(all_params))

        optimizable_params = [p for p in self.model.parameters() if p.requires_grad]
        print("len(optimizable_params)", len(optimizable_params))
        optimizer = torch.optim.Adam(optimizable_params, lr=1e-2, weight_decay=1e-4)

        num_iters = 10_000
        i_iter = 0
        while i_iter < num_iters:
            for i_batch, (image_batch, anno_batch) in enumerate(train_loader):
                self.model.train()
                image_batch = image_batch.cuda()
                anno_batch = anno_batch.cuda()
                pred_batch = self.model(image_batch)
                optimizer.zero_grad()
                loss = F.binary_cross_entropy(pred_batch, anno_batch)
                loss.backward()
                optimizer.step()

                hard_pred_batch = pred_batch > 0.5
                anno_bool_batch = anno_batch > 0.5
                if i_iter % 1000 == 0:
                    print(i_iter, " train_loss=", loss.item())
                    accuracy = torch.sum(torch.eq(hard_pred_batch, anno_bool_batch)) / len(anno_bool_batch)
                    print("train_accuracy=", accuracy.item())
                    if False:
                        sample_idx = 0
                        plt.imshow(image_batch[sample_idx, 0, ...].detach().cpu().numpy(), cmap='gray')

                        def to_text(flag):
                            return "OPEN" if flag else "CLOSED"

                        plt.title("pred=" + to_text(hard_pred_batch[sample_idx]) + \
                                  " anno=" + to_text(anno_batch[sample_idx]))
                        plt.show()
                    self._validate(val_loader)
                    torch.save(self.model.state_dict(), "classifier.pth")
                i_iter += 1

    def _validate(self, val_loader):
        self.model.train(False)
        image_batch, anno_batch = next(iter(val_loader))
        image_batch = image_batch.cuda()
        anno_batch = anno_batch.cuda()
        with torch.no_grad():
            pred_batch = self.model(image_batch)
            loss = F.binary_cross_entropy(pred_batch, anno_batch)
        print("val_loss=", loss.item())
        hard_pred_batch = pred_batch > 0.5
        anno_bool_batch = anno_batch > 0.5
        accuracy = torch.sum(torch.eq(hard_pred_batch, anno_bool_batch)) / len(anno_bool_batch)
        print("val_accuracy=", accuracy.item())


def main():
    print(torch.cuda.is_available())

    latent_size = LATENT_SIZE

    vae_trainer = VaeTrainer(latent_size=latent_size)
    vae_trainer.train()

    class_trainer = ClassifierTrainer(latent_size)
    class_trainer.train()


if __name__ == "__main__":
    main()
