import logging
import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from models import esrgan, vgg_loss
from src import super_resolution_dataset
from utils import costants


def build_dataloaders(batch_size):
    """
    Builds and returns training and validation
    dataloaders for the the training stage.

    Args:
        batch_size (int): batch size for the training
        dataloader

    Returns:
        - train_dataloader (DataLoader): training dataloader
        - val_dataloader (DataLoader): testing dataloader
    """
    HR = 128
    LR = HR//4

    transform_both = A.Compose([
        A.RandomCrop(HR, HR)
        ]
    )

    transform_hr = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
        ]
    )

    transform_lr = A.Compose([
        A.Resize(width=LR, height=LR, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
        ]
    )

    train_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_TRAIN,
        lr_path=costants.LR_TRAIN,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )
    val_dataset = super_resolution_dataset.SuperResolutionDataset(
        hr_path=costants.ORIGINAL_DS_VAL,
        lr_path=costants.LR_VAL,
        transform_both=transform_both,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader









def train(start_epoch, num_epochs, device, run_name,
          weights_folder, batch_size, lr,
          weights_gen, weights_disc, training_type, name_tensorboard):
    """
    Executes the training stage. It can execute the training
    of the generator or of the discriminator or both together
    (based on the value of "training_type").

    Args:
        - start_epoch (int): indicates the epochs you are restarting
        from (if 0 it means that it's training from scratch)
        - num_epochs (int): number of epochs e want to train
        - device (str): "cuda" or "cpu", it indicates whether we want
        to use cpu or gpu.
        - run_name (str): path of Tensorboard logger
        - weights_folder (str): path of the folder where to save the
        weights of the trained model
        - batch_size (int): batch size
        - lr (float): learning rate
        - weights_gen (str): path of the pretrained weights for the generator
        weights_disc (str): path of the pretrained weights for the discriminator
        - training_type (str): "generator" or "discriminator" or "gen_and_disc",
        it indicates whether we want to train only the generator, only the
        discriminator or both together.
        - name_tensorboard (str): name of plot we want to save the loss fucntion
        into (for Tenaorboard)

    Returns: None
    """
    generator = esrgan.Generator(upsample_algo="nearest", num_blocks=2)
    discriminator = esrgan.Discriminator()

    # LOAD PRETRAINED WEIGHTS
    generator.load_state_dict(torch.load(weights_gen))
    discriminator.load_state_dict(torch.load(weights_disc))

    optimizer_gen = optim.Adam(generator.parameters(), lr=lr)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)

    train_dataloader, val_dataloader = build_dataloaders(batch_size=batch_size)

    if training_type=="generator":
        train_gan_generator(
            generator=generator,
            discriminator=discriminator,
            optimizer_gen=optimizer_gen,
            train_dataloader=train_dataloader,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            device=device,
            run_name=run_name,
            weights_folder=weights_folder,
            name_tensorboard=name_tensorboard)

    elif training_type=="discriminator":
        train_discriminator(
            generator=generator,
            discriminator=discriminator,
            optimizer_disc=optimizer_disc,
            train_dataloader=train_dataloader,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            device=device,
            run_name=run_name,
            weights_folder=weights_folder,
            name_tensorboard=name_tensorboard
        )
    elif training_type=="gen_and_disc":
        train_gen_and_disc(generator=generator,
            discriminator=discriminator,
            optimizer_gen=optimizer_gen,
            optimizer_disc=optimizer_disc,
            train_dataloader=train_dataloader,
            start_epoch=start_epoch,
            num_epochs=num_epochs,
            device=device,
            run_name=run_name,
            weights_folder=weights_folder,
            name_tensorboard=name_tensorboard)







def train_gen_and_disc(generator,
                        discriminator,
                        optimizer_gen,
                        optimizer_disc,
                        train_dataloader,
                        start_epoch,
                        num_epochs,
                        device,
                        run_name,
                        weights_folder,
                        name_tensorboard):
    """
    Executes the training of the genrator and discriminator together.

    Args:
        - generator (esrgan.Generator): generator model
        - discriminator (esrgan.Discriminator): discriminator model
        - optimizer_gen (Optimizer): optimizer of the generator model
        - optimizer_disc (Optimizer): optimizer of the discriminator model
        - train_dataloader (DataLoader): training dataloader
        - start_epoch (int): indicates the epochs you are restarting
        from (if 0 it means that it's training from scratch)
        - num_epochs (int): number of epochs e want to train
        - device (str): "cuda" or "cpu", it indicates whether we want
        to use cpu or gpu.
        - run_name (str): path of Tensorboard logger
        - weights_folder (str): path of the folder where to save the
        weights of the trained model
        - name_tensorboard (str): name of plot we want to save the loss fucntion
        into (for Tenaorboard)

    Returns: None
    """
    generator.train()
    discriminator.train()
    generator.to(device)
    discriminator.to(device)

    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc = torch.cuda.amp.GradScaler()

    ## BINARY CROSS ENTROPY + SIGMOID
    bce = nn.BCEWithLogitsLoss()
    ## VGG LOSS (PERCEPTUAL LOSS)
    vgg = vgg_loss.VGGLoss(device=device, without_activation=True)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")

        tot_gen_loss = 0
        tot_adv_loss = 0
        tot_vgg_loss = 0
        tot_l1_loss = 0
        tot_disc_loss= 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)

            with torch.cuda.amp.autocast():
                fake_hr_images = generator(lr_images)
                out_fake = discriminator(fake_hr_images)
                out_real = discriminator(hr_images)

                ## GENERATOR LOSS FUCNTIONS
                adv_loss = 1e-3 * bce(
                    out_fake, torch.ones_like(out_fake)
                )
                vgg_loss = 0.006 * vgg(fake_hr_images, hr_images)
                gen_final_loss = vgg_loss + adv_loss
            ## OPTIMIZE GEN
            optimizer_gen.zero_grad()
            scaler_gen.scale(gen_final_loss).backward(retain_graph=True)
            scaler_gen.step(optimizer_gen)
            scaler_gen.update()

            with torch.cuda.amp.autocast():
                ## DISCRIMINATOR LOSS FUCNTIONS
                disc_loss_real = bce(
                    out_real, torch.ones_like(out_real)
                )
                disc_loss_fake = bce(
                    out_fake, torch.zeros_like(out_fake)
                )
                disc_final_loss = disc_loss_fake + disc_loss_real
            ## OPTIMIZE DISC
            optimizer_disc.zero_grad()
            scaler_disc.scale(disc_final_loss).backward()
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            pbar.set_postfix(GEN=gen_final_loss.item())

            ## TOT LOSS UPDATES
            tot_gen_loss += gen_final_loss.item()
            tot_adv_loss += adv_loss.item()
            tot_vgg_loss += vgg_loss.item()
            tot_disc_loss += disc_final_loss.item()

        tot_vgg_loss /= len_dataloader
        tot_adv_loss /= len_dataloader
        tot_gen_loss /= len_dataloader
        tot_disc_loss /= len_dataloader
        pbar.set_postfix(GEN=tot_gen_loss)

        logger.add_scalars(name_tensorboard, {
                'adv_loss': tot_adv_loss,
                'vgg_loss': tot_vgg_loss,
                'tot_gen_loss': tot_gen_loss,
                'tot_disc_loss': tot_disc_loss,
            }, start_epoch+epoch)

        if epoch%10==9:
            torch.save(generator.state_dict(), os.path.join(weights_folder, f"generator_{start_epoch+epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+epoch}.pt"))

    torch.save(generator.state_dict(), os.path.join(weights_folder, f"generator_{start_epoch+num_epochs}.pt"))
    torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+num_epochs}.pt"))










def train_gan_generator(generator,
                discriminator,
                optimizer_gen,
                train_dataloader,
                start_epoch,
                num_epochs,
                device,
                run_name,
                weights_folder,
                name_tensorboard):
    """
    Executes the training of the generator.

    Args:
        - generator (esrgan.Generator): generator model
        - discriminator (esrgan.Discriminator): discriminator model
        - optimizer_gen (Optimizer): optimizer of the generator model
        - train_dataloader (DataLoader): training dataloader
        - start_epoch (int): indicates the epochs you are restarting
        from (if 0 it means that it's training from scratch)
        - num_epochs (int): number of epochs e want to train
        - device (str): "cuda" or "cpu", it indicates whether we want
        to use cpu or gpu.
        - run_name (str): path of Tensorboard logger
        - weights_folder (str): path of the folder where to save the
        weights of the trained model
        - name_tensorboard (str): name of plot we want to save the loss fucntion
        into (for Tenaorboard)

    Returns: None
    """
    generator.train()
    discriminator.eval()
    generator.to(device)
    discriminator.to(device)

    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    scaler = torch.cuda.amp.GradScaler()

    ## BINARY CROSS ENTROPY + SIGMOID
    bce = nn.BCEWithLogitsLoss()
    ## L1 LOSS
    l1 = nn.L1Loss()
    ## VGG LOSS (PERCEPTUAL LOSS)
    vgg = vgg_loss.VGGLoss(device=device, without_activation=True)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")

        tot_gen_loss = 0
        tot_adv_loss = 0
        tot_vgg_loss = 0
        tot_l1_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)



            with torch.cuda.amp.autocast():
                fake_hr_images = generator(lr_images)
                out_fake = discriminator(fake_hr_images)
                #out_real = discriminator(hr_images)

                ## LOSS FUCNTIONS
                l1_loss = 0.5*l1(hr_images, fake_hr_images)
                #adv_loss = 1e-3 * bce(
                #    out_fake, torch.ones_like(out_fake)
                #)
                adv_loss = 1000*bce(
                    out_fake, torch.ones_like(out_fake)
                )

                vgg_loss = 0.006 * vgg(fake_hr_images, hr_images)
                #vgg_loss = 0.16 * vgg(fake_hr_images, hr_images)
                #vgg_loss = 0

                gen_final_loss = vgg_loss + adv_loss
                #gen_final_loss = vgg_loss + adv_loss + l1_loss

                tot_gen_loss += gen_final_loss.item()
                tot_adv_loss += adv_loss.item()
                tot_vgg_loss += vgg_loss.item()
                tot_l1_loss += l1_loss.item()

            ## OPTIMIZE GEN
            optimizer_gen.zero_grad()
            scaler.scale(gen_final_loss).backward()
            scaler.step(optimizer_gen)
            scaler.update()

            pbar.set_postfix(GEN=gen_final_loss.item())

        tot_vgg_loss /= len_dataloader
        tot_l1_loss /= len_dataloader
        tot_adv_loss /= len_dataloader
        tot_gen_loss /= len_dataloader
        #print(tot_adv_loss)

        pbar.set_postfix(GEN=tot_gen_loss)

        logger.add_scalars(name_tensorboard, {
                'adv_loss': tot_adv_loss,
                'vgg_loss': tot_vgg_loss,
                #'l1_loss': tot_l1_loss,
                'tot_loss': tot_gen_loss,
            }, start_epoch+epoch)

        if epoch%10==9:
            torch.save(generator.state_dict(), os.path.join(weights_folder, f"generator_{start_epoch+epoch}.pt"))

    torch.save(generator.state_dict(), os.path.join(weights_folder, f"generator_{start_epoch+num_epochs}.pt"))










def train_discriminator(generator,
                        discriminator,
                        optimizer,
                        train_dataloader,
                        start_epoch,
                        num_epochs,
                        device,
                        run_name,
                        weights_folder,
                        name_tensorboard):
    """
    Executes the training of the discriminator.

    Args:
        - generator (esrgan.Generator): generator model
        - discriminator (esrgan.Discriminator): discriminator model
        - optimizer_disc (Optimizer): optimizer of the discriminator model
        - train_dataloader (DataLoader): training dataloader
        - start_epoch (int): indicates the epochs you are restarting
        from (if 0 it means that it's training from scratch)
        - num_epochs (int): number of epochs e want to train
        - device (str): "cuda" or "cpu", it indicates whether we want
        to use cpu or gpu.
        - run_name (str): path of Tensorboard logger
        - weights_folder (str): path of the folder where to save the
        weights of the trained model
        - name_tensorboard (str): name of plot we want to save the loss fucntion
        into (for Tenaorboard)

    Returns: None
    """
    generator.eval()
    discriminator.train()
    generator.to(device)
    discriminator.to(device)
    # model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    scaler = torch.cuda.amp.GradScaler()

    ## BINARY CROSS ENTROPY + SIGMOID
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")
        tot_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)

            with torch.cuda.amp.autocast():
                fake_hr_images = generator(lr_images)
                out_fake = discriminator(fake_hr_images)
                out_real = discriminator(hr_images)
                loss_real = bce(
                    out_real, torch.ones_like(out_real)
                )
                loss_fake = bce(
                    out_fake, torch.zeros_like(out_fake)
                )
                loss = loss_fake + loss_real
                tot_loss += loss.item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()

            pbar.set_postfix(BCE=loss.item())

        tot_loss /= len_dataloader
        pbar.set_postfix(BCE=tot_loss)

        logger.add_scalars(name_tensorboard, {
                'train_loss': tot_loss,
            }, start_epoch+epoch)

        if epoch%10==9:
            torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+epoch}.pt"))

    torch.save(discriminator.state_dict(), os.path.join(weights_folder, f"discriminator_{start_epoch+num_epochs}.pt"))







def train_generator(model,
                    optimizer,
                    loss_fn,
                    train_dataloader,
                    val_dataloader,
                    start_epoch,
                    num_epochs,
                    device,
                    run_name,
                    weights_folder,
                    num_blocks_gen):
    """
    Pre-trains a "PSNR-based" Generator for the first stage of
    the training.

    Args:
        - model (esrgan.Generator): generator model
        - train_dataloader (DataLoader): training dataloader
        - optimizer (Optimizer): optimizer of the generator model
        - loss_fn (nn.Module): loss function
        - num_epochs (int): number of epochs e want to train
        - device (str): "cuda" or "cpu", it indicates whether we want
       to use cpu or gpu.

    Returns: None
    """
    model.train()
    model.to(device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    len_dataloader = len(train_dataloader)

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Training epoch {start_epoch+epoch}")
        train_tot_loss = 0

        for idx, (hr_images, lr_images) in enumerate(pbar):
            hr_images = hr_images.to(device)
            lr_images = lr_images.to(device)
            fake_hr_images = model(lr_images)

            #print(hr_images.shape, lr_images.shape, fake_hr_images.shape)

            loss = loss_fn(hr_images, fake_hr_images)
            train_tot_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(L1=loss.item())

        train_tot_loss /= len_dataloader
        pbar.set_postfix(TRAIN_L1=train_tot_loss)
        #logger.add_scalar("LOSS FUNCTIONS", tot_loss, global_step=epoch)

        if epoch%5==0:
            val_tot_loss = validation(model=model, loss_fn=loss_fn, val_dataloader=val_dataloader,
                       device=device, epoch=epoch)

            logger.add_scalars(f"LOSS FUNCTIONS {num_blocks_gen} BLOCKS", {
                'train_loss': train_tot_loss,
                'val_loss': val_tot_loss,
            }, start_epoch+epoch)
            torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+epoch}.pt"))
        else:
            logger.add_scalars(f"LOSS FUNCTIONS {num_blocks_gen} BLOCKS", {
                'train_loss': train_tot_loss,
            }, start_epoch+epoch)

    torch.save(model.state_dict(), os.path.join(weights_folder, f"model_{start_epoch+num_epochs}.pt"))





def validation(model,
               loss_fn,
               val_dataloader,
               device,
               epoch):
    """
    Executes one iteration of validation.

    Args:
        - model (esrgan.Generator): generator model
        - val_dataloader (DataLoader): validation dataloader
        - loss_fn (nn.Module): loss function
        - device (str): "cuda" or "cpu", it indicates whether we want
       to use cpu or gpu.

    Returns: None
    """

    len_dataloader = len(val_dataloader)
    pbar = tqdm(val_dataloader)
    tot_loss = 0
    pbar.set_description("Validation")
    for idx, (hr_images, lr_images) in enumerate(pbar):
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)

        fake_hr_images = model(lr_images)

        loss = loss_fn(hr_images, fake_hr_images)
        tot_loss += loss.item()

        pbar.set_postfix(L1=loss.item())

    tot_loss /= len(val_dataloader)

    return tot_loss
