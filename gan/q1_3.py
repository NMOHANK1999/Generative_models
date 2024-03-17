import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # #below method is bad for numerical stability as values less than log 1 as exp high and log 0 is -inf
    # b_s = discrim_real.shape[0]
    # loss = 1/b_s * torch.sum(torch.log(discrim_real) + torch.log(1 - discrim_fake))
    
    func = torch.nn.BCEWithLogitsLoss()  #this is better for numerical stability than just BCE loss, here it adds a sigmoid by itself
    loss = func(torch.cat((discrim_real, discrim_fake), dim = 1),
              torch.cat((torch.ones_like(discrim_real), torch.zeros_like(discrim_fake)), dim = 1))
    
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # b_s = discrim_fake.shape[0]
    # loss = 1/b_s * torch.sum(torch.log(1 - discrim_fake))
    
    func = torch.nn.BCEWithLogitsLoss()
    loss = func(discrim_fake, torch.ones_like(discrim_fake))
    
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)
    torch.cuda.empty_cache()

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1,
        amp_enabled=not args.disable_amp,
    )
