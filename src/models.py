import os
import torch
import torch.nn as nn
import torch.optim as optim
# from models.backbone_baseline import BaselineGenerator, Discriminator
from models.generator_baseline import build_generator_baseline
from models.generator_inptr import build_generator_inptr
from models.discriminator import Discriminator
from util.misc import NestedTensor
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self):
        if self.config.PRETRAINED and self.config.PRETRAINED_PATH is not None:
            load_gen_path = self.config.PRETRAINED_PATH + '_gen.pth'
            load_dis_path = self.config.PRETRAINED_PATH + '_dis.pth'
        else:
            load_gen_path = self.gen_weights_path
            load_dis_path = self.dis_weights_path
        if os.path.exists(load_gen_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(load_gen_path)
            else:
                data = torch.load(load_gen_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(load_dis_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(load_dis_path)
            else:
                data = torch.load(load_dis_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, save_gen_path=None, save_dis_path=None):
        print('\nsaving %s...\n' % self.name)
        if not save_gen_path:
            save_gen_path = self.gen_weights_path
        if not save_dis_path:
            save_dis_path = self.dis_weights_path

        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, save_gen_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, save_dis_path)


class BaseLine(BaseModel):
    def __init__(self, config):
        super(BaseLine, self).__init__('BaseLine', config)
        generator = build_generator_inptr(config)
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPUs) > 1:
            generator = nn.DataParallel(generator, config.GPUs)
            discriminator = nn.DataParallel(discriminator, config.GPUs)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, masks):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_perceptual_loss = self.perceptual_loss(outputs, images)
        gen_perceptual_loss = gen_perceptual_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_perceptual_loss

        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # create logs
        logs = [
            ("l_dis", dis_loss.item()),
            ("l_gan", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_perceptual_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, masks):   # mask: 0 for hole
        images_masked = images * masks.float()

        # inputs = torch.cat((images_masked, masks), dim=1)
        nt = NestedTensor(images_masked, masks)
        outputs = self.generator(nt)
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()