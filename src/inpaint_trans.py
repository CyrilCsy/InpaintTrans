import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import BaseLine as baseline
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy


class InpaintTransformer():
    def __init__(self, config):
        self.config = config

        self.debug = False
        self.model = baseline(config).to(config.DEVICE)
        self.model_name = self.model.name

        self.psnr = PSNR(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_MASK_FLIST, augment=False, mask_reverse=config.MASK_REVERSE, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_MASK_FLIST, augment=True, mask_reverse=config.MASK_REVERSE, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_MASK_FLIST, augment=False, mask_reverse=config.MASK_REVERSE, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')

    def load(self):
        self.model.load()

    def save(self, save_gen_path=None, save_dis_path=None):
        self.model.save(save_gen_path, save_dis_path)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.THREADS,
            drop_last=True,
            shuffle=True
        )

        keep_training = True
        epoch = int(float((self.config.START_EPOCH)))
        max_epochs = int(float((self.config.MAX_EPOCHS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while keep_training:
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                self.model.train()

                images, masks = self.cuda(*items)

                # train
                outputs, gen_loss, dis_loss, logs = self.model.process(images, masks)
                outputs_comp = (images * masks) + (outputs * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_comp))
                mae = (torch.sum(torch.abs(images - outputs_comp)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.model.backward(gen_loss, dis_loss)
                iteration = self.model.iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL0 and iteration % self.config.SAVE_INTERVAL0 == 0:
                    self.save()

                if self.config.SAVE_INTERVAL1 and iteration % self.config.SAVE_INTERVAL1 == 0:
                    gen_weights_path = os.path.join(self.config.PATH, self.model_name + '_' + str(iteration) + '_gen' + '.pth')
                    dis_weights_path = os.path.join(self.config.PATH, self.model_name + '_' + str(iteration) + '_dis' + '.pth')
                    self.save(gen_weights_path, dis_weights_path)

            epoch += 1
            if epoch > max_epochs:
                keep_training = False
                break

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            images, masks = self.cuda(*items)

            # eval
            outputs, gen_loss, dis_loss, logs = self.model.process(images, masks)
            outputs_comp = (images * masks) + (outputs * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_comp))
            mae = (torch.sum(torch.abs(images - outputs_comp)) / torch.sum(images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))
            logs = logs + logs

        logs = [("it", iteration), ] + logs
        progbar.add(len(images), values=logs)

    def test(self):
        self.model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images, masks = self.cuda(*items)
            index += 1


            outputs = self.model(images, masks)
            outputs_comp = (images * masks) + (outputs * (1 - masks))

            output = self.postprocess(outputs_comp)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)

            imsave(output, path)

            if self.debug:
                image_masked = self.postprocess(images * masks + (1 - masks))[0]
                fname, fext = name.split('.')

                imsave(image_masked, os.path.join(self.results_path + '_masked', fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.model.eval()

        items = next(self.sample_iterator)
        images, masks = self.cuda(*items)

        iteration = self.model.iteration
        inputs = images * masks.float()
        outputs = self.model(images, masks)
        outputs_comp = (images * masks) + (outputs * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(outputs),
            self.postprocess(outputs_comp),
            img_per_row = image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
