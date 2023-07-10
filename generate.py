# Created by Baole Fang at 7/9/23
import argparse

import yaml
import os
import clip
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from embedding import get_delta_t
from manipulator import Manipulator
from mapper import get_delta_s
from wrapper import Generator


def prepare(config, device):
    generator = Generator(config['model'], device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    fs3 = np.load(config['direction'])
    manipulator = Manipulator(generator, device)
    manipulator.set_real_img_projection(config['input'], inv_mode='w+', pti_mode='s')
    return model, fs3, manipulator


def generate(model, fs3, manipulator, neutral, target, beta, alpha, output, filenames):
    classnames = [neutral, target]
    delta_t = get_delta_t(classnames, model)
    delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta)
    # print(f'{num_channel} channels will be manipulated under the beta threshold {beta}')

    manipulator.set_alpha(alpha)
    styles = manipulator.manipulate(delta_s)
    all_imgs = manipulator.synthesis_from_styles(styles, 0, manipulator.num_images)
    save(all_imgs, alpha, os.path.join(output, target), filenames)


def save(all_imgs, lst_alpha, output, filenames):
    lst = []
    for imgs in all_imgs:
        lst.append((imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy())

    for i, alpha in enumerate(lst_alpha):
        imgs = lst[i]
        path = os.path.join(output, '{0:.1f}'.format(alpha))
        if not os.path.exists(path):
            os.makedirs(path)
        for j, img in enumerate(imgs):
            Image.fromarray(img, 'RGB').save(os.path.join(path, filenames[j]))


def main(config, device):
    model, fs3, manipulator = prepare(config, device)
    alpha = list(range(*config['alpha'][:-1]))
    for i in range(len(alpha)):
        alpha[i] /= config['alpha'][-1]
    filenames = sorted(os.listdir(config['input']))
    phar = tqdm(config['targets'])
    for target in phar:
        phar.set_postfix_str(target)
        generate(model, fs3, manipulator, config['neutral'], target, config['beta'], alpha, config['output'], filenames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slider parameters')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    main(config_, args.device)
