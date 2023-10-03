# Created by Baole Fang at 7/23/23

import argparse

import yaml
import os
import clip
import numpy as np
from PIL import Image
import torch

from wrapper import Generator
from embedding import get_delta_t
from manipulator import Manipulator
from mapper import get_delta_s

from dash import Dash, dcc, html, Input, Output, ctx, State
import dash_mantine_components as dmc


def setup(model, direction, path, device):
    generator = Generator(model, device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    fs3 = np.load(direction)
    manipulator = Manipulator(generator, device, face_preprocess=False)
    manipulator.set_real_img_projection(path, inv_mode='w', pti_mode='s')
    return model, fs3, manipulator


def get_app(path, slider, targets, H, W):
    app = Dash(__name__)
    images = []
    ids = []
    data = []
    for group, values in targets.items():
        for value in values:
            data.append({'value': value, 'label': value, 'group': group})
    for filename in sorted(os.listdir(path)):
        img = Image.open(os.path.join(path, filename))
        filename = filename.split('.')[0]
        div = html.Div(
            children=[html.Img(id=f'{filename}_orig', src=img, height=H, width=W),
                      html.Img(id=f'{filename}_gen', src=img, height=H, width=W)],
            style={'display': 'flex', 'flex-direction': 'column'}
        )
        images.append(div)
        ids.append(f'{filename}_gen')
    img_layout = html.Div(children=images, style={'display': 'flex', 'flex-direction': 'row', 'flex-flow': 'row wrap'})
    control = html.Div(
        [
            html.Div([dmc.Slider(id="slider", min=slider[0], max=slider[1], step=slider[2], precision=2)],
                     style={'width': '50%', 'display': 'inline-block'}),
            html.Div([dmc.Select(data=data, id="dropdown", value=data[0]['value'], clearable=False, creatable=True,
                                 searchable=True)],
                     style={'width': '30%', 'display': 'inline-block'}),
            html.Div([dcc.Loading(id="loading", children=html.Div(id=ids[0]), type="circle")],
                     style={'width': '10%', 'display': 'inline-block'})
        ]
    )

    app.layout = html.Div([
        html.H1('Facial attribute slider', style={'textAlign': 'center'}),
        img_layout,
        control,
    ])
    return app, ids


def main(config, device, port):
    H, W = config['size']
    neutral = config['neutral']
    beta = config['beta']
    model, fs3, manipulator = setup(config['model'], config['direction'], config['input'], device)
    app, ids = get_app(config['input'], config['alpha'], config['targets'], H, W)
    delta_s = None

    @app.callback(
        [Output(i, 'src') for i in ids],
        [Input('dropdown', 'value'), Input('slider', 'value')]
    )
    def handler(dropdown, slider):
        global delta_s
        if ctx.triggered_id == 'dropdown' or ctx.triggered_id == None:
            classnames = [neutral, dropdown]
            delta_t = get_delta_t(classnames, model)
            delta_s, _ = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta)

        manipulator.set_alpha([-slider])
        styles = manipulator.manipulate(delta_s)
        all_imgs = manipulator.synthesis_from_styles(styles, 0, manipulator.num_images)[0]

        all_imgs = (all_imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()
        imgs = []
        for img in all_imgs:
            imgs.append(Image.fromarray(img, 'RGB').resize((H, W)))
        return imgs

    app.run_server(port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='slider parameters')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--port', type=int, default=8050, help='port')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_ = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(config_, 'cuda', args.port)
