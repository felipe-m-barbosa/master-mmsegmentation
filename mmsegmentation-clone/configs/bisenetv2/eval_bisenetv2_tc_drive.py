_base_ = [
    'bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py'
]

import os.path as osp

# Modify dataset type and path
# dataset_type = 'newCityscapesDataset1'
data_root = '/gdrive/MyDrive/Mestrado/datasets/Cityscapes/'

dataset_type = 'newCityscapesDataset1'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)

test_pipeline = [
    dict(type='newLoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 240),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='newResize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'optflow']),
            dict(type='Collect', keys=['img', 'optflow']),
        ])
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root = data_root,
        img_dir ='demoVideo/stuttgart_00/leftImg8bit',
        # ann_dir = 'gtFine/test',
        optflow_dir = osp.join(data_root, 'demoVideo/stuttgart_00/opt_flow'),
        pipeline = test_pipeline,
        tc_eval=True
        # split = 'splits/test.txt'
    )
)