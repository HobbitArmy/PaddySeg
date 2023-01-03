_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/paddy_seg_dsw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]

num_classes = 5
model = dict(
    pretrained=None,
    decode_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(lr=5e-3)

img_norm_cfg = dict(mean=[88.823, 98.088, 71.003], std=[57.845, 56.351, 54.61], to_rgb=True)
IMG_SCALE = (820, 546)  # down sampling
crop_size = IMG_SCALE
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=IMG_SCALE, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=IMG_SCALE,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(samples_per_gpu=4, workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2\segformer_mit-b0_8x1_pdseg_dsw/'
