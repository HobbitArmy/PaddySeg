_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/segformer_mit-b1_8x1_1024x1024_160k_cityscapes.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint), embed_dims=64),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=5),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(546, 546)))  # 测试模块，‘whole’：整个图像全卷积测试，‘slide’：裁剪图像

# optimizer
optimizer = dict(lr=5e-3)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=1)

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2\segformer_mit-b1_8x1_pdseg_40k_SGD/'
