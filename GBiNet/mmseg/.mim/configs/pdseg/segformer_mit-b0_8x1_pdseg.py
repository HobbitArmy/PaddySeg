_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.pth'  # noqa

num_classes = 5
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)))

# optimizer
optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1)

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2\segformer_mit-b0_8x1_pdseg_40k/'
