_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/pspnet_r18-d8_512x1024_80k_cityscapes.pth'  # noqa

num_classes = 5
model = dict(
    pretrained=checkpoint,
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512, channels=128, num_classes=num_classes),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=num_classes,),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)))

optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1, )

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/pspnet_r18-d8_512x1024_pdseg_60k/'
