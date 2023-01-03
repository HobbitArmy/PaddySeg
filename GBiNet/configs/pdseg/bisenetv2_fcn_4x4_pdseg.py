_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.pth'  # noqa

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 5
model = dict(pretrained=checkpoint,
             decode_head=dict(num_classes=num_classes),
             auxiliary_head=[
                 dict(
                     type='FCNHead',
                     in_channels=16,
                     channels=16,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=1,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=32,
                     channels=64,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=2,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=64,
                     channels=256,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=3,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
                 dict(
                     type='FCNHead',
                     in_channels=128,
                     channels=1024,
                     num_convs=2,
                     num_classes=num_classes,
                     in_index=4,
                     norm_cfg=norm_cfg,
                     concat_input=False,
                     align_corners=False,
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
             ],

             test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)))

# lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1, )

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs3/bisenetv2_fcn_4x4_pdseg_40k/'
