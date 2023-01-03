_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/paddy_seg_ds_546x.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 5
model = dict(pretrained=None,
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

data = dict(samples_per_gpu=4,  # 单个 GPU 的 Batch size
            workers_per_gpu=2,  # 单个 GPU 分配的数据加载线程数
            num_gpus=2, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'G:/luxy/pds2022/work_dirs3/bisenetv2_fcn_2x4_60k_pdseg_ds/'
