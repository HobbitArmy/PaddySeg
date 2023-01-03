_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/paddy_seg_dsw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]

num_classes = 5

model = dict(pretrained=None,
             backbone=dict(depth=18),
             decode_head=dict(c1_in_channels=64, c1_channels=12,
                              in_channels=512, channels=128, num_classes=num_classes, ),
             auxiliary_head=dict(in_channels=256, channels=64, num_classes=num_classes, ),

             test_cfg=dict(mode='whole'))

optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1, )

work_dir = 'E:\project_data\p6_pdySeg2022/work_dirs2/deeplabv3plus_r18-d8_2x4_60k_pdseg_dsw/'
