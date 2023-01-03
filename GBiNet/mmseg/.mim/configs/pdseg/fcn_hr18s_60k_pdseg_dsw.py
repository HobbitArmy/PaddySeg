_base_ = [
    '../_base_/models/fcn_hr18.py',
    '../_base_/datasets/paddy_seg_dsw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]


num_classes = 5
model = dict(
    pretrained=None,
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2,)),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))),
    decode_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='whole'))

optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1, )

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/fcn_hr18s_60k_pdseg_dsw/'
