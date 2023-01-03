_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/fcn_hr18s_512x1024_40k_cityscapes.pth'  # noqa

num_classes = 5
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2,)),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))),
    decode_head=dict(num_classes=num_classes),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)
                  ))

optimizer = dict(lr=5e-3)

data = dict(samples_per_gpu=8, workers_per_gpu=1, )

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/fcn_hr18s_512x1024_40k_pdseg_40k/'
