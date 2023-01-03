_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint = 'E:/project_data/pre_w/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.pth'  # noqa

model = dict(
    decode_head=dict(num_classes=5,
                     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                                      class_weight=[0.621, 1.4926, 0.8392, 1.5485, 1.1334])),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(546, 546)),

    # auxiliary_head=[dict(type='FCNHead',in_channels=16,channels=16,num_classes=9,),
    #                 dict(type='FCNHead',in_channels=32,channels=64,num_classes=9,),
    #                 dict(type='FCNHead',in_channels=64,channels=256,num_classes=9,),
    #                 dict(type='FCNHead',in_channels=128, channels=1024,num_classes=9,),]

)

# lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=5e-3)
data = dict(samples_per_gpu=8, workers_per_gpu=1, )


work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/bisenetv2_fcn_4x4_pdseg_40k_cls_wei/'
