_base_ = [
    '../_base_/models/lraspp_m-v3-d8.py',
    '../_base_/datasets/paddy_seg_dsw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]

num_classes = 5
norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
model = dict(
    pretrained=None,
    type='EncoderDecoder',
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(0, 1, 12),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='LRASPPHead',
        in_channels=(16, 16, 576),
        in_index=(0, 1, 2),
        channels=128,
        input_transform='multiple_select',
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole')
    # test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364))
)

data = dict(samples_per_gpu=8,  # 单个 GPU 的 Batch size
            workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
            num_gpus=1, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2\lraspp_m-v3s-d8_8x1_pdseg_dsw/'
