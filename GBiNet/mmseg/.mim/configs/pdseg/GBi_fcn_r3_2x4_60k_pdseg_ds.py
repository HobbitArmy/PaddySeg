_base_ = [
    '../_base_/models/GBiT.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        detail_channels=(63, 63, 126), ghost_ratio=3,
        semantic_channels=(16, 32, 64, 126),
        semantic_expansion_ratio=6,
        bga_channels=126, ),
    decode_head=dict(
        type='FCNHead',
        in_channels=126,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=5,
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
            num_classes=5,
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
            num_classes=5,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=126,
            channels=1024,
            num_convs=2,
            num_classes=5,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)))

data = dict(samples_per_gpu=8,  # 单个 GPU 的 Batch size
            workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
            num_gpus=1, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/GBiT_r3_1x8_pdseg_test/'
