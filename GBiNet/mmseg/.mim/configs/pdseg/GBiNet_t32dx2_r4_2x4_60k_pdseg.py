_base_ = [
    '../_base_/models/GBiNet_t.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes=5
ghost_ratio=4                       # 2
detail_channels=(16, 16, 32)        # (64, 64, 128)
semantic_channels=(8, 16, 16, 32)  # (16, 32, 64, 128)
decode_channels = 64               # 1024
semantic_expansion_ratio=4
model = dict(
    backbone=dict(
        detail_channels=detail_channels,
        ghost_ratio=ghost_ratio,
        semantic_channels=semantic_channels,
        semantic_expansion_ratio=semantic_expansion_ratio,
        bga_channels=semantic_channels[-1],
    ),
    decode_head=dict(
        in_channels=semantic_channels[-1],
        in_index=0,
        channels=decode_channels,
        num_convs=1,
        ghost_ratio=ghost_ratio,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=semantic_channels[0],
            channels=semantic_channels[0],
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
            in_channels=semantic_channels[1],
            channels=semantic_channels[1]*2,
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
            in_channels=semantic_channels[2],
            channels=semantic_channels[2]*4,
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
            in_channels=semantic_channels[3],
            channels=semantic_channels[3]*8,
            num_convs=2,
            num_classes=num_classes,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # test_cfg=dict(mode='whole')
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364))
)

data = dict(samples_per_gpu=8,  # 单个 GPU 的 Batch size
            workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
            num_gpus=1, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/GBiNet_t32dx2_r4_2x4_60k_pdseg/'
