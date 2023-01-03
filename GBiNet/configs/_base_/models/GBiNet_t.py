# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

num_classes=5

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='GBiNet_t',
        detail_channels=(16, 16, 32),
        ghost_ratio=4,
        semantic_channels=(8, 16, 16, 32),
        semantic_expansion_ratio=4,
        bga_channels=32,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False),
    decode_head=dict(
        type='GCNHead',
        in_channels=32,
        in_index=0,
        channels=32,
        num_convs=1,
        kernel_size=3,
        ghost_ratio=4,
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
            in_channels=8,
            channels=8,
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
            in_channels=16,
            channels=32,
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
            in_channels=16,
            channels=64,
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
            in_channels=32,
            channels=256,
            num_convs=2,
            num_classes=num_classes,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
