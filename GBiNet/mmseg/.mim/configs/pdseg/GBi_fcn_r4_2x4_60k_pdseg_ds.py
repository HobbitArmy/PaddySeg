_base_ = [
    '../_base_/models/GBiT.py',
    '../_base_/datasets/paddy_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]

model = dict(
    backbone=dict(ghost_ratio=4,),
    test_cfg=dict(mode='slide', crop_size=(546, 546), stride=(364, 364)))

data = dict(samples_per_gpu=8,  # 单个 GPU 的 Batch size
            workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
            num_gpus=1, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'E:\project_data\p6_pdySeg2022\/work_dirs2/GBiT_r4_1x8_pdseg_test/'
