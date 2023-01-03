_base_ = [
    '../_base_/models/GBiNet.py',
    '../_base_/datasets/paddy_seg_dsw.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]

model = dict(
    test_cfg=dict(mode='whole'))

data = dict(samples_per_gpu=8,  # 单个 GPU 的 Batch size
            workers_per_gpu=1,  # 单个 GPU 分配的数据加载线程数
            num_gpus=1, )  # GPU 数量

optimizer = dict(lr=5e-3)
evaluation = dict(save_best='auto')

work_dir = 'G:/luxy/pds2022/work_dirs3/GBiNet_2x4_60k_pdseg_dsw/'
