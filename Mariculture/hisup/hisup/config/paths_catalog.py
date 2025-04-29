import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'crowdai_train_small': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation-small.json'
        },
        'crowdai_test_small': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation-small.json'
        },
        'crowdai_train': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation.json'
        },
        'crowdai_test': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation.json'
        },
        'inria_train': {
            'img_dir': 'yangzhiqu/lyg_3/train/images',
            'ann_file': 'yangzhiqu/lyg_3/train/annotation.json',
        },
        'inria_test': {
#             'img_dir': 'yangzhiqu/lyg_3/test',
#             'ann_file': 'yangzhiqu/lyg_3/test/annotation.json',
            'img_dir': 'yangzhiqu/lyg_3/train/images',
            'ann_file': 'yangzhiqu/lyg_3/train/annotation.json',
        },
        'lyg_train': {
            'img_dir': 'yangzhiqu/lyg_3/cut_512_new/images',
            'ann_file': 'yangzhiqu/lyg_3/cut_512_new/annotation.json',
        },
        'lyg_test': {
            'img_dir': 'yangzhiqu/lyg_3/cut_512_new/images',
            'ann_file': 'yangzhiqu/lyg_3/cut_512_new/annotation.json',
#              'img_dir': 'yangzhiqu/lyg_3/cut_300/train/img',
#              'ann_file': 'yangzhiqu/lyg_3/cut_300/train/annotation.json',
        },
        'lyg_mix_train': {
            'img_dir': 'lyg_3_mix/cut_300_new210/images',
            'ann_file': 'lyg_3_mix/cut_300_new210/annotation_mix.json',
#              'img_dir': 'yangzhiqu/lyg_3/cut_300/train/img',
#              'ann_file': 'yangzhiqu/lyg_3/cut_300/train/annotation.json',
        },        
        'lyg_mix_test': {
            'img_dir': 'lyg_3_mix/cut_300_new210/images',
            'ann_file': 'lyg_3_mix/cut_300_new210/annotation_mix.json',
#              'img_dir': 'yangzhiqu/lyg_3/cut_300/train/img',
#              'ann_file': 'yangzhiqu/lyg_3/cut_300/train/annotation.json',
        },   
        'dt_train': {
            'img_dir': 'dongtou/geo/cut_2048_u8_new/train/images',
            'ann_file': 'dongtou/geo/cut_2048_u8_new/train/annotation.json',
        },
        'dt_test': {
            'img_dir': 'dongtou/geo/cut_2048_u8_new/test/images',
            'ann_file': 'dongtou/geo/cut_2048_u8_new/test/annotation.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
