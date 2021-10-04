r"""Corr testing code"""
import argparse

from torch.utils.data import DataLoader
import torch

from mmdet.utils.corr_utils.evaluation import Evaluator
from mmdet.utils.corr_utils.logger import AverageMeter
from mmdet.utils.corr_utils.logger import Logger
from mmdet.utils.corr_utils import utils

# 202101 this script only support ncnet/ancnet style eval now (from trg to src)
from mmdet.models.corr_modules._bases.geom import Geometry

from mmdet.datasets.corr_datasets import download

from mmcv import DictAction

def load_model():
    import torch
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import (load_checkpoint)

    from mmdet.models import build_detector

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    return model

class InferenceCorrHelper:
    def __init__(self, model,fea_layer_id):
        from mmdet.models.corr_modules._bases.features import featureL2Norm
        from mmdet.models.corr_modules._bases.ncons import MutualMatching, FeatureCorrelation
        self.featureL2Norm = featureL2Norm
        self.MutualMatching = MutualMatching
        self.FeatureCorrelation = FeatureCorrelation(normalization=False)

        self.model = model
        self.fea_layer_id = fea_layer_id

        return

    def extract_fea_main(self, src_img, trg_img):
        # if hasattr(self.model,'extract_fea_for_corr_inference'):
        #     src_fea = self.model.extract_fea_for_corr_inference(src_img)
        #     trg_fea = self.model.extract_fea_for_corr_inference(trg_img)
        # else:
        #     src_fea = self._extract_fea_help(img=src_img)
        #     trg_fea = self._extract_fea_help(img=trg_img)

        src_fea = self._extract_fea_help(img=src_img)
        trg_fea = self._extract_fea_help(img=trg_img)

        return src_fea, trg_fea


    def inference_corr_main(self,  src_fea, trg_fea):
        if hasattr(self.model,'bbox_head.semantic_corr_solver.inference_corr'): #todo
            corr4d = self.model.bbox_head.semantic_corr_solver.inference_corr(src_fea, trg_fea)
        else:
            corr4d = self._inference_corr_help(src_fea=src_fea, trg_fea=trg_fea)
        return corr4d

    def _extract_fea_help(self, img):
        fea_all = self.model.extract_feat(img)
        fea = fea_all[self.fea_layer_id]
        return fea

    def _inference_corr_help(self, src_fea, trg_fea):
        src_fea = self.featureL2Norm(src_fea)
        trg_fea = self.featureL2Norm(trg_fea)
        corr4d_raw = self.FeatureCorrelation(src_fea, trg_fea)
        corr4d = self.MutualMatching(corr4d=corr4d_raw)
        b, c, hs, ws, ht, wt = corr4d.shape
        assert (c == 1)
        return corr4d


def test(model, dataloader):
    r"""Code for testing corr"""
    # Miscellaneous todo: inference process may differ with different corr models
    do_softmax = True
    upsample_size = [int(args.img_side / 4)] * 2
    assert (do_softmax == True)  # when transfer kps, input is un-normalized corr, so must do softmax
    Geometry.initialize(upsample_size, device, do_softmax=do_softmax)

    average_meter = AverageMeter(dataloader.dataset.benchmark)
    average_meter.time_start()
    for idx, batch in enumerate(dataloader):
        # 1. forward pass
        src_img, trg_img = batch['src_img'], batch['trg_img']

        src_fea, trg_fea = model.extract_fea_main(src_img=src_img,trg_img=trg_img)

        corr4d = model.inference_corr_main(src_fea=src_fea,trg_fea=trg_fea)

        # 2. Transfer key-points (nearest neighbor assignment)
        prd_kps = Geometry.transfer_kps(corr4d, batch['trg_kps'], batch['n_pts'],img_side=args.img_side)

        # 3. Evaluate predictions
        eval_result = Evaluator.evaluate(prd_kps, batch)

        average_meter.update(eval_result, batch['category'])

        average_meter.write_process(idx, len(dataloader))

    average_meter.time_end()

    # Write evaluation results
    average_meter.write_result('Test')

    avg_pck = utils.mean(average_meter.buffer['pck'])
    cost_time = average_meter.get_cost_time()

    print('Test done! avg_pck {}, cost time {}'.format(avg_pck,cost_time))
    return


if __name__ == '__main__':

    # Arguments parsing
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Semantic Correspondence Evaluation')
    parser.add_argument('--datapath', type=str, default='../data/DHPF_ECCV2020_Datasets')
    parser.add_argument('--benchmark', type=str, default='pfpascal', choices=['pfpascal', 'spair', 'spair-mini','pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--logroot', type=str, default='../ckpts/')
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--exp_name', type=str, default='exp_delete')

    # 20210501
    parser.add_argument('--img_side', type=int, default=240, help='image side')
    parser.add_argument('--fea_layer_id', type=int, default=2,choices=[0,1,2,3],help='select layer of features used for corr inference. 0 for stride4')

    # 20210501 arg from mmdet
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')

    args = parser.parse_args()
    Logger.initialize(args)
    utils.fix_randseed(seed=0)

    # Model initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_seg = load_model().to(device)
    model_seg.eval()
    model_corr = InferenceCorrHelper(model=model_seg,fea_layer_id=args.fea_layer_id)

    # Dataset download & initialization
    download.download_dataset(args.datapath, args.benchmark)
    test_ds = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test',img_side=args.img_side)
    test_dl = DataLoader(test_ds, batch_size=args.bsz, shuffle=False,) # diff from dhpf, be sure to add img side

    Evaluator.initialize(args.benchmark, args.alpha)

    # Test
    # with torch.no_grad(): test(model, test_dl)
    with torch.no_grad(): test(model_corr, test_dl)
