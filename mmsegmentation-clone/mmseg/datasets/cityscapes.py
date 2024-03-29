# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from PIL import Image
import random
import os

from .builder import DATASETS
from .custom import CustomDataset, newCustomDataset
from .pipelines import newLoadAnnotations

from collections import OrderedDict
from prettytable import PrettyTable

import time

@DATASETS.register_module()
class CityscapesDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs):
        super(CityscapesDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.

        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(CityscapesDataset,
                      self).evaluate(results, metrics, logger))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = imgfile_prefix

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        return eval_results


@DATASETS.register_module(name='newCityscapesDataset')
class newCityscapesDataset(newCustomDataset):
    """Cityscapes dataset.
    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelTrainIds.png',
                 **kwargs):
        super(newCityscapesDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.
        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results,
                if not set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)

        return result_files

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.
        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(newCityscapesDataset,
                      self).evaluate(results, metrics, logger))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.
        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file
        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_dir = imgfile_prefix

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        return eval_results

from mmseg.core.evaluation import get_palette


@DATASETS.register_module(name='newCityscapesDataset1')
class newCityscapesDataset1(newCityscapesDataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                'bicycle')
    PALETTE = get_palette('cityscapes')

    def __init__(self, img_suffix='.png', seg_map_suffix='.png', **kwargs):
        super(newCityscapesDataset1, self).__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
        assert osp.exists(self.img_dir) #and self.split is not None

        self.gt_seg_map_loader = newLoadAnnotations()
        self.seqs_list = [] # stores the filenames of images inside the sequences folder
        self.seq_dir = kwargs.get('seq_dir', None) # comes from kwargs (I think)
        self.optflow_dir = kwargs.get('optflow_dir', None) # comes from kwargs (I think)
        self.tc_eval = kwargs.get('tc_eval', False) # when tc_eval=True, opt_flow refers directly to images in img_dir
        self.depth_dir = kwargs.get('depth_dir', None)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                            self.ann_dir,
                                            self.seg_map_suffix, self.seq_dir, self.optflow_dir, self.split, self.depth_dir, tc_eval=self.tc_eval)
  


    # here, we only set the filenames
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, seq_dir, optflow_dir, split, depth_dir=None, depth_suffix='.png', tc_eval=False):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                line_content = line.strip()
                img_name = line_content
                
                if seq_dir is not None:
                    if 'ZED' in seq_dir:
                        img_name = line_content.split('**')[0]
                

                img_info = dict(filename=img_name + img_suffix)
                
                if ann_dir is not None:
                    seg_map = img_name.replace('leftImg8bit', 'gtFine_labelIds') + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                

                # for Cityscapes, seqdir == '.../sequences/images/train/'
                if seq_dir is not None:
                    flow_suffix = '.flo'

                    if 'Cityscapes' in seq_dir:
                        city_name = img_name.split('_')[0]
                        img_name_parts = img_name.split('_')
                        # prev_idx = int(img_name_parts[2])-1 # img_name_parts[2] encodes the frame position in the sequence
                        img_name_parts[2] = '000018' # we know current frame idx is 000019
                        prev_img_name = '_'.join(img_name_parts)
                        s1_name = osp.join(seq_dir, city_name, prev_img_name+img_suffix) # previous frame in the sequence
                        s2_name = osp.join(seq_dir, city_name, img_name+img_suffix)

                        img_info['s1'] = dict(filename=s1_name)
                        img_info['s2'] = dict(filename=s2_name)
                        img_info['from_target'] = False



                        if optflow_dir is not None:
                            # forward optical flow
                            optflow_name = osp.join(optflow_dir, city_name, prev_img_name.replace('leftImg8bit', 'opt_flow')+flow_suffix)
                            
                            if not osp.exists(optflow_name):
                                continue

                            img_info['optflow'] = dict(filename=optflow_name) # the optical flow is computed from frame in t to t+1, (SURE?)
                            # hence, we select the optical flow corresponding to frame t (in this case, idx)

                    else: # for now, if not Cityscapes, it must be ZED2... in the future, we will also consider the SYNTHIA dataset
                        # ZED2 dataset
                        seq_file = line_content.split('**')[1]
                        seq_name = seq_file.split(osp.sep)[0]
                        seq_file_name = seq_file.split(osp.sep)[1]

                        seqs_list = sorted(os.listdir(osp.join(seq_dir, seq_name)), key=lambda x: int(x.split('_')[-2]))
                        idx = seqs_list.index(seq_file_name)
                        # sequences information
                        if not osp.exists(osp.join(seq_dir, seq_name, seq_file_name)):
                            continue
                        
                        img_info['s1'] = dict(filename=osp.join(seq_dir, seq_name, seq_file_name))
                        img_info['from_target'] = True

                        if idx+1 < len(seqs_list):
                            if not osp.exists(osp.join(seq_dir, seq_name, seqs_list[idx+1])):
                                continue

                            img_info['s2'] = dict(filename=osp.join(seq_dir, seq_name, seqs_list[idx+1]))

                        else:
                            continue # skip to the next iteration, and discard the current content of img_info
                        
                        if optflow_dir is not None:

                            if not osp.exists(osp.join(optflow_dir, seq_name, seq_file_name.replace('leftImg8bit.png', 'opt_flow.flo'))):
                                continue

                            # forward optical flow
                            # optflow_list = sorted(os.listdir(osp.join(optflow_dir, seq_name)), key=lambda x: int(x.split('_')[-3]))
                            img_info['optflow'] = dict(filename=osp.join(optflow_dir, seq_name, seq_file_name.replace('leftImg8bit.png', 'opt_flow.flo'))) # the optical flow is computed from frame in t to t+1,
                            # hence, we select the optical flow corresponding to frame t (in this case, idx)

                if depth_dir is not None:
                    depth_name = osp.join(depth_dir, img_name.split('_')[0], img_name.replace('leftImg8bit', 'disparity') + depth_suffix)

                    if not osp.exists(depth_name):
                        continue
                    
                    img_info['gt_depth'] = dict(filename=depth_name)

                img_infos.append(img_info)

        else: #TO DO (DOING ... DONE)
            filenames = sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[-2]))

            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)

                if ann_dir is not None:
                    seg_map = img.replace('leftImg8bit', 'gtFine_labelIds')
                    img_info['ann'] = dict(seg_map=seg_map)
                
                if depth_dir is not None:
                    img_info['gt_depth'] = dict(filename=osp.join(depth_dir, img_name.split('_')[0], img_name.replace('leftImg8bit', 'disparity') + depth_suffix))
                
                if tc_eval: # in tc_eval, we evaluate the overall temporal consistency of the input sequence
                    idx = filenames.index(img) # find index corresponding to current image name
                    if idx < len(filenames)-1:
                        # img_info['optflow'] = img.replace('leftImg8bit.png', 'opt_flow.flo')
                        img_info['optflow'] = dict(filename=osp.join(optflow_dir, img.replace('leftImg8bit.png', 'opt_flow.flo')))
                    else: # the last image in the sequence doesn't have a neighboring frame, neither optical flow associated with it
                        continue # jump to the next iteration and discard current img_info
                else:
                    if seq_dir is not None:
                        self.seqs_list = sorted(os.listdir(seq_dir))
                        idx = random.randint(0,len(self.seqs_list)-2)
                        # sequences information
                        img_info['s1'] = dict(filename=osp.join(seq_dir, self.seqs_list[idx]))
                        img_info['s2'] = dict(filename=osp.join(seq_dir, self.seqs_list[idx+1]))
                        
                        if optflow_dir is not None:
                            # self.optflow_list = sorted(os.listdir(optflow_dir))
                            img_info['optflow'] = dict(filename=osp.join(optflow_dir, self.seqs_list[idx].replace('leftImg8bit', 'opt_flow').replace('png','flo'))) # the optical flow is computed from frame in t to t+1,
                            # hence, we select the optical flow corresponding to frame t (in this case, idx)

                img_infos.append(img_info)

            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


    def evaluate(self,
                results,
                metric='mIoU',
                logger=None,
                gt_seg_maps=None,
                **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict segmentation map for computing evaluation
                metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset
        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'TC(mDice)', 'TC(mIoU)']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))


        optflows = kwargs.get('optflows', None)
        names = kwargs.get('names', None)

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            
            if len(set(metric).intersection({'mIoU','mDice','mFscore'})) > 0:
                if gt_seg_maps is None:
                    gt_seg_maps = self.get_gt_seg_maps()
                    if isinstance(gt_seg_maps, tuple):
                        gt_seg_maps = gt_seg_maps[0]
                        depth_map = gt_seg_maps[1]


            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label,
                tc_eval = self.tc_eval,
                optflows = optflows,
                names = names)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results