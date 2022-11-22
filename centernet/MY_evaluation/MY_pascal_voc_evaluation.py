from detectron2.evaluation.pascal_voc_evaluation import PascalVOCDetectionEvaluator
import logging
import os
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager

class MY_PascalVOCDetectionEvaluator(PascalVOCDetectionEvaluator):
    def __init__(self, dataset_name):
        super.__init__(dataset_name)
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Layout", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)