import os
import itertools
from typing import Any, Dict, List, Set
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import atexit
import bisect
import multiprocessing as mp

import torch
from torch import nn

from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data.detection_utils import read_image
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.visualizer_vintext import decoder
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from projects.SWINTS.swints import add_SWINTS_config

device = "cuda" if torch.cuda.is_available() else "cpu"

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5

class Extractor(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, confidence_threshold, path):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "sem_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
            )
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            instances = instances[instances.scores > confidence_threshold]
            predictions["instances"] = instances
            vis_output = visualizer.draw_instance_predictions(predictions=instances, path=path)

        return predictions, vis_output

def ctc_decode_recognition(rec):
    CTLABELS = [" ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
                "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
                "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "ˋ", "ˊ", "﹒", "ˀ", "˜", "ˇ", "ˆ", "˒", "‑"]
    # ctc decoding
    last_char = False
    s = ''
    for c in rec:
        c = int(c)
        if 0<c < 107:
            s += CTLABELS[c-1]
            last_char = c
        elif c == 0:
            s += u''
        else:
            last_char = False
    if len(s) == 0:
        s = ' '
    s = decoder(s)

    return s

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_SWINTS_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    extractor = Extractor(cfg)

    BASE_DIR = "/kaggle/input/train-vis/ViSignboardVQA/train_image"
    for image_file in tqdm(os.listdir(BASE_DIR), desc="Extracting"):
        image_id = int(image_file.split(".")[0].split("_")[-1])
        img = read_image(os.path.join(BASE_DIR, image_file), format="BGR")
        predictions, visualized_output = extractor.run_on_image(img, 0.3, os.path.join(BASE_DIR, image_file))
        result = predictions["instances"]
        rec = result.pred_rec
        detected_texts = [ctc_decode_recognition(rrec) for rrec in rec]
        scores = result.scores.tolist()
        h, w = result.image_size
        boxes = result.pred_boxes.tensor.detach().cpu() / torch.tensor([w, h, w, h])
        features = {
            "det_features": result.det_features.detach().cpu().numpy(),
            "rec_features": result.rec_features.detach().cpu().numpy(),
            "scores": scores,
            "texts": detected_texts,
            "boxes": boxes.numpy()
        }
        np.save(f"/kaggle/working/train_private_ocr/{image_id}.npy", features, allow_pickle=True)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
