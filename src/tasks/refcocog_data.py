# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src import eval_utils
from src.param import args
from src.utils import load_obj_tsv, load_spatial_data

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class RefCOCOgDataset:
    """
    A GQA data example in json file:
    {
        "caption": caption,
        "sent_id": sent_id,
        "image_id": image_id,
        "refBox": refBox,
        "ref_id": ref_id, --> unique id assigned to each data sample
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("../../data/refcocog/annotations_%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))


        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['sent_id']: datum
            for datum in self.data
        }

        # Answers
        # self.ans2label = json.load(open("data/refcoco/trainval_ans2label.json"))
        # self.label2ans = json.load(open("data/refcoco/trainval_label2ans.json"))
        # assert len(self.ans2label) == len(self.label2ans)
        # for ans, label in self.ans2label.items():
        #     assert self.label2ans[label] == ans

    @property
    # def num_answers(self):
    #     return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class RefCOCOgBufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        # if name == 'testdev':
        #     # path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        #     path = "data/refcoco/refcoco_testdev_spatial.h5"
        # else:
        #     # path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        #     path = "data/refcoco/refcoco_testdev_spatial.h5"
        path = "../../data/refcocog/{}_features.hdf5".format(name)
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_spatial_data(
                path,
                topk=number
            )
        return self.key2data[key]


refcocog_buffer_loader = RefCOCOgBufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class RefCOCOgTorchDataset(Dataset):
    def __init__(self, dataset: RefCOCOgDataset):
        super().__init__()
        self.weakly_supervise = args.train_paradigm == 'weak'
        self.raw_dataset = dataset
        # self.img_info_data = json.load('data/gqa/gqa_spatial_merged_info.json')

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'test' in dataset.splits or 'test' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(refcocog_buffer_loader.load_data('test', -1))
        elif 'valid' in dataset.splits or 'valid' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(refcocog_buffer_loader.load_data('valid', -1))
        else:
            img_data.extend(refcocog_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['image_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['image_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['image_id']
        sent_id = datum['sent_id']
        sent = datum['caption']

        # If weakly supervision, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        if self.weakly_supervise:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data) - 1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data) - 1)]
                sent = other_datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        # boxes = img_info['boxes'].copy()

        feats = img_info['features'].copy()
        ##Aisha change:

        boxes = np.ones(feats.shape[1]*feats.shape[2]+1, dtype=np.float32) #assuming feats of shape [d, h, w]
        # assert len(boxes) == len(feats) == obj_num

        target_box = datum['refBox']
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        target_box = target_box.copy()
        # target_box[:, (0, 2)] /= img_w
        # target_box[:, (1, 3)] /= img_h
        target_box[0] /= img_w
        target_box[2] /= img_w
        target_box[1] /= img_h
        target_box[3] /= img_h
        np.testing.assert_array_less(np.array(target_box), 1+1e-5)
        np.testing.assert_array_less(-np.array(target_box), 0+1e-5)

        # Create target
        # if 'label' in datum:
        #     label = datum['label']
        #     target = torch.zeros(self.raw_dataset.num_answers)
        #     for ans, score in label.items():
        #         if ans in self.raw_dataset.ans2label:
        #             target[self.raw_dataset.ans2label[ans]] = score
        #     return ref_id, feats, target_box, sent, target
        # else:
        return sent_id, feats, boxes, sent, torch.tensor(target_box), is_matched


class RefCOCOgEvaluator:
    def __init__(self, dataset: RefCOCOgDataset):
        self.dataset = dataset

    def evaluate(self, sentid2box: dict):
        sid2iou = {}

        for sentid, pred_box in sentid2box.items():
            datum = self.dataset.id2datum[sentid.item()]
            gt_box = datum['refBox']
            miou, accu = eval_utils.trans_vg_eval_val(torch.as_tensor(pred_box), torch.as_tensor(gt_box))
            sid2iou[sentid] = miou.detach().cpu().numpy()

        accu = self.iou_acc(sid2iou)
        return accu.float() / len(sentid2box)

    def iou_acc(self, sid2iou):
        accu = torch.sum(torch.FloatTensor(list(sid2iou.values())) >= 0.5)
        return accu

    def save_json(self, data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f)

    def dump_result(self, sentid2box: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param sentid2box: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for sent_id, box in sentid2box.items():
                result.append({
                    'questionId': sent_id,
                    'prediction': box
                })
            json.dump(result, f, indent=4, sort_keys=True)


