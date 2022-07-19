# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.param import args
from src.utils import load_obj_tsv, load_spatial_data

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),

class MSCOCODataset:
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
            if split == 'train':
                self.data.extend(
                json.load(open("/media/data/data/data/lxmert/mscoco_%s.json" % split)))
            else:
                self.data.extend(json.load(open("/media/data/data/data/lxmert/mscoco_karpathy_retrieval_%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        data_flattened = self.flatten_data()
        self.data = data_flattened
        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['uid']: datum
            for datum in self.data
        }

        # Answers
        # self.ans2label = json.load(open("data/refcoco/trainval_ans2label.json"))
        # self.label2ans = json.load(open("data/refcoco/trainval_label2ans.json"))
        # assert len(self.ans2label) == len(self.label2ans)
        # for ans, label in self.ans2label.items():
        #     assert self.label2ans[label] == ans

    def flatten_data(self):
        data_flattened = []
        for datum in self.data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat == 'mscoco':
                    # print(sents_cat)
                    if sents_cat in datum['labelf']:
                        labels = datum['labelf'][sents_cat]
                    else:
                        labels = None
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent_id': sent_idx,
                            'sent': sent
                        }
                        if labels is not None:
                            new_datum['label'] = labels[sent_idx]
                        data_flattened.append(new_datum)
                        break
        print("Use %d data in torch dataset" % (len(data_flattened)))
        return data_flattened

    @property
    # def num_answers(self):
    #     return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class MSCOCOBufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        # if name == 'testdev':
        #     # path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        #     path = "data/refcoco/refcoco_testdev_spatial.h5"
        # else:
        #     # path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        #     path = "data/refcoco/refcoco_testdev_spatial.h5"
        path = "/media/data/data/data/mscoco_imgfeat/{}_features.hdf5".format(name)
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_spatial_data(
                path,
                topk=number
            )
        return self.key2data[key]


mscoco_buffer_loader = MSCOCOBufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class MSCOCOTorchDataset(Dataset):
    def __init__(self, dataset: MSCOCODataset):
        super().__init__()
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
            img_data.extend(mscoco_buffer_loader.load_data('test', -1))
        elif 'valid' in dataset.splits or 'valid' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(mscoco_buffer_loader.load_data('valid', -1))
        else:
            img_data.extend(mscoco_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        sent_id = datum['uid']
        sent = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        # boxes = img_info['boxes'].copy()

        feats = img_info['features'].copy()
        ##Aisha change:

        boxes = np.ones(feats.shape[1]*feats.shape[2]+1, dtype=np.float32) #assuming feats of shape [d, h, w]
        # assert len(boxes) == len(feats) == obj_num

        # target_box = datum['refBox']
        # # Normalize the boxes (to 0 ~ 1)
        # img_h, img_w = img_info['img_h'], img_info['img_w']
        # target_box = target_box.copy()
        # # target_box[:, (0, 2)] /= img_w
        # # target_box[:, (1, 3)] /= img_h
        # target_box[0] /= img_w
        # target_box[2] /= img_w
        # target_box[1] /= img_h
        # target_box[3] /= img_h
        # np.testing.assert_array_less(np.array(target_box), 1+1e-5)
        # np.testing.assert_array_less(-np.array(target_box), 0+1e-5)

        # Create target
        # if 'label' in datum:
        #     label = datum['label']
        #     target = torch.zeros(self.raw_dataset.num_answers)
        #     for ans, score in label.items():
        #         if ans in self.raw_dataset.ans2label:
        #             target[self.raw_dataset.ans2label[ans]] = score
        #     return ref_id, feats, target_box, sent, target
        # else:
        return sent_id, feats, boxes, sent


class MSCOCOEvaluator:
    def __init__(self, dataset: MSCOCODataset):
        self.dataset = dataset

    def evaluate(self, sentid2box: dict):
        score = 0.
        for sentid, box in sentid2box.items():
            datum = self.dataset.id2datum[sentid]
            label = datum['refBox']
            if box in label:
                score += label[box]
        return score / len(sentid2box)

    def save_json(self, data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


