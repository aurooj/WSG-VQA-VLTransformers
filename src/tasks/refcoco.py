# coding=utf-8
# Copyleft 2019 project LXRT.

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import collections

import gc
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from src import eval_utils
from src.box_utils import xywh2xyxy, xyxy2xywh, generalized_box_iou
from src.param import args
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.refcoco_model import RefCOCOModel
from src.tasks.refcoco_data import RefCOCODataset, RefCOCOTorchDataset, RefCOCOEvaluator

print(args)
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = RefCOCODataset(splits)
    tset = RefCOCOTorchDataset(dset)
    evaluator = RefCOCOEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

def giou_loss(batch_pred, batch_target):
    #xywh2xyxy takes center pixel, w, h i.e. xc, yc, w, h
    loss_giou = 1 - torch.diag(generalized_box_iou(
        xywh2xyxy(batch_pred),
        xywh2xyxy(batch_target)
    ))
    return loss_giou

class RefCOCO:
    def __init__(self):
        self.weakly_supervise = args.train_paradigm == 'weak'
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = args.batch_size #2048 if args.multiGPU else args.batch_size#512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = RefCOCOModel(train_paradigm=args.train_paradigm)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = self.model.to(self.device)
        if args.multiGPU and self.device == 'cuda':
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.l1_loss = nn.L1Loss(reduction='mean')
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from src.lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output

        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            # log_str = ''
            sentid2pbox = {}
            for i, (sent_id, feats, boxes, sent, target, target_params, is_matched) in iter_wrapper(enumerate(loader)):
                if i == 0:
                    print(target[0])
                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target, target_params, is_matched = feats.to(self.device), boxes.to(self.device), \
                                                   target.to(self.device), target_params.to(self.device), is_matched.to(self.device)

                logit, param_logit, attn_probs = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2 or logit.dim() == target.dim() == 4

                if self.weakly_supervise:
                    loss = self.bce_loss(logit, is_matched)
                else:
                    loss1 = self.l1_loss(logit.float(), target.float())
                    loss2 = self.l1_loss(param_logit.float(), target_params.float())

                    # loss = loss / logit.shape[0]
                    # loss_giou = 0.0
                    loss_giou = giou_loss(logit, target)
                    loss = loss1 + loss2 + loss_giou
                    # loss_giou = loss_giou.sum() / logit.shape[0]
                    # losses = loss + loss_giou


                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                if self.weakly_supervise:
                    print(
                        'Epoch: {epoch}, Iteration: {iter}, loss: {loss:.6f}'.format(
                            epoch=epoch,
                            iter=i,
                            loss=loss.item()
                        ))
                else:
                    miou, accu = eval_utils.trans_vg_eval_val(logit, target)
                    print('Epoch: {epoch}, Iteration: {iter}, loss: {loss:.6f}, miou: {miou:.4f}, Accuracy: {acc:.4f}'.format(
                        epoch=epoch,
                        iter=i,
                        loss=loss.item(),
                        miou=miou.detach().mean().cpu().numpy(),
                        acc=accu
                    ))
                    #todo: fix evaluation code for ref expression task
                    pred_boxes = eval_utils.get_pred_boxes(logit)
                    # score, label = logit.max(1)
                    for sid, pbox in zip(sent_id, pred_boxes.cpu().detach().numpy()):
                        sentid2pbox[sid] = pbox

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, self.evaluate(train_tuple) * 100.)

            # to handle GPU OOM error
            # torch.cuda.empty_cache()

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        sentid2ans = {}
        results = []
        for i, datum_tuple in enumerate(loader):
            # ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            sent_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            attention = []

            with torch.no_grad():
                feats, boxes = feats.to(self.device), boxes.to(self.device)
                logits, logit_params, attn_probs  = self.model(feats, boxes, sent)

                # print(attn_probs)
                if self.model.args.output_attention:
                    last_layer_att_score = torch.squeeze(attn_probs[1][-1]['attn'][:, :, 0, :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                    # print(last_layer_att_score.shape)
                    last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()
                else:
                    last_layer_att_score = []

                # score, label = logit.max(1)
                pred_boxes = eval_utils.get_pred_boxes(logits)

                for j, sid in enumerate(sent_id):
                    # ans = dset.label2ans[l]
                    sentid2ans[sid] = logits[j]
                    results.append(
                        {
                            "questionId": sid.tolist(),
                            "prediction": pred_boxes[j].cpu().numpy().tolist(),
                            "attention": last_layer_att_score
                        }
                    )

            # del logit, attn_probs, datum_tuple
            # gc.collect()
        exp_name = args.output.split('/')[-1]
        evaluator.save_json(results, 'snap/refcoco/{output_dir}/attentions.json'.format(output_dir=exp_name))

        if dump is not None:
            evaluator.dump_result(sentid2ans, dump)
        return sentid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        sentid2box = self.predict(eval_tuple, dump)
        return evaluator.evaluate(sentid2box)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        sentid2box = {}
        for i, (ques_id, feats, boxes, sent, target_box, target_params, is_matched) in enumerate(loader):
            # target_ = torch.stack(target_box, dim=0).permute(1, 0)
            miou, acc = eval_utils.trans_vg_eval_val(target_box, target_box, oracle=True)
            # _, label = target_box.max(1)
            for sid, iou in zip(ques_id, miou.cpu().numpy()):
                # ans = dset.label2ans[l]
                sentid2box[sid] = iou
        return evaluator.iou_acc(sentid2box)/dset.__len__


    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path, map_location=torch.device('cpu'))
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Build Class
    refcoco = RefCOCO()

    # Load Model
    if args.load is not None:
        refcoco.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            refcoco.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'test' in args.test:
            result = refcoco.evaluate(
                get_tuple('test', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
        if 'valid' in args.test:
            result = refcoco.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'valid_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', refcoco.train_tuple.dataset.splits)
        if refcoco.valid_tuple is not None:
            print('Splits in Valid data:', refcoco.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (refcoco.oracle_score(refcoco.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        refcoco.train(refcoco.train_tuple, refcoco.valid_tuple)


