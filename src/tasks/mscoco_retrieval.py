# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import numpy as np
import gc
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from src.param import args
from src.pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.mscoco_retrieval_model import MSCOCOModel
from src.tasks.mscoco_retrieval_data import MSCOCODataset, MSCOCOTorchDataset, MSCOCOEvaluator
from src.compute_metrics import compute_metrics, print_computed_metrics, retrieval_metrics

print(args)
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = MSCOCODataset(splits)
    tset = MSCOCOTorchDataset(dset)
    evaluator = MSCOCOEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class MSCOCO:
    def __init__(self):
        self.root_dir = ' '
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

        self.model = MSCOCOModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
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
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit, attn_probs = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # del logit, attn_probs
                # gc.collect()

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            # to handle GPU OOM error
            torch.cuda.empty_cache()

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

    # def Eval_retrieval(model, eval_dataloader, dataset_name):
    #     model.eval()
    #     print('Evaluating retrieval on {} data'.format(dataset_name))
    #     with th.no_grad():
    #         for data in eval_dataloader:
    #             video = data['video'].cuda()
    #             audio = data['audio'].cuda()
    #             nframes = data['nframes'].cuda()
    #             if args.tri_modal:
    #                 text = data['text'].cuda()
    #
    #                 if args.use_text and args.use_audio:  # Performs T->A+V
    #                     video, audio, text = model(video, audio, nframes, text)
    #                     m = (th.matmul(text, video.t()) + th.matmul(text, audio.t())).cpu().detach().numpy()
    #                 else:  # Performs T->V
    #                     video, text = model(video, audio, nframes, text)
    #                     m = th.matmul(text, video.t()).cpu().detach().numpy()
    #             else:  # Performs A->V
    #                 video, audio = model(video, audio, nframes)
    #                 m = th.matmul(audio, video.t()).cpu().detach().numpy()
    #
    #             metrics = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
    #             print_computed_metrics(metrics)

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        results = []
        lang_ft = torch.zeros(5000, 768).cuda()
        visn_ft = torch.zeros(5000, 768).cuda()
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            attention = []

            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                feats, x, attn_probs  = self.model(feats, boxes, sent)
                lang_feats, visn_feats = feats[0], feats[1]
                bsz = lang_feats.shape[0]
                lang_ft[i*bsz: (i+1)*bsz] = lang_feats[:,0]
                visn_ft[i * bsz: (i+1)*bsz] = visn_feats[:, 0]

                # print(attn_probs)
                # if self.model.args.output_attention:
                #     last_layer_att_score = torch.squeeze(attn_probs[1][-1]['attn'][:, :, 0, :])  # batch_size, att_head, target_num_feat, source_num_feat -> use all att head and CLS as target
                #     # print(last_layer_att_score.shape)
                #     last_layer_att_score = last_layer_att_score.cpu().numpy().tolist()
                # else:
                #     last_layer_att_score = []
                #
                # # score, label = logit.max(1)
                # for qid in ques_id:
                #     # ans = dset.label2ans[l]
                #     # quesid2ans[qid] = ans
                #     results.append(
                #         {
                #             "questionId": qid.tolist(),
                #             # "prediction": ans,
                #             "attention": last_layer_att_score
                #         }
                #     )

            # del logit, attn_probs, datum_tuple
            # gc.collect()
        # lang_ft = torch.nn.functional.normalize(lang_ft)
        # visn_ft = torch.nn.functional.normalize(visn_ft)
        m = torch.matmul(lang_ft, visn_ft.t()).cpu().detach().numpy()

        # Retrieving video clips given input language
        # metrics = retrieval_metrics(m)
        # print(metrics)
        metrics = compute_metrics(m, eval_lang_retrieval=False, eval_msrvtt=False)
        print_computed_metrics(metrics)
        # Retrieving language given input video clips
        metrics = compute_metrics(m, eval_lang_retrieval=True, eval_msrvtt=False)
        print_computed_metrics(metrics)
        evaluator.save_json(results, os.path.join(self.root_dir, 'snap/mscoco/metrics.json'))

        # if dump is not None:
        #     evaluator.dump_result(quesid2ans, dump)
        # return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        self.predict(eval_tuple, dump)
        # return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # Build Class
    mscoco = MSCOCO()

    # Load Model
    if args.load is not None:
        mscoco.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            mscoco.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'test' in args.test:
            result = mscoco.evaluate(
                get_tuple('test', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
            print(result)
        if 'valid' in args.test:
            result = mscoco.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'valid_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', mscoco.train_tuple.dataset.splits)
        if mscoco.valid_tuple is not None:
            print('Splits in Valid data:', mscoco.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (mscoco.oracle_score(mscoco.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        mscoco.train(mscoco.train_tuple, mscoco.valid_tuple)


