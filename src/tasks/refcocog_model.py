# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from src.param import args
from src.lxrt.entry import LXRTEncoder
from src.lxrt.modeling_capsbert import BertLayerNorm, GeLU, MLP

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class RefCOCOgModel(nn.Module):
    def __init__(self, train_paradigm='full'):
        super().__init__()
        # train_paradigm has two options: 'full', 'weak'
        # 'full' for bounding box supervision
        # 'weak' for image-text pair supervision
        self.train_paradigm = train_paradigm
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        if self.train_paradigm == 'full':
            # train with bounding box labels
            self.logit_fc = MLP(hid_dim, hid_dim, 4, 3)
        elif self.train_paradigm == 'weak':
            # weak supervision, only use image-text labels
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim * 2, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, 2)
            )
        else:
            raise NotImplementedError()

        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        self.args = args

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """

        _, x, attn_probs = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)
        assert logit.size()[-1] == 2 if self.train_paradigm == 'weak' else \
            logit.size()[-1] == 4 and self.train_paradigm == 'full'

        return logit, attn_probs


