# Weakly Supervised Grounding for VQA in Vision-Language Transformers [ECCV 2022]

[Aisha Urooj Khan](https://aishaurooj.wixsite.com/aishaurooj), [Hilde Kuehne](https://hildekuehne.github.io/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Niels Da Vitoria Lobo](https://www.crcv.ucf.edu/person/niels-lobo/), [Mubarak Shah](https://www.crcv.ucf.edu/person/mubarak-shah/)

[`Website`]() | [`arXiv`]() | [`BibTeX`]()

Official Pytorch implementation and pre-trained models for Weakly Supervised Grounding for VQA in Vision-Language Transformers (coming soon).

## Abstract
Transformers for visual-language representation learning have been getting a lot of interest and shown tremendous performance on visual question answering (VQA) and grounding. But most systems that show good performance of those tasks still rely on pre-trained object detectors during training, which limits their applicability to the object classes available for those detectors. 
To mitigate this limitation, the following paper focuses on the problem of weakly supervised grounding in context of visual question answering in transformers.  The approach leverages capsules by grouping each visual token in the visual encoder and uses activations from language self-attention layers as a text-guided selection module to mask those capsules before they are forwarded to the next layer. 
We evaluate our approach on the challenging GQA as well as VQA-HAT dataset for VQA grounding.
Our experiments show that: while removing the information of masked objects from standard transformer architectures leads to a significant drop in performance, the integration of capsules significantly improves the grounding ability of such systems and provides new state-of-the-art results compared to other approaches in the field.



