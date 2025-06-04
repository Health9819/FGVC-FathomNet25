# FathomNet25 Solution

Hello this is Team 911's solution for FathomNet25, where we ranked 4/79 finally.

We are obliged to attend such an exciting and thrilling competition, and willing to share our solution, experience with the community.

Our model is trained on 4 * vGPU(32G)

## Model Selection

At the beginning, we tried the Bio-Clip model which is recommended by the hoster, however, we find Clip-like models having difficulties in recognizing these ocean species.The acc of Bio-Clip is quite low and to ulitize the trainning data we decide to use supervised learning models such as Swin-Transformer, ResNet.

After trying many models and their variations, we decided to use Swin-Transformer.The pre-trained swin model we use is `swin_large_patch4_window7_224_22k` from `timm` as our backbone.

In fact, we tried the higher resolution version swin_large_384 model but end up worse performance on the test set. Thanks to other participates for pointing out that the **average image size of the test set is smaller than that of the training set**. So we guess higher resolution Swin-Transformer would not be better than lower one, and after weighing the model size and computing resources, we decided to use swin-l-224.

## Methods

Inspired by the paper "Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020)", we added Conv layers and MLP after the 4 stages of Swin.This could just help the model to learn the features from different levels.

And we also notice that the score of the competition is defined by the avg error between true label and predicted label.This gives us an inspiration that if we could just predict a class with the mininum error score rather than the class with the highest probability.So we just calculated the distance matrix of the classes and use it to minimize the error score.

We only use the ROIS images provided in our training and testing progress, which may lead to the worse score than the Top3 teams.

We also use some other tricks to help the model speed up training and convergence, this could be seen in our codes.

## Results
Our results are saved in a pkl file and run `result.py' could get the final csv. The result can be calculated by top1-possiblity or minimal score(by calculating the expection of the label prediction).
The results_v4.pkl reflects our best score(with expection).