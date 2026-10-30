# Model artifact notice

This folder contains the fitted classifier, preprocessing configuration, evaluation report, confusion matrix, and split manifest for Musify's four-source, 12-label YAMNet + PANNs fusion model.

## Training provenance

The classifier was fitted using GTZAN, mapped genres from FMA-Small, Saraga Hindustani, and Saraga Carnatic. Its input combines pooled YAMNet frame embeddings with PANNs CNN14 embeddings. The audio datasets and the YAMNet/PANNs pretrained feature-extractor files are not included here.

## Rights and reuse

The root MIT license covers Musify-authored source code only. It does not grant rights to this trained artifact, the datasets used to train it, or the pretrained feature extractors. Saraga is identified by the Mirdata dataset documentation as CC BY-NC-SA 4.0. FMA's license terms apply per track, and the GTZAN Kaggle listing used during development did not provide a clear reuse license. Whether model weights are treated as adapted material under each source's terms has not been determined here.

This artifact is included for the non-commercial academic internship submission. Preserve dataset and model attribution, review the applicable terms for each source, and obtain any required permission before redistributing it further or using it commercially. This notice is not a legal determination or a license grant.

## Evaluation limits

The report records 79.53% pooled accuracy and 0.8238 pooled macro-F1 on 679 held-out tracks. Earlier experiments used the GTZAN, FMA-Small, and Hindustani Saraga held-out splits, so this is not a blind final estimate. The Indian-classical test tracks come only from Saraga and are limited in number; these results do not establish generalization to other recordings or collections.
