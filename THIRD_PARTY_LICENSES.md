# Third-party notices

This project combines Musify-authored code with references to separate audio models and a separately obtained dataset. Each component retains its own terms. The root LICENSE applies to Musify-authored code only and does not grant rights to external model weights, model files, or dataset audio.

## PANNs CNN14 architecture

src/genre_classifier/panns.py contains a minimal Cnn14_16k implementation adapted from Qiuqiang Kong and contributors' PANNs repository:
https://github.com/qiuqiangkong/audioset_tagging_cnn

The upstream repository provides this architecture under the MIT License. The adaptation is attributed in the source file, and the MIT notice is reproduced here:

~~~text
The MIT License

Copyright (c) 2018-2020 Qiuqiang Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~

The pretrained CNN14 checkpoint is not included. The setup helper downloads it from the PANNs Zenodo record:
https://zenodo.org/records/3987831
and checks the published MD5. Check the current record and its terms before downloading, using, or redistributing the checkpoint.

## YAMNet

The YAMNet SavedModel is not included. The README points to the TensorFlow Hub YAMNet model as the source for a local download:
https://tfhub.dev/google/yamnet/1

Follow the model and source repository terms for the downloaded files; they are kept outside Git.

## GTZAN

GTZAN audio is not included. Its examples contributed to the included four-source classifier. The Kaggle distribution used during development labeled its license “Other (specified in description),” and its description did not provide additional reuse terms when checked on 2026-10-21. This note is not a conclusion that redistribution is permitted. Check the current Kaggle listing and original dataset-provider terms before reuse or redistribution.

## Saraga Hindustani and Carnatic datasets

The four-source model artifact in `artifacts/musify_four_dataset_hybrid_v2/` was trained in part with Saraga Hindustani and Carnatic examples from Zenodo record 4301737 (https://zenodo.org/records/4301737). The Mirdata dataset documentation identifies the Saraga collection as CC BY-NC-SA 4.0: https://mirdata.readthedocs.io/en/stable/_modules/mirdata/datasets/saraga_hindustani.html. The audio and extracted feature caches are excluded from Git; the fitted classifier is included with its own `MODEL_NOTICE.md`. This repository does not determine whether the trained weights are an adapted work under that license. Preserve attribution and review component terms before further redistribution. Do not infer commercial rights from the root MIT license.

## FMA-Small

The four-source model includes mapped Hip-Hop, Pop, and Rock examples from FMA-Small. FMA audio and feature caches are excluded, but the fitted classifier is included. The FMA terms direct users to the license applying to each track; retired FMA-Limited terms prohibit redistribution. See https://freemusicarchive.org/terms_of_use/ and https://freemusicarchive.org/FAQ/. Review the licenses for the source tracks before redistributing this model. This repository does not determine whether or how those terms apply to trained weights.
## Packages

Python and JavaScript packages remain subject to the terms published by their respective maintainers. This file is an attribution and scope note, not legal advice or a grant of rights.
