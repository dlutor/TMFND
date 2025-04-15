# Confidence-Aware Multimodal Learning for Trustworthy FakeNews Detection

[paper]()

With the rapid growth of social media, fake news has become widespread. Recent studies have focused on multimodal information for improved detection, but two issues remain: first, existing methods neglect the effectiveness of multimodal semantic information, leading to noise and irrelevant data; second, existing models lack reliability in output, with confidence calibration methods lacking theoretical guarantees. To address these, we propose a confidence-aware multimodal learning framework with two key contributions: a novel fusion module for effective multimodal integration and a post-hoc confidence calibration method with theoretical guarantees. Experimental results on three public datasets show that our framework outperforms existing models.

![Image 2](results/fig2_5.png)

# Preparation

The framework is compatible with Python 3.12 and torch >= 2.3.0. Make sure to install the following dependencies:
```bash
scipy
scikit-learn
tqdm
transformers
timm
```

```bash
pip install -r requirements.txt
```

# Datasets
Place the downloaded dataset images in the corresponding directory under `datasets`.

**Twitter:** We follow [CAFE](https://github.com/cyxanna/CAFE) to download the split dataset at [https://pan.baidu.com/s/1Vn75mXe69jC9txqB81QzUQ](https://pan.baidu.com/s/1Vn75mXe69jC9txqB81QzUQ) ( extraction code: 78uo ).

**Weibo**: We use the dataset provided from [EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18), you can download image datas (nonrumor_images/rumor_images) via [https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing](https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing). (Approximately 1.3GB)

**Fakeddit**: We provide  the proceesed dataset, and just download image datas from the official repository [Fakeddit](https://github.com/entitize/Fakeddit) via [https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing](https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing). (Approximately 113.6GB) The label is different from above datasets, and 1 is the real news.

# Train and Test

## Training a fake news detection model
Model training can be started using the following script:

```bash
bash ./scripts/train_twitter.sh
bash ./scripts/train_weibo.sh
bash ./scripts/train_fakeddit.sh
```

## Calibrate model
After training, perform confidence calibration.
```bash
bash ./scripts/train_calibration_twitter.sh
bash ./scripts/train_calibration_weibo.sh
bash ./scripts/train_calibration_fakeddit.sh
```

## Citation
If you find our work useful in your research please consider citing our paper:
<!-- ```
@misc{CacheTest,
  author =        {T. Ralphs},
  publisher =     {INFORMS Journal on Computing},
  title =         {{CacheTest}},
  year =          {2020},
  doi =           {10.1287/ijoc.2019.0000.cd},
  url =           {https://github.com/INFORMSJoC/2019.0000},
  note =          {Available for download at https://github.com/INFORMSJoC/2019.0000},
}  
``` -->