# TMFND

 Trustworthy multimodal fake news detection

# Preparation

```bash
pip install -r requirements.txt
```

# Datasets

**Twitter:** We follow [CAFE](https://github.com/cyxanna/CAFE) to download the split dataset at [https://pan.baidu.com/s/1Vn75mXe69jC9txqB81QzUQ](https://pan.baidu.com/s/1Vn75mXe69jC9txqB81QzUQ) ( extraction code: 78uo ).

**Weibo**: We use the dataset provided from [EANN-KDD18](https://github.com/yaqingwang/EANN-KDD18), you can download image datas (nonrumor_images/rumor_images) via [https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing](https://drive.google.com/file/d/14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn/view?usp=sharing). (Approximately 1.3GB)

**Fakeddit**: We provide  the proceesed dataset, and just download image datas from the official repository [Fakeddit](https://github.com/entitize/Fakeddit) via [https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing](https://drive.google.com/file/d/1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b/view?usp=sharing). (Approximately 113.6GB) The label is different from above datasets, and 1 is the real news.

# Train and Test

## Training a fake news detection model

```bash
bash ./shells/train_twitter.sh
bash ./shells/train_weibo.sh
bash ./shells/train_fakeddit.sh
```

## Calibration model

```bash
bash ./shells/train_calibration_twitter.sh
bash ./shells/train_calibration_weibo.sh
bash ./shells/train_calibration_fakeddit.sh
```
