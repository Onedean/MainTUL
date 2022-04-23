# MainTUL

PyTorch implementation for paper: Mutual Distillation Learning Network for Trajectory-User Linking (IJCAI'22)

## Dependencies

- Python 3.9
- torch==1.10.1 (cuda10.2)
- scikit-learn==1.0.1
- tqdm==4.62.3
- pandas==1.3.4
- matplotlib==3.5.0


## Datasets

#### Raw data：
- Foursquare:  <http://sites.google.com/site/yangdingqi/home/foursquare-dataset>
- Weeplaces: <http://www.yongliu.org/datasets.html>

#### Prepocessed data：

- The prepocessed data is uploaded in the folder [data](./data).
- We provide two forms for each dataset, one is full data and the other is sampled small data (for quick testing).

## Usage
- Preprocess original data. (If you want to use other dataset, please preprocess the dataset into the data format under the folder data.)
- Run main.py
- Adjust the hyperparameters and strategies according to the needs
  - e.g. ```python main.py --temperature 0.1/2/5/10/15 --lambda_parm 1/5/10/15/20```

## Parameter setting of baselines
  + Tradition Methods:  
    + DT: (Entropy)  
    + LDA: (LDA Matrix solver: SVD)  
    + LCSS  
    + SR: (spatial signature / Reduced (m = 10))  
  
  + Deep Learning Methods:  
    + TULER and its variants: (LR: 0.00095 / Dimension: 250 / Hidden size: 300 / Dropout rate: 0.5 / Layers: 2)  
    + TULVAE: (LR: 0.001 / decays: 0.9 / $\beta$: 0.5-1 / POI Dimension: 250 / Hidden size: 300 / Latent Variable Dimension: 100)  
    + DeepTUL: ($D_p$: 64 / $D_t$: 32 / $D_u$: 32 / LR: 0.005/ decays: 0.5 / time interval: 120)

## Citation

If you want to use our codes in your research, please cite:

```
@inproceedings{chen2022MainTUL,
  title={Mutual Distillation Learning Network for Trajectory-User Linking},
  author={Chen, Wei and Li, Shuzhe and Huang, Chao and Yu, Yanwei and Jiang, Yongguo and Dong, Junyu},
  booktitle={IJCAI},
  year={2022}
}
```

## Acknowledgement