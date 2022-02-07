# MainTUL

PyTorch implementation for paper: Mutual Distillation Learning Network for Trajectory-User Linking

For IJCAI'22 review


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
  

## Citation

If you want to use our codes in your research, please cite:

```
@article{MainTUL22,
  title     = {Mutual Distillation Learning Network for Trajectory-User Linking},
  author    = {Anonymous}
```

## Acknowledgement