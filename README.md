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

## Models

#### Pretrained Model: (Due to the limitation of upload speed, I will update the link as soon as possible)
- Foursquare-mini:<>
- Foursquare-all:<>
- Weeplaces-mini:<>
- Weeplaces-all:<>

## Usage
- make preprocessing original data, the processed format is like the data in folder [data](./data). (if you want user other dataset)
- run main.py
- Adjust the hyperparameters and strategies according to the needs

## Citation

If you want to use our codes in your research, please cite:

```
@article{MainTUL22,
  title     = {Mutual Distillation Learning Network for Trajectory-User Linking},
  author    = {Anonymous}
```

## Acknowledgement