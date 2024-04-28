# DCJT
Joint Training on Multiple Datasets With Inconsistent Labeling Criteria for Facial Expression Recognition, IEEE Transactions on Affective Computing

# Training

* Step 1: Download relabeled Affectnet listpath, https://pan.baidu.com/s/1dMJDsTJwu0pFtSaKaTLR8Q?pwd=eg24. And put it into ./listpath

* Step 2: Open './listpath', change path in these txt files. e.g. '/NASdata/frank/emoDataset/RAFDB/Image/aligned/test_0001_aligned.jpg 4' to '/yourpath/yourpath/.../RAFDB/Image/aligned/test_0001_aligned.jpg 4'

* Step 3: run 'python train.py --config configs/rafdb.yaml'

# Citation

```
@article{yu2024joint,
  title={Joint Training on Multiple Datasets With Inconsistent Labeling Criteria for Facial Expression Recognition},
  author={Yu, Chengyan and Zhang, Dong and Zou, Wei and Li, Ming},
  journal={IEEE Transactions on Affective Computing},
  year={2024},
  publisher={IEEE}
}
```


# Note

We refined our codes after the paper was accepted. The best performance using ARM as the backbone on RAF-DB dataset can achieve 93.5%.
