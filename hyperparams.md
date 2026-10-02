# Glacier hyperparams


| Hyperparameter  |    OLS     | Ridge ($L_2$) |      Lasso ($L_1$)      | ElasticNet ($L_1 + L_2$) |
| :-------------- | :--------: |:-------------:| :---------------------: | :----------------------: |
| `fit_intercept` |    Yes     |      Yes      |           Yes           |           Yes            |
| `alpha`         |     —      |      Yes      |           Yes           |           Yes            |
| `l1_ratio`      |     —      |       —       |            —            |           Yes            |
| `max_iter`      |     —      |      Yes      |           Yes           |           Yes            |
| `tolerance`     |     —      |      Yes      |           Yes           |           Yes            |


# mlpack hyperparams

MLPack only provides L2 regularized Linear regression


| Hyperparameter                      | Ridge ($L_2$) |
|:------------------------------------|:-------------:|
| `data` (dataset)                    |               |
| `responses` (len of xpred)          |               |
| `weights` (predertermined weights_) |               |
| `intercept` (fit_intercept)         |      Yes      |
| `lambda` (alpha)                    |      Yes      |
