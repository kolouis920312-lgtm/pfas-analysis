# legacy/ — 整合前的舊獨立腳本（已封存，僅供參考）

這些是把分析功能**整合進 `pfas_toolkit` 套件之前**的原始獨立腳本。功能已全部被
`pfas_toolkit/methods/` 的對應模組取代，且**沒有任何程式 import 它們**；保留只為追溯歷史。

| 舊腳本 | 現行對應（請改用） |
|---|---|
| `pca_analysis.py` | `pfas_toolkit/methods/pca.py` |
| `hca_analysis.py` | `pfas_toolkit/methods/hca.py` |
| `kmeans_pca.py` | `pfas_toolkit/methods/kmeans.py` |
| `xgboost_regression.py` | `pfas_toolkit/methods/xgboost_reg.py` |
| `ml_drivers.py` | `pfas_toolkit/methods/ml_drivers.py` |
| `bdl_censored.py` | `pfas_toolkit/methods/bdl.py` |
| `coda_transform.py` | `pfas_toolkit/methods/coda.py` |
| `som_fingerprint.py` | `pfas_toolkit/methods/som.py` |
| `nonparam_stats.py` | `pfas_toolkit/methods/nonparam.py` |

新功能/修改請一律改 `pfas_toolkit/`；這裡的檔案不再維護。
