# BiBLDR
BiBLDR: Bidirectional Behavior Learning for  Drug Repositioning
![The proposed BiBLDR framework. (a) Utilize similarity data to construct prototype spaces for drugs and diseases separately. (b) Utilize prototypes and bidirectional behavioral sequence information to predict drug-disease associations.](main.png)
## Run
```bash
python train.py -dataset Gdataset -promote_embedding_dim 1024 -rating_T 2
python train.py -dataset Cdataset -promote_embedding_dim 1024 -rating_T 2
python train.py -dataset lrssl -promote_embedding_dim 1024 -rating_T 3
```
