# BiBLDR: Bidirectional Behavior Learning for  Drug Repositioning
## Abstract
Drug repositioning aims to identify potential new indications for existing drugs to reduce the time and financial costs associated with developing new drugs. Most existing deep learning-based drug repositioning methods predominantly utilize graph-based representations. However, drug-disease associations are highly sparse in large biomedical graphs, with weakly connected edges limiting their ability to model complex interactions. The heterogeneity of these graphs adds complexity, and scaling them raises computational and storage costs, hindering practical model deployment. Unlike traditional graph-based approaches, we propose a bidirectional behavior learning strategy for drug repositioning, known as BiBLDR. This innovative framework redefines drug repositioning as a behavior sequential learning task to capture drug-disease interaction patterns. First, we construct bidirectional behavioral sequences based on drug and disease sides. The consideration of bidirectional information ensures a more meticulous and rigorous characterization of the behavioral sequences. Subsequently, we propose a two-stage strategy for drug repositioning. In the first stage, we construct prototype spaces to characterize the representational attributes of drugs and diseases. In the second stage, these refined prototypes and bidirectional behavior sequence data are leveraged to predict potential drug-disease associations. Based on this learning approach, the model can more robustly and precisely capture the interactive relationships between drug and disease features from bidirectional behavioral sequences. Extensive experiments demonstrate that our method achieves state-of-the-art performance on benchmark datasets.

![The proposed BiBLDR framework. (a) Utilize similarity data to construct prototype spaces for drugs and diseases separately. (b) Utilize prototypes and bidirectional behavioral sequence information to predict drug-disease associations.](main.png)
## Requirements
```bash
pip install -r requirements.txt
```
## Run
### Gdataset
```bash
python train.py -dataset Gdataset -promote_embedding_dim 1024 -rating_T 2
```
### Cdataset
```bash
python train.py -dataset Cdataset -promote_embedding_dim 1024 -rating_T 2
```
### LRSSL
```bash
python train.py -dataset lrssl -promote_embedding_dim 1024 -rating_T 3
```
