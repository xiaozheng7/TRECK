# Traffic Representation Extraction with Contrastive learning frameworK (TRECK)
![License](https://img.shields.io/badge/license-Apache-green)![Python](https://img.shields.io/badge/-Python-blue)![PyTorch](https://img.shields.io/badge/-PyTorch-red)

Implementation of **TRECK: Long-Term Traffic Forecasting with Contrastive Representation Learning**. This work was published in *IEEE Transactions on Intelligent Transportation Systems* (https://ieeexplore.ieee.org/document/10596071).

<p align="center">
<img src=".\image\fig 2 (a).svg" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> Representation generation with TRECK (encoding stage).
</p>

<p align="center">
<img src=".\image\fig 2 (a).svg" height = "360" alt="" align=center />
<br><br>
<b>Figure 1.</b> Representation generation with TRECK (encoding stage).
</p>


## Datasets
A spatial-temporal traffic flow dataset with 60 detectors spanning 3 years is studied in this paper. They are gathered from vehicle detector stations (VDS) equipped with loop detectors in San Diego, which is collected from [the Caltrans Performance Measurement System (PeMS)](https://pems.dot.ca.gov/). We share this dataset at [Google Drive](https://drive.google.com/file/d/1oqMvSZBfvDbpFwKU4HzqgteyW4Wkpwsj/view?usp=drive_link).

<p align="center">
<img src=".\image\detector_distribution.svg" height = "360" alt="" align=center />
<br><br>
<b>Figure 2.</b> Loop detector distribution.
</p>



## Cite this work
```
@ARTICLE{10596071,
  author={Zheng, Xiao and Bagloee, Saeed Asadi and Sarvi, Majid},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={TRECK: Long-Term Traffic Forecasting With Contrastive Representation Learning}, 
  year={2024},
  volume={25},
  number={11},
  pages={16964-16977},
  keywords={Forecasting;Predictive models;Representation learning;Contrastive learning;Data models;Task analysis;Casting;Traffic forecasting;contrastive learning;LSTM;Transformer;GNN;prediction interval},
  doi={10.1109/TITS.2024.3421328}}

```


