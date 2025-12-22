# Clustering of synapses belonging to inhibitory neurons

## Licensing

- Source code is released under the MIT License.
- Data files are subject to additional usage restrictions.
  See DATA_USAGE.md for details.


The following tables summarize the notebooks, definitions and trained models used in this project. 
Detailed methodology and results are documented inside the individual Jupyter notebooks.

| File name | Description |
|----------|-------------|
| neuron_synapse_clustering_part1.ipynb | Data preprocessing and unsupervised clustering of the historical synapse dataset. This notebook standardizes raw synapse features and explores the intrinsic cluster structure. |
| neuron_synapse_clustering_part2.ipynb | Model training on the historical dataset. Cluster labels are learned separately for major synapse types (excitatory, inhibitory) and neuron types (calbindin, calretinin, parvalbumin). |
| neuron_synapse_clustering_part3.ipynb | Prediction of clusters for newly observed synapses using the previously trained models and dimensionality reduction pipelines. |

| File name | Description |
|----------|-------------|
| pca_models_per_neuronid.pkl | PCA models used for neuron-specific spatial alignment of synapse coordinates (one model per neuron). |
| robust_scaler.pkl           | RobustScaler fitted on the historical dataset to normalize synapse features before dimensionality reduction. |
| umap_as_model.pkl           | Trained UMAP model projecting excitatory (asymmetric, as) synapses into a 2D embedding space. |
| umap_ss_model.pkl           | Trained UMAP model projecting inhibitory (symmetric, ss) synapses into a 2D embedding space. |
| umap_cb_model.pkl           | Trained UMAP model for synapses belonging to calbindin (cb) neurons. |
| umap_cr_model.pkl           | Trained UMAP model for synapses belonging to calretinin (cr) neurons. |
| umap_pv_model.pkl           | Trained UMAP model for synapses belonging to parvalbumin (pv) neurons. |
| as_xgb_bundle.pkl           | Trained XGBoost classification model for excitatory (as) synapses, including preprocessing and outlier detection components. |
| ss_xgb_bundle.pkl           | Trained XGBoost classification model for inhibitory (ss) synapses, including preprocessing and outlier detection components. |
| cb_xgb_bundle.pkl           | Trained XGBoost classification model for calbindin (cb) synapses. |
| cr_knn_bundle.pkl           | Trained K-Nearest Neighbors (KNN) model for calretinin (cr) synapses, including outlier detection. |
| cr_xgb_bundle.pkl           | Trained XGBoost classification model for calretinin (cr) synapses. |
| pv_xgb_bundle.pkl           | Trained XGBoost classification model for parvalbumin (pv) synapses. |


