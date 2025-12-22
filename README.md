# Clustering of synapses belonging to inhibitory neurons

## Licensing

- Source code is released under the MIT License.
- Data files are subject to additional usage restrictions.
  See DATA_USAGE.md for details.


The following tables summarize the notebooks, definitions and trained models used in this project. 
Detailed methodology and results are documented inside the individual Jupyter notebooks.  

## Dataset

| File name | Description |
|----------|-------------|
| neuron_synapse_raw_data.xlsx | Raw historical dataset containing synapse features used for clustering and model training. |
| neuron_synapse_clustering_HDBSCAN_result.csv | UMAP embedding coordinates and HDBSCAN cluster labels of the historical synapse dataset. |
| new_neuron_synapse_raw_data.xlsx | Raw dataset of newly observed synapses used for cluster prediction. |

## Notebooks

| File name | Description |
|----------|-------------|
| neuron_synapse_clustering_part1.ipynb | Data preprocessing and unsupervised clustering of the historical synapse dataset. This notebook standardizes raw synapse features and explores the intrinsic cluster structure. |
| neuron_synapse_clustering_part2.ipynb | Model training on the historical dataset. Cluster labels are learned separately for major synapse types (excitatory, inhibitory) and neuron types (calbindin, calretinin, parvalbumin). |
| neuron_synapse_clustering_part3.ipynb | Prediction of clusters for newly observed synapses using the previously trained models and dimensionality reduction pipelines. | 

## Source Code (src)

| File name | Function | Description |
|----------|----------|-------------|
| dimension_reduction.py | pca_align_one_neuron | Performs PCA-based spatial alignment separately for each neuron (NeuronId). |
|  | make_umap | Applies UMAP with configurable 2D or 3D embedding. |
| cluster.py | run_hdbscan_umap | Performs HDBSCAN clustering on UMAP embedding coordinates. |
| plot.py | plot_umap_2d | Visualizes 2D UMAP embeddings. |
|  | plot_umap_3d | Visualizes 3D UMAP embeddings. |
|  | plot_hdbscan_umap_clusters | Visualizes HDBSCAN clustering results, automatically detecting 2D or 3D embeddings. |
| train.py | split_umap_data | Performs train–test split on UMAP coordinates for supervised model training. |
|  | knn_with_outlier_filtering | Trains a KNN classifier with outlier filtering using OneClassSVM and hyperparameter tuning (GridSearchCV). |
|  | plot_confusion_matrix | Plots confusion matrices for KNN and Random Forest models. |
|  | random_forest_with_outlier_filtering | Trains a Random Forest classifier with OneClassSVM-based outlier filtering. |
|  | xgboost_with_outlier_filtering | Trains an XGBoost classifier with OneClassSVM-based outlier filtering and label encoding. |
|  | xgboost_plot_confusion_matrix | Plots confusion matrices for XGBoost models. |
| prediction.py | apply_pca | Applies the previously trained PCA models to new synapse data. |
|  | apply_pca_robust_scaler | Applies the previously fitted RobustScaler to PCA-transformed features of new data. |
|  | umap_model_syntype | Projects new synapses into UMAP space using synapse-type-specific UMAP models (excitatory/inhibitory). |
|  | umap_model_neurontype | Projects new synapses into UMAP space using neuron-type-specific UMAP models. |
|  | predict_clusters | Predicts cluster labels for new synapses using trained classification models. |

 ## Models

| File name | Description |
|----------|-------------|
| pca_models_per_neuronid.pkl           | PCA models used for neuron-specific spatial alignment of synapse coordinates (one model per neuron). |
| robust_scaler.pkl                     | RobustScaler fitted on the historical dataset to normalize synapse features before dimensionality reduction. |
| umap_as_model.pkl                     | Trained UMAP model projecting excitatory (asymmetric, as) synapses into a 2D embedding space. |
| umap_ss_model.pkl                     | Trained UMAP model projecting inhibitory (symmetric, ss) synapses into a 2D embedding space. |
| umap_cb_model.pkl                     | Trained UMAP model for synapses belonging to calbindin (cb) neurons. |
| umap_cr_model.pkl                     | Trained UMAP model for synapses belonging to calretinin (cr) neurons. |
| umap_pv_model.pkl                     | Trained UMAP model for synapses belonging to parvalbumin (pv) neurons. |
| as_xgb_bundle.pkl                     | Trained XGBoost classification model for excitatory (as) synapses, including preprocessing and outlier detection components. |
| ss_xgb_bundle.pkl                     | Trained XGBoost classification model for inhibitory (ss) synapses, including preprocessing and outlier detection components. |
| cb_xgb_bundle.pkl                     | Trained XGBoost classification model for calbindin (cb) synapses. |
| cr_knn_bundle.pkl                     | Trained K-Nearest Neighbors (KNN) model for calretinin (cr) synapses, including outlier detection. |
| cr_xgb_bundle.pkl                     | Trained XGBoost classification model for calretinin (cr) synapses. |
| pv_xgb_bundle.pkl                     | Trained XGBoost classification model for parvalbumin (pv) synapses. |


