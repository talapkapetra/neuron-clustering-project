# Clustering Synapses of Inhibitory Neurons Using Machine Learning

## Licensing

- Source code is released under the MIT License.
- Data files are subject to additional usage restrictions.
  See DATA_USAGE.md for details.

## Project Overview

The goal of this project is to build a machine learning pipeline for clustering synapses (neuronal connections) of inhibitory neurons based on their spatial and morphological properties.

Understanding the organization of synaptic inputs is essential for designing biologically realistic artificial neural networks and computational models. While excitatory neurons (such as pyramidal cells) are relatively well characterized, comprehensive datasets describing diverse inhibitory neuron types are still limited due to the technical challenges of high-resolution synapse characterization.

This project uses a curated dataset of synapses belonging to calcium-binding protein (CBP) expressing inhibitory neurons, specifically calbindin, calretinin, and parvalbumin neurons. These proteins play a critical role in regulating intracellular calcium concentration, which is fundamental for neuronal signal transmission. In the primary visual cortex, approximately 95% of inhibitory neurons belong to this CBP category.

The synapse dataset contains key morphological and spatial features extracted from mouse primary visual cortex, including:

* Area and volume of axon terminals (boutons) connecting to dendritic synaptic fields

* Area of synaptic contact regions on dendrites

Three-dimensional coordinates of synaptic field centers and their distance from the cell body (soma) at nanometer resolution

(Data were acquired using combined light and electron microscopy.)

Using these features, the project investigates whether synapses form distinct clusters across:

Major synapse types (excitatory vs. inhibitory; asymmetric vs. symmetric ultrastructure)

Synapses belonging to specific CBP neuron types (calbindin, calretinin, parvalbumin)

Unsupervised clustering is performed using **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise), which is well suited for spatial datasets with variable density. The resulting clusters are then used to train supervised, **tree-based models** (Random Forest and XGBoost) for cluster prediction of newly observed synapses.

Although the dataset contains a limited number of synapses for some neuron types, robust and biologically meaningful clusters could be identified. These clusters provide a reliable basis for predictive modeling and offer valuable insights into synaptic organization relevant for fine-tuning neuronal network simulations and computational models.


The following tables summarize the datasets, notebooks, definitions and trained models used in this project. 
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


