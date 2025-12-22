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


<h2>Dataset</h2>

<table>
  <colgroup>
    <col style="width: 30%">
    <col style="width: 70%">
  </colgroup>
  <thead>
    <tr>
      <th>File name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="neuron_synapse_raw_data.xlsx">neuron_synapse_raw_data.xlsx</a></td>
      <td>Raw historical dataset containing synapse features used for clustering and model training.</td>
    </tr>
    <tr>
      <td><a href="neuron_synapse_clustering_HDBSCAN_result.csv">neuron_synapse_clustering_HDBSCAN_result.csv</a></td>
      <td>UMAP embedding coordinates and HDBSCAN cluster labels of the historical synapse dataset.</td>
    </tr>
    <tr>
      <td><a href="new_neuron_synapse_raw_data.xlsx">new_neuron_synapse_raw_data.xlsx</a></td>
      <td>Raw dataset of newly observed synapses used for cluster prediction.</td>
    </tr>
  </tbody>
</table>


<h2>Notebooks</h2>

<table>
  <colgroup>
    <col style="width: 30%">
    <col style="width: 70%">
  </colgroup>
  <thead>
    <tr>
      <th>File name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="neuron_synapse_clustering_part1.ipynb">neuron_synapse_clustering_part1.ipynb</a></td>
      <td>Data preprocessing and unsupervised clustering of the historical synapse dataset. This notebook standardizes raw synapse features and explores the intrinsic cluster structure.</td>
    </tr>
    <tr>
      <td><a href="neuron_synapse_clustering_part2.ipynb">neuron_synapse_clustering_part2.ipynb</a></td>
      <td>Model training on the historical dataset. Cluster labels are learned separately for major synapse types (excitatory, inhibitory) and neuron types (calbindin, calretinin, parvalbumin).</td>
    </tr>
    <tr>
      <td><a href="neuron_synapse_clustering_part3.ipynb">neuron_synapse_clustering_part3.ipynb</a></td>
      <td>Prediction of clusters for newly observed synapses using the previously trained models and dimensionality reduction pipelines.</td>
    </tr>
  </tbody>
</table>


<h2>Source Code (src)</h2>

<table>
  <colgroup>
    <col style="width: 25%">
    <col style="width: 25%">
    <col style="width: 50%">
  </colgroup>
  <thead>
    <tr>
      <th>File name</th>
      <th>Function</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="src/dimension_reduction.py">dimension_reduction.py</a></td>
      <td>pca_align_one_neuron</td>
      <td>Performs PCA-based spatial alignment separately for each neuron (NeuronId).</td>
    </tr>
    <tr>
      <td><a href="src/dimension_reduction.py">dimension_reduction.py</a></td>
      <td>make_umap</td>
      <td>Applies UMAP with configurable 2D or 3D embedding.</td>
    </tr>
    <tr>
      <td><a href="src/cluster.py">cluster.py</a></td>
      <td>run_hdbscan_umap</td>
      <td>Performs HDBSCAN clustering on UMAP embedding coordinates.</td>
    </tr>
    <tr>
      <td><a href="src/plot.py">plot.py</a></td>
      <td>plot_umap_2d</td>
      <td>Visualizes 2D UMAP embeddings.</td>
    </tr>
    <tr>
      <td><a href="src/plot.py">plot.py</a></td>
      <td>plot_umap_3d</td>
      <td>Visualizes 3D UMAP embeddings.</td>
    </tr>
    <tr>
      <td><a href="src/plot.py">plot.py</a></td>
      <td>plot_hdbscan_umap_clusters</td>
      <td>Visualizes HDBSCAN clustering results, automatically detecting 2D or 3D embeddings.</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>split_umap_data</td>
      <td>Performs train–test split on UMAP coordinates for supervised model training.</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>knn_with_outlier_filtering</td>
      <td>Trains a KNN classifier with outlier filtering using OneClassSVM and hyperparameter tuning (GridSearchCV).</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>plot_confusion_matrix</td>
      <td>Plots confusion matrices for KNN and Random Forest models.</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>random_forest_with_outlier_filtering</td>
      <td>Trains a Random Forest classifier with OneClassSVM-based outlier filtering.</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>xgboost_with_outlier_filtering</td>
      <td>Trains an XGBoost classifier with OneClassSVM-based outlier filtering and label encoding.</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>xgboost_plot_confusion_matrix</td>
      <td>Plots confusion matrices for XGBoost models.</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>apply_pca</td>
      <td>Applies the previously trained PCA models to new synapse data.</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>apply_pca_robust_scaler</td>
      <td>Applies the previously fitted RobustScaler to PCA-transformed features of new data.</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>umap_model_syntype</td>
      <td>Projects new synapses into UMAP space using synapse-type-specific UMAP models (excitatory/inhibitory).</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>umap_model_neurontype</td>
      <td>Projects new synapses into UMAP space using neuron-type-specific UMAP models.</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>predict_clusters</td>
      <td>Predicts cluster labels for new synapses using trained classification models.</td>
    </tr>
  </tbody>
</table>


<h2>Models</h2>

<table>
  <colgroup>
    <col style="width: 30%">
    <col style="width: 70%">
  </colgroup>
  <thead>
    <tr>
      <th>File name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="pca_models_per_neuronid.pkl">pca_models_per_neuronid.pkl</a></td>
      <td>PCA models used for neuron-specific spatial alignment of synapse coordinates (one model per neuron).</td>
    </tr>
    <tr>
      <td><a href="robust_scaler.pkl">robust_scaler.pkl</a></td>
      <td>RobustScaler fitted on the historical dataset to normalize synapse features before dimensionality reduction.</td>
    </tr>
    <tr>
      <td><a href="umap_as_model.pkl">umap_as_model.pkl</a></td>
      <td>Trained UMAP model projecting excitatory (asymmetric, as) synapses into a 2D embedding space.</td>
    </tr>
    <tr>
      <td><a href="umap_ss_model.pkl">umap_ss_model.pkl</a></td>
      <td>Trained UMAP model projecting inhibitory (symmetric, ss) synapses into a 2D embedding space.</td>
    </tr>
    <tr>
      <td><a href="umap_cb_model.pkl">umap_cb_model.pkl</a></td>
      <td>Trained UMAP model for synapses belonging to calbindin (cb) neurons.</td>
    </tr>
    <tr>
      <td><a href="umap_cr_model.pkl">umap_cr_model.pkl</a></td>
      <td>Trained UMAP model for synapses belonging to calretinin (cr) neurons.</td>
    </tr>
    <tr>
      <td><a href="umap_pv_model.pkl">umap_pv_model.pkl</a></td>
      <td>Trained UMAP model for synapses belonging to parvalbumin (pv) neurons.</td>
    </tr>
    <tr>
      <td><a href="as_xgb_bundle.pkl">as_xgb_bundle.pkl</a></td>
      <td>Trained XGBoost classification model for excitatory (as) synapses, including preprocessing and outlier detection components.</td>
    </tr>
    <tr>
      <td><a href="ss_xgb_bundle.pkl">ss_xgb_bundle.pkl</a></td>
      <td>Trained XGBoost classification model for inhibitory (ss) synapses, including preprocessing and outlier detection components.</td>
    </tr>
    <tr>
      <td><a href="cb_xgb_bundle.pkl">cb_xgb_bundle.pkl</a></td>
      <td>Trained XGBoost classification model for calbindin (cb) synapses.</td>
    </tr>
    <tr>
      <td><a href="cr_knn_bundle.pkl">cr_knn_bundle.pkl</a></td>
      <td>Trained K-Nearest Neighbors (KNN) model for calretinin (cr) synapses, including outlier detection.</td>
    </tr>
    <tr>
      <td><a href="cr_xgb_bundle.pkl">cr_xgb_bundle.pkl</a></td>
      <td>Trained XGBoost classification model for calretinin (cr) synapses.</td>
    </tr>
    <tr>
      <td><a href="pv_xgb_bundle.pkl">pv_xgb_bundle.pkl</a></td>
      <td>Trained XGBoost classification model for parvalbumin (pv) synapses.</td>
    </tr>
  </tbody>
</table>


