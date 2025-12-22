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
  <thead>
    <tr>
      <th>File name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="neuron_synapse_clustering_part1.ipynb">neuron_synapse_clustering_part1.ipynb</a></td>
      <td>Data preprocessing and unsupervised clustering of the historical synapse dataset.</td>
    </tr>
    <tr>
      <td><a href="neuron_synapse_clustering_part2.ipynb">neuron_synapse_clustering_part2.ipynb</a></td>
      <td>Model training on the historical dataset for synapse and neuron types.</td>
    </tr>
    <tr>
      <td><a href="neuron_synapse_clustering_part3.ipynb">neuron_synapse_clustering_part3.ipynb</a></td>
      <td>Prediction of clusters for newly observed synapses using trained models.</td>
    </tr>
  </tbody>
</table>


<h2>Source Code (src)</h2>

<table>
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
      <td>PCA-based spatial alignment separately for each neuron.</td>
    </tr>
    <tr>
      <td></td>
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
      <td></td>
      <td>plot_umap_3d</td>
      <td>Visualizes 3D UMAP embeddings.</td>
    </tr>
    <tr>
      <td></td>
      <td>plot_hdbscan_umap_clusters</td>
      <td>Visualizes HDBSCAN clustering results (2D or 3D).</td>
    </tr>
    <tr>
      <td><a href="src/train.py">train.py</a></td>
      <td>random_forest_with_outlier_filtering</td>
      <td>Random Forest classifier with OneClassSVM-based outlier filtering.</td>
    </tr>
    <tr>
      <td></td>
      <td>xgboost_with_outlier_filtering</td>
      <td>XGBoost classifier with OneClassSVM and label encoding.</td>
    </tr>
    <tr>
      <td><a href="src/prediction.py">prediction.py</a></td>
      <td>predict_clusters</td>
      <td>Predicts cluster labels for newly observed synapses.</td>
    </tr>
  </tbody>
</table>


<h2>Models</h2>

<table>
  <thead>
    <tr>
      <th>File name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="pca_models_per_neuronid.pkl">pca_models_per_neuronid.pkl</a></td>
      <td>PCA models for neuron-specific spatial alignment.</td>
    </tr>
    <tr>
      <td><a href="robust_scaler.pkl">robust_scaler.pkl</a></td>
      <td>RobustScaler fitted on historical dataset.</td>
    </tr>
    <tr>
      <td><a href="as_xgb_bundle.pkl">as_xgb_bundle.pkl</a></td>
      <td>XGBoost model for excitatory synapses.</td>
    </tr>
    <tr>
      <td><a href="cr_knn_bundle.pkl">cr_knn_bundle.pkl</a></td>
      <td>KNN model for calretinin synapses.</td>
    </tr>
    <tr>
      <td><a href="cr_rf_bundle.pkl">cr_rf_bundle.pkl</a></td>
      <td>Random Forest model for calretinin synapses.</td>
    </tr>
    <tr>
      <td><a href="pv_xgb_bundle.pkl">pv_xgb_bundle.pkl</a></td>
      <td>XGBoost model for parvalbumin synapses.</td>
    </tr>
  </tbody>
</table>


