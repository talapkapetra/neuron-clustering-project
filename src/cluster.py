import numpy as np
import hdbscan

def run_hdbscan_umap(
    umap_df_source,
    min_cluster_size=15,
    min_samples=3,
    syn_type=None,      # optional filter
    neuron_type=None,   # optional filter
    neuron_ids=None,    # optional filter
):
    """
    Run HDBSCAN in UMAP space, automatically detecting whether the data is 2D (UMAP_1, UMAP_2) or 3D (UMAP_1, UMAP_2, UMAP_3).

    Parameters:
    - df_cluster: DataFrame with filtered data and HDBSCAN labels ('HDB_label', 'Cluster')
    - clusterer : The fitted HDBSCAN object
    """

    df_sub = umap_df_source.copy()

    # Optional filters
    if syn_type is not None and 'SynType' in df_sub.columns:
        df_sub = df_sub[df_sub['SynType'] == syn_type]

    if neuron_type is not None and 'NeuronType' in df_sub.columns:
        df_sub = df_sub[df_sub['NeuronType'] == neuron_type]

    if neuron_ids is not None and 'NeuronId' in df_sub.columns:
        if isinstance(neuron_ids, str):
            neuron_ids = [neuron_ids]
        df_sub = df_sub[df_sub['NeuronId'].isin(neuron_ids)]

    if df_sub.shape[0] == 0:
        raise ValueError("run_hdbscan_umap: df_sub is empty after filter – "
                         "check SynType / NeuronType / NeuronId!")

    # automatic recongition of dimension
    has_umap3 = 'UMAP_3' in df_sub.columns
    if has_umap3:
        cols = ['UMAP_1', 'UMAP_2', 'UMAP_3']
        n_components = 3
    else:
        cols = ['UMAP_1', 'UMAP_2']
        n_components = 2

    X_hdb = df_sub[cols].values

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )

    labels = clusterer.fit_predict(X_hdb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"HDBSCAN: dim={n_components}D, clusters={n_clusters}, noise={n_noise}, "
          f"min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    cluster_num = np.where(labels == -1, -1, labels + 1).astype(int)

    df_cluster = df_sub.copy()
    df_cluster['HDB_label'] = labels
    df_cluster['Cluster']   = cluster_num

    return df_cluster, clusterer
