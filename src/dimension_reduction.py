import numpy as np
from sklearn.decomposition import PCA
import umap

coord_cols = ['zero_corr_x', 'zero_corr_y', 'zero_corr_z']

# 1. DEFINITION
# Transform the 3D coordinates (x, y, z) of a single neuron into a new coordinate system using PCA.
def pca_align_one_neuron(group):
    X_orig = group[coord_cols].values
    X_centered = X_orig - X_orig.mean(axis=0, keepdims=True)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_centered)

# If the PCA axis points in the opposite direction to the original axis, the sign is flipped to match the original orientation.
    for comp in range(3):
        corr = np.corrcoef(X_pca[:, comp], X_orig[:, comp])[0, 1]
        if corr < 0:
            X_pca[:, comp] *= -1.0

# Loading the new PCA coordinates into the database as new columns.
    group['pca_x'] = X_pca[:, 0]
    group['pca_y'] = X_pca[:, 1]
    group['pca_z'] = X_pca[:, 2]
    return group, pca

# 2. DEFINITION
# run UMAP in 2d or 3d

UMAP_BASE_COLS = ['pca_x', 'pca_y', 'pca_z', 'SynArea', 'BoutonArea']

# designate parameters
def make_umap(
    df_source,
    syn_type=None,          # 'as' / 'ss' / None
    neuron_type=None,       # 'cb', 'cr', 'pv',... / None
    neuron_ids=None,        # ['cb1','cb2'] or 'cb1' / None
    coord_weight=1.0,       # 1.0 = no extra weight, 0.1 = weaker 
    morph_weight=1.0,       # 1.0 = no extra weight
    n_components=3,         # 2 or 3
    n_neighbors=20,
    min_dist=0.1,
    random_state=42,
    feature_cols=None,      # if None -> UMAP_BASE_COLS
    coord_cols=('pca_x', 'pca_y', 'pca_z'),
    morph_cols=('SynArea', 'BoutonArea'),
    **umap_kwargs          # add extra UMAP parameters
):
    """
    General UMAP definition.

    Parameters:
    - df_source: Input DataFrame (e.g., neuron_df_pca)
    - syn_type: Filter by 'as', 'ss', or None
    - neuron_type: Filter by neuron type (e.g., 'cb', 'cr', 'pv') or None
    - neuron_ids: List or string of NeuronIds to include, or None
    - coord_weight, morph_weight: Weights for coordinate and morphometric features
    - n_components: Number of UMAP dimensions (2 or 3)
    - feature_cols: Columns used as UMAP input (default: UMAP_BASE_COLS)
    - coord_cols, morph_cols: Feature column names for coordinates and morphometrics
    - **umap_kwargs: Additional keyword arguments passed to umap.UMAP()
    """

    # 1) Initial subset (always a copy)
    df_sub = df_source.copy()

    # 2) Filter SynType
    if syn_type is not None:
        df_sub = df_sub[df_sub['SynType'] == syn_type]

    # 3) Filter NeuronType
    if neuron_type is not None:
        df_sub = df_sub[df_sub['NeuronType'] == neuron_type]

    # 4) Filter NeuronId
    if neuron_ids is not None:
        if isinstance(neuron_ids, str):
            neuron_ids = [neuron_ids]
        df_sub = df_sub[df_sub['NeuronId'].isin(neuron_ids)]

    # 5) If empty, send a bug message
    if df_sub.shape[0] == 0:
        raise ValueError(
            "make_umap after filtering received an empty df_sub – "
            "check the syn_type / neuron_type / neuron_ids parameters!"
        )

    # 6) Which columns are used by UMAP?
    if feature_cols is None:
        feature_cols = UMAP_BASE_COLS

    df_scaled = df_sub[feature_cols].copy()

    # 6/a) coordinate weight
    for c in coord_cols:
        if c in df_scaled.columns:
            df_scaled[c] *= coord_weight

    # 6/b) morphology weight
    for c in morph_cols:
        if c in df_scaled.columns:
            df_scaled[c] *= morph_weight

    X_umap = df_scaled.values

    # 7) UMAP calculation
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='euclidean',
        n_jobs=-1,
        **umap_kwargs,
    )

    print(
        f"Running UMAP (SynType={syn_type}, NeuronType={neuron_type}, "
        f"coord_w={coord_weight}, morph_w={morph_weight}, "
        f"n_comp={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist})"
    )

    emb = reducer.fit_transform(X_umap)

    # 8) New DataFrame with UMAP-coordinates – ORIGINAL COLUMNS + UMAP
    df_umap = df_sub.copy()
    df_umap['UMAP_1'] = emb[:, 0]
    if n_components > 1:
        df_umap['UMAP_2'] = emb[:, 1]
    if n_components > 2:
        df_umap['UMAP_3'] = emb[:, 2]
        
    return df_umap, reducer

