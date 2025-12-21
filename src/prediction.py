import numpy as np
import joblib

# Column configs
COORD_COLS_ZERO = ['zero_corr_x', 'zero_corr_y', 'zero_corr_z']

def apply_pca(group, pca_models):
    neuron_id = group['NeuronId'].iloc[0]

    if neuron_id not in pca_models:
        raise ValueError(f"PCA model not found for NeuronId: {neuron_id}")

    pca = pca_models[neuron_id]

    X_orig = group[COORD_COLS_ZERO].values
    X_centered = X_orig - X_orig.mean(axis=0, keepdims=True)

    X_pca = pca.transform(X_centered)

    group = group.copy()
    group['pca_x'] = X_pca[:, 0]
    group['pca_y'] = X_pca[:, 1]
    group['pca_z'] = X_pca[:, 2]

    return group

def apply_pca_robust_scaler(newsyn_df_final):
    import joblib

    pca_models = joblib.load("pca_models_per_neuronid.pkl")
    scaler = joblib.load("robust_scaler.pkl")

    FEATURE_COLS = ['pca_x', 'pca_y', 'pca_z', 'SynArea', 'BoutonArea']

    # PCA
    df_pca = newsyn_df_final.groupby("NeuronId", group_keys=False).apply(apply_pca, pca_models=pca_models)

    # Scaling
    X = df_pca[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    df_scaled = df_pca.copy()
    for i, col in enumerate(FEATURE_COLS):
        df_scaled[col + '_scaled'] = X_scaled[:, i]

    return df_scaled

def umap_model_syntype(df_scaled):
    import joblib

    FEATURE_COLS = ['pca_x', 'pca_y', 'pca_z', 'SynArea', 'BoutonArea']
    umap_models = {
        'as': joblib.load("umap_as_model.pkl"),
        'ss': joblib.load("umap_ss_model.pkl"),
    }

    df_result = df_scaled.copy()

    for syn_type in ['as', 'ss']:
        mask = df_result['SynType'] == syn_type
        if not mask.any():
            continue

        model = umap_models[syn_type]
        X = df_result.loc[mask, [col + '_scaled' for col in FEATURE_COLS]].values
        emb = model.transform(X)

        df_result.loc[mask, 'UMAP_1'] = emb[:, 0]
        df_result.loc[mask, 'UMAP_2'] = emb[:, 1]

    return df_result

def umap_model_neurontype(df_scaled):
    import joblib

    FEATURE_COLS = ['pca_x', 'pca_y', 'pca_z', 'SynArea', 'BoutonArea']
    umap_models = {
        'cb': joblib.load("umap_cb_model.pkl"),
        'cr': joblib.load("umap_cr_model.pkl"),
        'pv': joblib.load("umap_pv_model.pkl"),
    }

    neuron_type_map = {
        'calbindin': 'cb',
        'calretinin': 'cr',
        'parvalbumin': 'pv',
    }

    df_result = df_scaled.copy()

    for full_name, short_name in neuron_type_map.items():
        mask = df_result['NeuronType'] == full_name
        if not mask.any():
            continue

        model = umap_models[short_name]
        X = df_result.loc[mask, [col + '_scaled' for col in FEATURE_COLS]].values
        emb = model.transform(X)

        df_result.loc[mask, 'UMAP_1'] = emb[:, 0]
        df_result.loc[mask, 'UMAP_2'] = emb[:, 1]

    return df_result

def predict_clusters(df, bundle_path):
    """
    Predict clusters using a saved model bundle.
    Handles:
    - scaling (if present)
    - OneClassSVM outlier detection
    - label decoding (XGBoost)
    """

    bundle = joblib.load(bundle_path)

    model = bundle["model"]
    ocsvm = bundle.get("ocsvm", None)
    label_encoder = bundle.get("label_encoder", None)
    scaler = bundle.get("scaler", None)

    X = df[['UMAP_1', 'UMAP_2']].values

    # scale ONLY if scaler exists
    if scaler is not None:
        X = scaler.transform(X)

    # outlier detection
    if ocsvm is not None:
        inlier_mask = ocsvm.predict(X) == 1
    else:
        inlier_mask = np.ones(len(X), dtype=bool)

    clusters = np.full(len(X), -1, dtype=int)

    if inlier_mask.any():
        X_inliers = X[inlier_mask]
        y_pred = model.predict(X_inliers)

        # decode labels if needed (XGB)
        if label_encoder is not None:
            y_pred = label_encoder.inverse_transform(y_pred)

        clusters[inlier_mask] = y_pred.astype(int)

    df_output = df.copy()
    df_output["Cluster"] = clusters

    return df_output
