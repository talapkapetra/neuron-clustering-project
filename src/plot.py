import plotly as py
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Color palette for NeuronId
neuron_order = ['cb1', 'cb2', 'cb3', 'cr1', 'cr2', 'cr3', 'pv1', 'pv2', 'pv3']
base_palette = sns.color_palette('tab10', len(neuron_order))
NEURON_COLORS = dict(zip(neuron_order, base_palette))

# 3D UMAP plot
def plot_umap_3d(df_umap, neuron_colors=NEURON_COLORS, title=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter by NeuronId
    for nid in df_umap['NeuronId'].unique():
        sub = df_umap[df_umap['NeuronId'] == nid]
        color = neuron_colors.get(nid, 'gray')
        ax.scatter(
            sub['UMAP_1'], sub['UMAP_2'], sub['UMAP_3'],
            s=30, alpha=0.7, color=color, label=nid
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='NeuronId')

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

# 2D UMAP plot
def plot_umap_2d(df_umap, neuron_colors=NEURON_COLORS, title=None):
    plt.figure(figsize=(8, 6))

    for nid in df_umap['NeuronId'].unique():
        sub = df_umap[df_umap['NeuronId'] == nid]
        color = neuron_colors.get(nid, 'gray')
        plt.scatter(
            sub['UMAP_1'], sub['UMAP_2'],
            s=30, alpha=0.7, color=color, label=nid
        )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='NeuronId')
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- cím használata 2D UMAP-nél ---
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

# HDBSCAN plot
def plot_hdbscan_umap_clusters(df_cluster, figsize=(10, 8), point_size=30, title=None):

        if 'Cluster' not in df_cluster.columns:
            raise ValueError("Missing 'Cluster' column")

        labels = df_cluster['Cluster'].values
        is_3d = 'UMAP_3' in df_cluster.columns
        X = df_cluster[['UMAP_1', 'UMAP_2', 'UMAP_3']] if is_3d else df_cluster[['UMAP_1', 'UMAP_2']]
        X = X.values

        unique_labels = np.unique(labels)
        cluster_labels = unique_labels[unique_labels != -1]
        cmap = plt.cm.get_cmap('tab10', max(len(cluster_labels), 1))
        label_to_color_idx = {lab: i for i, lab in enumerate(cluster_labels)}

        # -------- 2D eset --------
        if not is_3d:
            plt.figure(figsize=figsize)
            for k in unique_labels:
                mask = labels == k
                xy = X[mask]
                if k == -1:
                    plt.scatter(xy[:,0], xy[:,1], c='black', marker='x', s=point_size, label="Noise (-1)")
                else:
                    col = cmap(label_to_color_idx[k])
                    plt.scatter(xy[:,0], xy[:,1], c=[col], s=point_size, label=f"Cluster {k}")
      
            plt.xlabel("UMAP Component 1")
            plt.ylabel("UMAP Component 2")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, linestyle='--', alpha=0.5)

            # --- cím használata 2D HDBSCAN-nél ---
            if title is not None:
                plt.title(title)

            plt.tight_layout()
            plt.show()
            return

        # -------- 3D eset --------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        for k in unique_labels:
            mask = labels == k
            xyz = X[mask]
            if k == -1:
                ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c='black', marker='x', s=point_size, label="Noise (-1)")
            else:
                col = cmap(label_to_color_idx[k])
                ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=[col], s=point_size, label=f"Cluster {k}")
      
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        ax.set_zlabel("UMAP Component 3")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # --- cím használata 3D HDBSCAN-nél ---
        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()
