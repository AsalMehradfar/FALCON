import matplotlib.pyplot as plt
import networkx as nx
import umap
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib as mpl


def compute_layout(netlist_graph):
    """Computes the layout for node positioning."""
    return nx.spring_layout(netlist_graph, seed=42, iterations=100)


def draw_nodes_and_labels(netlist_graph, pos):
    """Draws nodes and their labels."""
    nx.draw_networkx_nodes(
        netlist_graph, 
        pos, 
        node_size=500,  
        node_color='black',
        # alpha=0.7,
        # node_color='skyblue',  
        edgecolors='black',  
        linewidths=1.5
    )

    nx.draw_networkx_labels(
        netlist_graph, 
        pos, 
        font_size=8,  
        font_color='white',
        verticalalignment='center_baseline',
        # font_color='black',
        font_weight='bold'
    )


def process_edges(netlist_graph, pos, edge_colors_dict):
    """Processes edges, assigns curvatures, and returns edge labels & positions."""
    edge_index = {}  # Tracks how many edges exist between each node pair
    edge_labels = {}  # Dictionary to store edge labels
    curvatures = []  # Store curvature values for correct label placement
    edge_colors = []  # Store edge colors for each edge
    colors_used = []

    for i, (u, v, key) in enumerate(netlist_graph.edges(keys=True)):
        edge_data = netlist_graph[u][v][key]
        component = edge_data['component']
        name = edge_data['name']

        # Generate edge labels
        label = f"{name}_{component.split('_')[-1]}" if component.startswith(('nmos', 'pmos', 'balun')) else name
        edge_labels[(u, v, key)] = label

        # Get number of edges between (u, v)
        num_edges = netlist_graph.number_of_edges(u, v)

        # Calculate curvature for multiple edges
        curvature = (edge_index.get((u, v), 0) - (num_edges - 1) / 2) * 0.35 if num_edges > 1 else 0
        edge_index[(u, v)] = edge_index.get((u, v), 0) + 1
        curvatures.append(curvature)

        # Compute edge midpoint for label placement
        mid_x, mid_y = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2

        # Shift labels for multi-edges
        if num_edges > 1:
            shift_amount = 0.10  # Space between multi-edge labels
            shift = (key - (num_edges - 1) / 2) * shift_amount
            mid_y += shift  # Shift labels vertically

        # Store edge colors
        edge_colors.append(edge_colors_dict[component.split('_')[0]])
        colors_used.append(component.split('_')[0])

        # Draw the edge
        nx.draw_networkx_edges(
            netlist_graph, 
            pos, 
            edgelist=[(u, v)], 
            edge_color=[edge_colors[-1]],
            width=2,  
            connectionstyle=f'arc3,rad={curvature}'  
        )

    return edge_labels, curvatures, colors_used


def draw_edge_labels(netlist_graph, pos, edge_labels, curvatures):
    """Draws edge labels with appropriate spacing and alignment."""
    [
        nx.draw_networkx_edge_labels(
            netlist_graph, 
            pos, 
            edge_labels={e: v}, 
            font_size=7, 
            font_color="black", 
            font_weight="bold",
            connectionstyle=f'arc3,rad={curvatures[i % len(curvatures)]}'
        ) 
        for i, (e, v) in enumerate(edge_labels.items())
    ]


def add_legend(edge_colors_dict, colors_used):
    """Adds a legend for edge colors."""
    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=component)
        for component, color in edge_colors_dict.items() if component in colors_used
    ]
    plt.legend(
        handles=legend_handles, 
        # loc="best",  # Automatically find the best position
        fontsize=9, 
        title="Components", 
        title_fontsize=10,
        frameon=False,
        loc="upper left",
    )


def plot_netlist_graph(netlist_graph, circuit, edge_colors_dict, save_path=None):
    """
    Plots a publication-quality netlist graph with properly spaced edge labels using 
    networkx's draw_networkx_edge_labels function.
    """
    plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    # "text.usetex": True,
    # "font.size": 6  # adjust as needed
    })
    plt.figure(figsize=(7, 7))

    pos = compute_layout(netlist_graph)
    draw_nodes_and_labels(netlist_graph, pos)

    edge_labels, curvatures, colors_used = process_edges(netlist_graph, pos, edge_colors_dict)
    draw_edge_labels(netlist_graph, pos, edge_labels, curvatures)
    
    add_legend(edge_colors_dict, colors_used)

    # Add a title and save the figure
    # plt.title(f'Graph Representation of {circuit} Netlist', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=400)
    plt.show()


def plot_tsne(dataset, title="t-SNE of Performance Vectors", save_path=None):
    print("[INFO] Preparing data for t-SNE...")

    # Extract performance vectors and labels
    X = []
    y = []

    n = len(dataset)
    tmp = list(range(n))
    
    random.seed(42)
    random.shuffle(tmp)
    
    for i in tqdm(tmp):
        sample = dataset.__getitem__(i)
        X.append(sample[0])
        y.append(sample[1])

    X = np.array(X)
    y = np.array(y)
    num_classes = len(np.unique(y))

    print("[INFO] Running t-SNE...")
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = reducer.fit_transform(X)

    print("[INFO] Plotting...")
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(-0.5, num_classes+0.5, 1), ncolors=num_classes)
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap='tab20', norm=norm, alpha=0.7)
    # plt.title(title)
    ax.set_xlabel("t-SNE-1", labelpad=5, fontsize=14)
    ax.set_ylabel("t-SNE-2", labelpad=-10, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12) 
    
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04, ticks=np.arange(num_classes))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Circuit Topology Code", labelpad=10, fontsize=12)
    cbar.ax.minorticks_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='png', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved t-SNE plot to {save_path}")
    plt.show()


def plot_umap(dataset, title="UMAP of Performance Vectors", save_path=None):
    print("[INFO] Preparing data for UMAP...")

    # Extract performance vectors and labels
    X = []
    y = []

    n = len(dataset)
    tmp = list(range(len(dataset)))
    
    random.seed(42)
    random.shuffle(tmp)
    for i in tqdm(tmp[: 50000]):
        sample = dataset.__getitem__(i)
        X.append(sample[0])
        y.append(sample[1])

    X = np.array(X)
    y = np.array(y)
    num_classes = len(np.unique(y))

    print("[INFO] Running UMAP...")
    reducer = umap.UMAP(n_components=2)
    X_embedded = reducer.fit_transform(X)

    print("[INFO] Plotting...")
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mpl.colors.BoundaryNorm(boundaries=np.arange(-0.5, num_classes+0.5, 1), ncolors=num_classes)
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], s=5, c=y, cmap='tab20', norm=norm, alpha=0.7)
    # plt.title(title)
    ax.set_xlabel("UMAP-1", labelpad=5, fontsize=14)
    ax.set_ylabel("UMAP-2", labelpad=-10, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12) 
    
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04, ticks=np.arange(num_classes))
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Circuit Topology Code", labelpad=10, fontsize=12)
    cbar.ax.minorticks_off()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved UMAP plot to {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = np.round(cm_normalized * 100, 1)  # in percentages
        fmt = ".1f"
        cmap = "Purples"
    else:
        cm_display = cm
        fmt = "d"
        cmap = "Blues"

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title if not normalize else f"{title} (Normalized %)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Confusion Matrix plot to {save_path}")
    plt.show()


def plot_confused_classes_only(y_true, y_pred, class_names, title="Confused Classes Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Normalize to %

    # # 🧠 Find confused classes:
    # off_diagonal = cm.copy()
    # np.fill_diagonal(off_diagonal, 0)

    # Ignore diagonal by masking
    off_diag_mask = ~np.eye(len(cm_normalized), dtype=bool)
    off_diagonal = cm_normalized * off_diag_mask

    # Find indices where either the row or column has error
    confused_rows = (off_diagonal.sum(axis=1) > 0.01)
    confused_cols = (off_diagonal.sum(axis=0) > 0.01)
    confused_indices = np.where(confused_rows | confused_cols)[0]

    # 🔥 Submatrix of confused classes only
    cm_confused = cm_normalized[np.ix_(confused_indices, confused_indices)]
    class_names_confused = [class_names[i] for i in confused_indices]

    # 🎨 Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_confused, annot=True, fmt=".2f", cmap="BuPu",
                xticklabels=class_names_confused, yticklabels=class_names_confused, cbar=True, linecolor='lightgray', linewidths=0.5, cbar_kws={'shrink': 0.9}, annot_kws={"size": 9})
    
    ax = plt.gca()
    cbar = ax.collections[0].colorbar
    cbar.set_label("Percentage (%)", fontsize=11)

    plt.xlabel("Predicted Topology", fontsize=12)
    plt.ylabel("True Topology", fontsize=12)
    # plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Confused Matrix plot to {save_path}")

    plt.show()


def plot_per_class_accuracy(accuracies, class_names=None, title="Per-Class Accuracy", save_path=None):
    num_classes = len(accuracies)
    x = np.arange(num_classes)
    accs = [acc * 100 for acc in accuracies]

    plt.figure(figsize=(12, 3))
    bars = plt.bar(x, accs, color="#4C72B0", edgecolor='black', alpha=0.8)

    plt.xticks(x, class_names if class_names else x, rotation=45, ha='right')
    # plt.xlabel("Circuit Topology", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlim(-0.6, len(class_names) - 0.4)
    plt.ylim(0, 115)
    # plt.title(title, pad=20)
    plt.grid(axis="y", linestyle="--", linewidth=0.5,alpha=0.5, zorder=0)
    for bar in bars:
        bar.set_zorder(3)

    # Annotate bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f"{accuracies[i]*100:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Per Class Accuracy plot to {save_path}")
    plt.show()


def plot_loss_curves(train_losses, val_losses, title="Training Loss", save_path=None, log_scale=False):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (list or array): Training loss values per epoch.
        val_losses (list or array): Validation loss values per epoch.
        title (str): Title of the plot.
        save_path (str or None): If provided, saves the plot to this file.
        log_scale (bool): Whether to use log scale on the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Loss plot to {save_path}")
    plt.show()


def plot_relative_error_distribution(errors, title="Mean Relative Error Distribution", save_path=None):
    """
    Plots the distribution of relative errors (as percentages) with a paper-friendly style.

    Args:
        errors (list or array): Relative errors (assumed as fractions, not percentages).
        title (str): Optional plot title.
        save_path (str): If given, saves the figure as a .pdf (recommended for papers).
    """

    trim_percentiles=(0, 95)
    errors = np.array(errors)
    lower, upper = np.percentile(errors, trim_percentiles)
    errors = errors[(errors >= lower) & (errors <= upper)]

    # Convert to percentage
    errors = [e * 100 for e in errors if e is not None]

    # Plot setup
    plt.figure(figsize=(8, 6))
    sns.kdeplot(errors, fill=True, color='navy', linewidth=1.2, clip=(0, None), alpha=0.3)

    # Labels and styling
    plt.xlabel("Relative Error (%)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Optional title
    if title:
        plt.title(title, fontsize=14)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()


def plot_relative_error_distribution_with_stats(errors, title="Mean Relative Error Distribution", save_path=None):
    """
    Enhanced plot for relative error distribution with paper-style formatting and statistics.
    """

    # Convert to numpy and clip extreme values
    errors = np.array(errors)
    trim_percentiles = (0, 95)
    lower, upper = np.percentile(errors, trim_percentiles)
    errors = errors[(errors >= lower) & (errors <= upper)]

    # Convert to percentage
    errors = np.array([e * 100 for e in errors if e is not None])

    # Compute stats
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    iqr_low, iqr_high = np.percentile(errors, [25, 75])

    # Plot setup
    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Step 1: Hidden KDE line to extract peak
    line_obj = sns.kdeplot(
        errors,
        fill=False,
        bw_adjust=3,
        clip=(0, None) ,
        color='none'  # don't show
    )

    # Extract curve data from hidden line
    kde_x, kde_y = line_obj.get_lines()[0].get_data()
    peak_idx = np.argmax(kde_y)
    peak_x = kde_x[peak_idx]
    peak_y = kde_y[peak_idx]

    # Step 2: Plot filled KDE now
    sns.kdeplot(
        errors,
        fill=True,
        color='navy',
        linewidth=1.6,
        clip=(0, None),
        alpha=0.3,
        bw_adjust=3,
        label=None
    )

    # Mean, Median, and Mode lines
    plt.axvline(mean_err, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_err:.2f}%')
    plt.axvline(median_err, color='green', linestyle='--', linewidth=1.5, label=f'Median: {median_err:.2f}%')
    plt.axvline(peak_x, color='darkorange', linestyle='--', linewidth=1.5, label=f'Mode: {peak_x:.2f}%')

    # Axis & style
    plt.xlabel("Relative Error (%)", fontsize=15)
    plt.ylabel("Density", fontsize=15, labelpad=10)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(loc='upper right', fontsize=15, frameon=True)
    # plt.xticks(list(plt.xticks()[0]) + [mean_err, median_err, peak_x], fontsize=12)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()


def plot_loss_backward(losses, title="Backward Loss", save_path=None, log_scale=False):
    """
    Plots training and validation loss curves.

    Args:
        train_losses (list or array): Training loss values per epoch.
        val_losses (list or array): Validation loss values per epoch.
        title (str): Title of the plot.
        save_path (str or None): If provided, saves the plot to this file.
        log_scale (bool): Whether to use log scale on the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=400, bbox_inches='tight')
        print(f"[INFO] Saved Loss plot to {save_path}")
    plt.show()