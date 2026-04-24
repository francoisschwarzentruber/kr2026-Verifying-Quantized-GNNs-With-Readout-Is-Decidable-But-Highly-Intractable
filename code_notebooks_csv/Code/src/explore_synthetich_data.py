import pandas as pd
from torch_geometric.data import DataLoader
from utils.util import load_data

# === Path to the data ===
file_path = "."
data_path = "data"
data_dir = "datasets"
keys = ["p1", "p2", "p3"]

records = []

for dataset_key in keys:
    # Paths
    train_path = f"{data_dir}/{dataset_key}/train-random-erdos-5000-40-50"
    test1_path = f"{data_dir}/{dataset_key}/test-random-erdos-500-40-50"
    test2_path = f"{data_dir}/{dataset_key}/test-random-erdos-500-51-60"

    # Load datasets
    train_graphs, (a, b, tr_num_classes) = load_data(f"{file_path}/{data_path}/{train_path}.txt")
    test1_graphs, (at1, bt1, t1_num_classes) = load_data(f"{file_path}/{data_path}/{test1_path}.txt")
    test2_graphs, (at2, bt2, t2_num_classes) = load_data(f"{file_path}/{data_path}/{test2_path}.txt")

    def summarize(graphs, dataset_key, set_name, graph_labels, node_features, node_labels):
        return {
            "Dataset": dataset_key,
            "Set": set_name,
            "Graph Labels": graph_labels,
            "Node Features": node_features,
            "Node Labels": node_labels,
            "Num Graphs": len(graphs),
            "Min Nodes": min(g.num_nodes for g in graphs),
            "Max Nodes": max(g.num_nodes for g in graphs),
            "Avg Nodes": round(sum(g.num_nodes for g in graphs) / len(graphs), 2),
            "Min Edges": min(g.num_edges for g in graphs),
            "Max Edges": max(g.num_edges for g in graphs),
            "Avg Edges": round(sum(g.num_edges for g in graphs) / len(graphs), 2),
        }

    records.append(summarize(train_graphs, dataset_key, "Train", a, b, tr_num_classes))
    records.append(summarize(test1_graphs, dataset_key, "Test1", at1, bt1, t1_num_classes))
    records.append(summarize(test2_graphs, dataset_key, "Test2", at2, bt2, t2_num_classes))

# === CREATE FINAL TABLE ===
df = pd.DataFrame(records)
#df.to_csv("dataset_statistics_summary.csv", index=False)

df_toprint = df.drop(columns=['Graph Labels', 'Node Features', 'Node Labels', 'Num Graphs'])
print(df)
latex_table = df_toprint.to_latex(index=False, float_format="%.2f", caption="Dataset Statistics Summary", label="tab:dataset_summary")
print(latex_table)
