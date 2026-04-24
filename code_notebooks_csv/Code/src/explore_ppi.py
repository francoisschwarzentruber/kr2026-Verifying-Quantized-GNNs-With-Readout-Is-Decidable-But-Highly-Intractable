import torch
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI

def summarize(graphs, set_name):
    all_labels = torch.cat([g.y for g in graphs], dim=0)
    return {
        "Dataset": set_name,
        "Num Graphs": len(graphs),
        "Node Feature Dim": graphs[0].num_features,
        "Label Dim": graphs[0].y.shape[1],
        "Avg Active Labels/Node": round(all_labels.sum(dim=1).float().mean().item(), 2),
        
        "Min Nodes": min(g.num_nodes for g in graphs),
        "Max Nodes": max(g.num_nodes for g in graphs),
        "Avg Nodes": round(sum(g.num_nodes for g in graphs) / len(graphs), 2),
        
        "Min Edges": min(g.num_edges for g in graphs),
        "Max Edges": max(g.num_edges for g in graphs),
        "Avg Edges": round(sum(g.num_edges for g in graphs) / len(graphs), 2),

        "Avg Degree": round(2 * sum(g.num_edges for g in graphs) / sum(g.num_nodes for g in graphs), 2),
    }

# Load PPI dataset
train_dataset = PPI(root='Supplement_materials\Code\src\data\ppi', split='train')
val_dataset = PPI(root='Supplement_materials\Code\src\data\ppi', split='val')
test_dataset = PPI(root='Supplement_materials\Code\src\data\ppi', split='test')



records = []
records.append(summarize(train_dataset, "Train"))
records.append(summarize(val_dataset, "Validation"))
records.append(summarize(test_dataset, "Test"))
df = pd.DataFrame(records)
print(df)
df1 = df[["Dataset", "Num Graphs", "Node Feature Dim", "Label Dim", "Avg Active Labels/Node", "Avg Degree"]]
latex_table = df[["Dataset","Min Nodes", "Max Nodes", "Avg Nodes","Min Edges", "Max Edges", "Avg Edges"]].to_latex(index=False, float_format="%.2f", caption="Dataset Statistics Summary", label="tab:dataset_summary")
print("Dataset Summary")

print(df1.to_latex(index=False, float_format="%.2f", caption="Dataset Summary", label="tab:dataset_summary"))
print("Dataset Statistics Summary")
print(latex_table)

# Save the DataFrame to a CSV file
file_path = "Supplement_materials/Code/src/for_analysis/results_ppi"
df1.to_csv(f"{file_path}/DatasetSummaryppibenchmark.csv", index=False)
df[["Dataset","Min Nodes", "Max Nodes", "Avg Nodes","Min Edges", "Max Edges", "Avg Edges"]].to_csv(f"{file_path}/DatasetStatisticSummaryppibenchmark.csv", index=False)

