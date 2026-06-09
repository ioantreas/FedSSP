from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree

root = "./Data1/TUDataset"

datasets = [
    "MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",
    "Peking_1", "OHSU", "PROTEINS",
    "IMDB-MULTI", "IMDB-BINARY",
    "Letter-high", "Letter-med", "Letter-low",
    "SYNTHETICnew", "SYNTHIE", "SYNTHETIC", 
    "DD", "REDDIT-BINARY", "FIRSTMM_DB"
]

for name in datasets:
    if name == "IMDB-BINARY":
        ds = TUDataset(root, name, pre_transform=OneHotDegree(135, cat=False))
    elif name == "IMDB-MULTI":
        ds = TUDataset(root, name, pre_transform=OneHotDegree(88, cat=False))
    elif "Letter" in name:
        ds = TUDataset(root, name, use_node_attr=True)
    else:
        ds = TUDataset(root, name)
    print(name, len(ds))