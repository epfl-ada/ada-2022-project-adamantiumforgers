# %%
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
# %%
df_edges = pd.read_csv("data/graph.csv", sep=';')
display(df_edges)

# %%
G = nx.from_pandas_edgelist(df_edges, edge_attr=True)
display(G)
# %%
nx.draw_networkx(G)
# %%
louvain_partitions = nx_comm.louvain_communities(G, resolution=1e-1, threshold=1e-100,seed=1)

# %%
for idx in range(len(louvain_partitions)):
    print("Community number {i} has {number} members".format(i=idx,number=len(louvain_partitions[idx])))
# %%
display(df_edges)
# %%
print(louvain_partitions[2])

filtered_channels = pd.DataFrame(louvain_partitions[2])
# %%
display(filtered_channels)
# %%
filtered_channels = filtered_channels[0].sort_values(ascending=True)

# %%
display(filtered_channels)
# %%
filtered_channels.to_csv("data/louvain_filtered_channels.csv", sep=';', index=False)
# %%
