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
def filter_function(n):
    return n in louvain_partitions[2]
# %%
sub_G = nx.subgraph_view(G, filter_node=filter_function)
# %%
louvain_communities = nx_comm.louvain_communities(sub_G, resolution=0.9,threshold=1e-7)
display(len(louvain_communities))
# %%
for idx in range(len(louvain_communities)):
    print("Community number {i} has {number} members".format(i=idx,number=len(louvain_communities[idx])))
# %%
display(louvain_communities[0])
# CNN, Vox, MSNBC, The Young Turks --> Sérieux Très à gauche
# %%
display(louvain_communities[1])
# Philip de Franco, Drama Alert, IntMensOrg -->Pas très sérieux Plutôt de droite mais doit y avoir des erreurs
# %%
display(louvain_communities[2])
# Truly, ABC News, BBC News, True Crime Daily, Business Insider, NYT --> Sérieux Gauche / Leaning left / centre
# %%
display(louvain_communities[3])
# Pat Condell (Conspiration ?), Sky News Australia, National Post  --> Canadien, Australien, British, Conspirationist
# %%
display(louvain_communities[4])
# Inside Edition, Fox News, Daily Wire, Rebel News(Très à droite mais canadien) --> Très à droite 
# %%
display(louvain_communities[5])
# Today, China uncensored, Fox Business, The dedicated Citizen --> compliqué à classer (business et Chine ?)
# %%
display(louvain_communities[6])
# Al jazeera(lean left), African Diaspora news channel, France 24 English, Visual Politik EN
# %%
