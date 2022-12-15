import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc("axes.spines", top = False, right = False)
plt.rc("axes", titlelocation = "left")
plt.rc('font', family = 'Helvetica')
plt.style.use('seaborn-deep')

data = pd.read_csv("outcomes_covars.csv")\
    .drop(columns = ["Unnamed: 0"])\
    .assign(leader = lambda _: (_.leader > 0).astype(int))

bins = list(range(0, 91, 2))
plt.hist(
    [data[data.leader == 0].degree, data[data.leader == 1].degree],
    bins,
    label = ["non-leaders", "leaders"],
    density = True
)
plt.legend(loc = "upper right", framealpha = 0, handlelength = 0.5)
plt.gca().set(xlabel = "node degree", ylabel = "density")
plt.show()

plt.gca().set(xlabel = "degree", ylabel = "leaders", title = "degree distributions for leaders and non-leaders")


# ideal split, no network
g_ideal = nx.Graph()
g_ideal.add_nodes_from(range(1, 13))
edge_set = [(1, 2), (2, 3), (1, 4), (4, 5), (2, 5), (3, 5), (4, 6), (5, 6)]
g_ideal.add_edges_from(edge_set + [(u + 6, v + 6) for (u, v) in edge_set])
pos = { 
        1: (0, 0), 
        2: (1, 0),
        3: (2, 0),
        4: (0, -1),
        5: (1, -1),
        6: (0, -2),
    }
nx.draw(
    g_ideal,
    node_size = [75, 25, 25, 25, 25, 25] * 2,
    node_color = (["green"] * 6) + (["red"] * 6),
    pos = pos | {n + 6: (2 - x + 0.2, -2 - y) for (n, (x, y)) in pos.items()}
)
plt.gca().set_aspect('equal')
plt.show()


# example graph
ga27 = nx.graph_atlas(27)
nx.draw(ga27, 
    node_size = 50, 
    node_color = sns.color_palette("Set2", 5), 
    pos = nx.spring_layout(ga27, k = 5)
)
plt.gca().set_aspect('equal')
plt.show()