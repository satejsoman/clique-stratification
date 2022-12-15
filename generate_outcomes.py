import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

import matplotlib.pyplot as plt

plot = False

try:
    root = Path(__file__).parent
except NameError:
    root = Path.cwd()
data_dir = root / "data"
adj_matrices_dir = data_dir / "1. Network Data" / "Adjacency Matrices"

data = pd.read_stata(data_dir / "2. Demographics and Outcomes" / "household_characteristics.dta")\
    .drop(columns = ["hohreligion", "castesubcaste", "hhSurveyed"])\
    .assign(adjmatrix_key = lambda df: df["adjmatrix_key"] - 1)

dummies = [
    pd.get_dummies(data.electricity, prefix = "electricity") .rename(columns = lambda s: s.lower().replace("yes, ", "")),
    pd.get_dummies(data.latrine,     prefix = "latrine")     .rename(columns = lambda s: s.lower()),
    pd.get_dummies(data.ownrent,     prefix = "ownrent")     .rename(columns = lambda s: s.lower().replace(".0", "").replace(" ", "_"))
]

pattern = "adj_allVillageRelationships_HH_vilno_*.csv"
regex   = pattern.replace("*", "([0-9]+)")


A, G, L, delta = {}, {}, {}, {}
for path in adj_matrices_dir.glob(pattern):
    # load data 
    filename = path.name
    village = int(re.match(regex, path.name).group(1))
    a = np.genfromtxt(path, delimiter = ",")
    g = nx.from_numpy_matrix(a)
    l = nx.floyd_warshall_numpy(g)
    l[l == np.inf] = 0

    # annotate graph with leader indicator and adjust edge weights
    nx.set_node_attributes(g, 
        data[data.village == village][["adjmatrix_key", "leader"]].set_index("adjmatrix_key").to_dict()["leader"],
        "leader")
    nx.set_edge_attributes(g, 
        {(u, v): 10 if g.nodes[u]["leader"] or g.nodes[v]["leader"] else 1 for (u, v) in g.edges}, 
        "weight")
    
    # run kernighan-lin to bisect graph, with and without weights
    treatment_subgraph, _       = community.kernighan_lin_bisection(g, weight = "weight", seed = 0)
    naive_treatment_subgraph, _ = community.kernighan_lin_bisection(g, weight = None, seed = 0)

    assert treatment_subgraph != naive_treatment_subgraph

    nx.set_node_attributes(g, 
        {n: 1 if n in treatment_subgraph else 0 for n in g.nodes}, 
        "treatment")

    nx.set_node_attributes(g, 
        {n: 1 if n in naive_treatment_subgraph else 0 for n in g.nodes}, 
        "naive_treatment")

    A[village], G[village], L[village] = a, g, l 

if plot:
    degrees = nx.degree(g)
    t = {n for (n, attrs) in dict(g.nodes(data = True)).items() if attrs["treatment"] == 1 and degrees[n] > 0}
    c = {n for (n, attrs) in dict(g.nodes(data = True)).items() if attrs["treatment"] == 0 and degrees[n] > 0}
    nodes = t | c
    g_ = g.subgraph(nodes)

    pos = {n: p + np.array([0.5 if n in t else -0.5, 0]) for (n, p) in nx.spring_layout(g_).items()}
    node_colors = {n: "green" if n in t else "red" for n in nodes}
    edge_colors = ["black" if g.nodes[u]["treatment"] == g.nodes[v]["treatment"] else "grey" for (u, v) in g_.edges]
    sizes  = [75 if g.nodes[n]["leader"] else 25 for n in nodes]

    nx.draw(
        g_, 
        node_color = node_colors.values(), 
        node_size  = sizes, 
        edge_color = edge_colors, 
        pos = pos)
    plt.gca().set_aspect('equal')
    plt.show()

delta_D = []
for (v_id, g) in G.items():
    for (n, d) in dict(g.degree).items():
        delta_D.append((v_id, n, d, g.nodes[n]["treatment"], g.nodes[n]["naive_treatment"]))

data = data\
    .drop(columns = ['electricity', 'latrine', 'ownrent'])\
    .join(dummies)\
    .merge(pd.DataFrame(delta_D, columns = ['village', 'adjmatrix_key', 'degree', 'treatment', 'naive_treatment']))
data["D"]       = data["leader"] * data["treatment"]
data["D_naive"] = data["leader"]

# generate outcomes

## covariates
covariates = ['rooftype1', 'rooftype2', 'rooftype3', 'rooftype4', 'rooftype5', 'room_no', 'bed_no', 'leader', 'electricity_private', 'electricity_government', 'electricity_no', 'latrine_owned', 'latrine_common', 'latrine_none', 'ownrent_0', 'ownrent_owned', 'ownrent_owned_but_shared', 'ownrent_rented', 'ownrent_leased', 'ownrent_given_by_government', 'ownrent_6']
data[covariates] = data[covariates] - data[covariates].mean(axis = 0)
X = data[covariates].values

## parameters
np.random.seed(0)
beta  = np.random.normal(0,   0.5,  X.shape[1]) # covariates coefficients
gamma = np.random.normal(0.5, 0.25, X.shape[0]) # direct effects
kappa = np.random.normal(0.3, 0.15, X.shape[0]) # weak ties
nu    = np.random.normal(0, 0.5,    X.shape[0]) # idiosyncratic shock
intercept = -0.5

DE = [] # direct effects
WT = [] # weak ties

DE_naive = [] # direct effects
WT_naive = [] # weak ties
for village in G.keys():
    g = G[village]
    l = L[village]
    D = np.fromiter((g.nodes[n]["treatment"] for n in g.nodes), dtype = int)
    DE.extend( ((l == 1) @ D) / (l == 1).sum(axis = 1) )
    WT.extend( ((l >= 1) @ D) / (l >= 2).sum(axis = 1) )

    D_naive = np.fromiter((g.nodes[n]["naive_treatment"] for n in g.nodes), dtype = int)
    DE_naive.extend( ((l == 1) @ D_naive) / (l == 1).sum(axis = 1) )
    WT_naive.extend( ((l >= 1) @ D_naive) / (l >= 2).sum(axis = 1) )

data["DE"]       = pd.Series(DE).fillna(0)
data["WT"]       = pd.Series(WT).fillna(0)
data["DE_naive"] = pd.Series(DE_naive).fillna(0)
data["WT_naive"] = pd.Series(WT_naive).fillna(0)

data["z"]     = intercept + X @ beta + gamma * data["DE"] + kappa * data["WT"]
data["Y_raw"] = 1/(1 + np.exp(-data["z"]))
data["Y"]     = (data["Y_raw"] > 0.5).astype(int)

data["z_naive"]     = intercept + X @ beta + gamma * data["DE_naive"] + kappa * data["WT_naive"]
data["Y_raw_naive"] = 1/(1 + np.exp(-data["z_naive"]))
data["Y_naive"]     = (data["Y_raw_naive"] > 0.5).astype(int)

data.to_csv(root / "outcomes_covars.csv")
