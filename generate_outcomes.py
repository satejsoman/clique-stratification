import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community

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

dataframes = []
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
        {(u, v): 2 if g.nodes[u]["leader"] or g.nodes[v]["leader"] else 1 for (u, v) in g.edges}, 
        "weight")
    
    # run kernighan-lin to bisect graph
    treatment_subgraph, _ = community.kernighan_lin_bisection(g, weight = "weight", seed = 0)

    nx.set_node_attributes(g, 
        {n: 1 if n in treatment_subgraph else 0 for n in g.nodes}, 
        "treatment")

    A[village], G[village], L[village] = a, g, l 

delta_D = []
for (v_id, g) in G.items():
    for (n, d) in dict(g.degree).items():
        delta_D.append((v_id, n, d, g.nodes[n]["treatment"]))

data = data\
    .drop(columns = ['electricity', 'latrine', 'ownrent'])\
    .join(dummies)\
    .merge(pd.DataFrame(delta_D, columns = ['village', 'adjmatrix_key', 'degree', 'treatment']))
data["D"] = data["leader"] * data["treatment"]

# generate outcomes

## covariates
covariates = ['rooftype1', 'rooftype2', 'rooftype3', 'rooftype4', 'rooftype5', 'room_no', 'bed_no', 'leader', 'electricity_private', 'electricity_government', 'electricity_no', 'latrine_owned', 'latrine_common', 'latrine_none', 'ownrent_0', 'ownrent_owned', 'ownrent_owned_but_shared', 'ownrent_rented', 'ownrent_leased', 'ownrent_given_by_government', 'ownrent_6']
data[covariates] = data[covariates] - data[covariates].mean(axis = 0)
X = data[covariates].values

## parameters
np.random.seed(0)
# beta  = np.random.normal(0,   0.5,  X.shape[1]) # covariates coefficients
# gamma = np.random.normal(0.5, 0.25, X.shape[0]) # direct effects
# kappa = np.random.normal(0.3, 0.15, X.shape[0]) # weak ties
# nu    = np.random.normal(0, 0.5,    X.shape[0]) # idiosyncratic shock

beta      = [0.01 for _ in range(X.shape[1])]
gamma     = 0.5
kappa     = 0.3
intercept = -0.5

DE = [] # direct effects
WT = [] # weak ties
for village in G.keys():
    g = G[village]
    l = L[village]
    D = np.fromiter((g.nodes[n]["treatment"] for n in g.nodes), dtype = int)
    DE.extend( ((l == 1) @ D) / (l == 1).sum(axis = 1) )
    WT.extend( ((l >= 1) @ D) / (l >= 2).sum(axis = 1) )

data["DE"] = pd.Series(DE).fillna(0)
data["WT"] = pd.Series(WT).fillna(0)

data["z"]     = intercept + X @ beta + gamma * data["DE"] + kappa * data["WT"]
data["Y_raw"] = 1/(1 + np.exp(-data["z"]))
data["Y"]     = (data["Y_raw"] > 0.5).astype(int)

data.to_csv(root / "outcomes_covars.csv")