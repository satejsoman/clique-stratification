import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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