import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("outcomes_covars.csv").drop(columns = ["Unnamed: 0"])
# data = data.join(pd.get_dummies(data["village"], prefix = "v"))

T = data[data.treatment == 1]
C = data[data.treatment == 0]

# sm implementations - log-odds known

covars   = ['rooftype2', 'rooftype3', 'rooftype4', 'rooftype5', 'room_no', 'bed_no', 'electricity_government', 'electricity_no', 'latrine_common', 'latrine_none', 'ownrent_owned', 'ownrent_owned_but_shared', 'ownrent_rented', 'ownrent_leased', 'ownrent_given_by_government', 'ownrent_6']
# controls = " + ".join(covars + [col for col in data.columns if col.startswith('v')][:-1])
controls = " + ".join(covars)
formula  = f"Y_raw ~ {controls} + DE + WT"

reg = smf.logit(formula = formula, data = data).fit()
reg.summary()
reg1 = smf.logit(formula = formula, data = T).fit()
reg1.summary()
reg0 = smf.logit(formula = formula, data = C).fit()
reg0.summary()

# sklearn implementations - log-odds unknown, only binary variables observed
LogisticRegression().fit(T[covars + ["DE", "WT"]], T["Y"]).coef_
LogisticRegression().fit(C[covars + ["DE", "WT"]], C["Y"]).coef_