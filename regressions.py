import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("outcomes_covars.csv").drop(columns = ["Unnamed: 0"])
# data = data.join(pd.get_dummies(data["village"], prefix = "v"))

treatment = data.treatment == 1
control   = data.treatment == 0

covars   = ['rooftype2', 'rooftype3', 'rooftype4', 'rooftype5', 'room_no', 'bed_no', 'electricity_government', 'electricity_no', 'latrine_common', 'latrine_none', 'ownrent_owned', 'ownrent_owned_but_shared', 'ownrent_rented', 'ownrent_leased', 'ownrent_given_by_government', 'ownrent_6']
# controls = " + ".join(covars + [col for col in data.columns if col.startswith('v')][:-1])
controls = " + ".join(covars)
formula  = f"Y_raw ~ {controls} + DE + WT"

reg = smf.logit(formula = formula, data = data[treatment]).fit()
reg.summary()



reg1 = smf.logit(formula = formula, data = data[treatment]).fit()
reg1.summary()

reg0 = smf.logit(formula = formula, data = data[control]  ).fit()
reg0.summary()

# lin estimator
