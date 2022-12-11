library(magrittr)
library(tidyverse)
data  <- readr::read_csv("./outcomes_covars.csv")
f     <- Y     ~ rooftype2 + rooftype3 + rooftype4 + rooftype5 + room_no + bed_no + electricity_government + electricity_no + latrine_common + latrine_none + ownrent_owned + ownrent_owned_but_shared + ownrent_rented + ownrent_leased + ownrent_given_by_government + ownrent_6 + DE + WT
f_raw <- Y_raw ~ rooftype2 + rooftype3 + rooftype4 + rooftype5 + room_no + bed_no + electricity_government + electricity_no + latrine_common + latrine_none + ownrent_owned + ownrent_owned_but_shared + ownrent_rented + ownrent_leased + ownrent_given_by_government + ownrent_6 + DE + WT
glm(f, data = data, family = "binomial")
glm(f_raw, data = data)

prop <- glm(Y ~ leader + degree, data = data, family = "binomial")

predict(prop, data)
