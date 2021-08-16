import sys

from tigramite import data_processing
from tigramite.independence_tests import CMIknn
from tigramite.pcmci import PCMCI
import pandas as pd
import numpy as np

good_trials = ['DAP094(1)']
features = ["coughing", "pm2_5", "temperature", "humidity"]


for trial in good_trials:
    full_p = list()
    full_v = list()
    print("Trial: {}".format(trial))
    try:
        data = pd.read_csv(f"../../data/DAPHNE/PCMCI/{trial}.csv",
                                  infer_datetime_format=True, parse_dates=["timestamp"], index_col="timestamp")
        data = data[features]
        # Create a dataframe with the trial data
    except:
        print("ERROR")
        break
    # Create a dataframe with the trial data
    dataframe = data_processing.DataFrame(data.values, missing_flag=999.)
    # Initialise the non-linear CMIknn test
    cmi_knn = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', n_jobs=-1)
    # Initialise PCMCI
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cmi_knn, verbosity=3)
    # For 7 lags in the first hour and every ten minutes afterwards up to the 8th hour
    for lag in [1, 5, 10, 15, 30, 45, 60] + [i * 10 for i in range(7, 49)]:
        print("Time lag: {}".format(lag))
        # Use a RNG seed to reproduce results
        np.random.seed(0)
        # Run PCMCI
        results = pcmci.run_pcmci(tau_min=lag, tau_max=lag, pc_alpha=0.05)
        pvalues = results["p_matrix"][1][0][-1]
        stats = results["val_matrix"][1][0][-1]
        print("Lag {}".format(lag))
        print(pvalues)
        print(stats)
        full_p.append(pvalues)
        full_v.append(stats)

        #df = pd.DataFrame(np.array(full_p))
        #df.to_csv("results/p/non_linear_p_trial_{}_{}_8h.csv".format(trial, lag), index=False)

        #df = pd.DataFrame(np.array(full_v))
        #df.to_csv("results/v/non_linear_v_trial_{}_{}_8h.csv".format(trial, lag), index=False)

    try:
        # Save results
        df = pd.DataFrame(np.array(full_p).reshape(1, 49),
                          columns=[1, 5, 10, 15, 30, 45, 60] + [i * 10 for i in range(7, 49)])
        df.to_csv("results/p/non_linear_p_{}_8h.csv".format(trial), index=False)

        df = pd.DataFrame(np.array(full_v).reshape(1, 49),
                          columns=[1, 5, 10, 15, 30, 45, 60] + [i * 10 for i in range(7, 49)])
        df.to_csv("results/v/non_linear_v_{}_8h.csv".format(trial), index=False)

    except:
        continue