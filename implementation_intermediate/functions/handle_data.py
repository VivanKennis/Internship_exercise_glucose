import json
import numpy as np

def load_and_process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f) # read the data from the json file

    for _, d in data.items():
        # make sure that "0" is included in the time vector
        times = [t for _, obs in d["Observables"].items() # get the time points from all observables in the experiment
            for t in obs["Time"]]+[0]
        # make sure that the event time points are included for the model events to trigger properly
        times += [t for _, inp in d["input"].items()
            for t in inp["t"] if float(t) != -np.inf]

        # create a time vector that includes the unique set of all time points from all the observables within the experiment
        # this allows us to only simulate an experiment once even if multiple observables are included for the experiment
        d["all_times"] = np.unique(np.array(times))
        for _, obs in d["Observables"].items():
            obs["Mean"] = np.array(obs["Mean"])
            obs["SEM"] = np.array(obs["SEM"])

    return data


def get_dof(data):
    dof = 0 # degrees of freedom
    # calculate the number of data points - exclude data points that has SEM=Infinity (often data points with a zero value)
    for _, d in data.items():
        for _, obs in d["Observables"].items():
            dof += len([element for element in obs["SEM"]
                if not float(element) == float('inf')])

    return dof
