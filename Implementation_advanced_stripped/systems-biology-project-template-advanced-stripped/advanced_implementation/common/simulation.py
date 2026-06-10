import numpy as np
import sund


def populate_activity_from_input_dict(act: sund.Activity, inputs: dict, use_as_feature: bool = True) -> None:
    
    for key, inp in inputs.items():
        if "type" in inp:
            if inp["type"].lower() == "constant":
                act.add_output(
                    name=key, type=inp["type"].lower(),
                    f=np.array( inp["f"]), feature=use_as_feature
                )

            elif inp["type"].lower() == "piecewise_constant" or inp["type"].lower() == "piecewise_linear":
                # ignore first "-INFINITY" entry in time vector
                act.add_output(
                    name=key, type=inp["type"].lower().replace(" ", "_"), t=np.array(inp["t"][1:]), 
                    f=np.array(inp["f"]), feature=use_as_feature
                )
            
            else:
                print(
                    f"Provided activity type: {inp['type']} - is not recognized as a sund activity type. No activity added.")
        
        elif "mode" in inp:
            if inp["mode"].lower() == "add" or inp["mode"].lower() == "set":
                act.add_state_manipulation(
                    name=inp["name"], mode=inp["mode"].lower(), 
                    t = np.array(inp["t"]), f=np.array(inp["f"]) 
                )
            
            else:
                print(
                    f"Provided activity mode: {inp['mode']} - is not recognized as a sund activity mode. No activity added.")


def create_sim_from_input_dict(model, inputs: dict, time_points: list[float], use_as_feature: bool = True) -> sund.Simulation:
    # Define simulation object creation helper
    act = sund.Activity(time_unit=model.time_unit)
    populate_activity_from_input_dict(act, inputs, use_as_feature)
    
    # note that the time unit can be changed and do not need to be the same as the model
    sim = sund.Simulation(
        models=model, activities=act,
        time_unit=model.time_unit, time_vector=time_points
    )
    return sim


def create_sims_from_data(model, data: dict) -> dict[str, sund.Simulation]:
    # Setup simulation object from data
    sims = dict()
    for k, v in data.items():
        sims[k] = create_sim_from_input_dict(model, v["input"], v["all_times"])
    return sims


def simulate_model_steady_state(sim: sund.Simulation, ic: list[float], params: np.ndarray) -> None:
    steady_state_time_vector = np.linspace(-1000, 0, 11)
    sim.simulate(
        time_vector=steady_state_time_vector,
        state_values=ic, parameter_values=params)


def simulate_model(sim: sund.Simulation, ic: list[float], params: np.ndarray, time_vector: list[float] = []) -> None:
    # in case of plotting - we want to simulate using a time vector with higher resolution
    if len(time_vector) > 0:
        sim.simulate(
            time_vector=time_vector, state_values=ic,
            parameter_values=params, iterative=True
        )
    else:
        sim.simulate(state_values=ic, parameter_values=params, iterative=True)
