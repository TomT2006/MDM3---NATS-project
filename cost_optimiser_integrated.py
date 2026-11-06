"""ensemble sweep one model parameter. plots cost.

eg:
    rows = run_parameter_sweep(
        "stands_per_port",
        values=np.linspace(2, 8, 7),
        seeds=(1, 2, 3),
    )

Note: code only works on my computer when i run "python "cost_optimiser_integrated.py"" from terminal.
may work normally on your machine, but if it doesnt, type the above into terminal.
"""
import csv
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

from model import ScenarioConfig, Simulation
from results import measurements


# baseline model inputs, subject to change
BASELINE_PARAMETERS = {
    "port_names": ("A", "B", "C", "D"),
    "fleet": 8,
    "seats": 4,
    "stands_per_port": 4,
    "queue_capacity": 4,
    "chargers_per_port": 1,
    "chargers_A": 2,
    "chargers_B": 0,
    "chargers_C": 1,
    "chargers_D": 0,
    "pads_per_port": 1,
    "initial_soc": 0.9,
    "routes_km": None,
    "default_distance_km": 30.0,
    "speed_kmh": 200.0,
    "headway_min": 3.0,
    "taxi_out_min": 1.0,
    "pad_service_mean_min": 1.0,
    "taxi_in_min": 1.0,
    "board_min_per_passenger": 1.0,
    "unload_min_per_passenger": 1.0,
    "battery_kwh": 100.0,
    "vertical_energy_kwh": 8.0,
    "cruise_kwh_per_km": 0.4,
    "charger_kw": 120.0,
    "charging_efficiency": 1.0,
    "reserve_soc": 0.15,
    "charge_trigger_soc": 0.6,
    "charge_target_soc": 0.9,
    "holding_kw": 30.0,
    "holding_quantile": 0.99,
    "mission_energy_margin": 0.1,
    "max_charger_wait_min": 60.0,
    "total_demand_per_hour": 36.0,
    "demand_per_hour": None,
    "demand_mode": "poisson",
    "timetable_interval_min": 30.0,
    "service_intervals_min": {},
    "first_departure_min": 10.0,
    "max_assignment_delay_min": 30.0,
    "decision_interval_min": 0.5,
    "ground_delay_mean_min": 0.0,
    "demand_stop_min": 600.0,
    "horizon_min": 720.0,
    "time_scale_min": 10.0,
    "length_scale_km": 30.0,
    "wait_target_min": 60.0,
    "served_target": 0.95,
    "timely_target": 0.95,
}

# cost function weightings
COST_WEIGHTS = {
    "passenger_wait_min": 1.0,
    "uncompleted_passenger": 200.0,
    "flight": 60.0,
    "aircraft_hour": 50.0,
}

DEFAULT_MULTIPLIERS = np.linspace(0.5, 2.0, 10)

INTEGER_PARAMETERS = {
    "fleet",
    "seats",
    "stands_per_port",
    "queue_capacity",
    "chargers_per_port",
    "chargers_A",
    "chargers_B",
    "chargers_C",
    "chargers_D",
    "pads_per_port",
}


def build_config(parameters, seed):
    """creates a validated model configuration from baseline/sweep parameter values."""
    config_values = deepcopy(parameters)
    ports = config_values["port_names"]
    
    port_overrides = {}
    for port in ports:
        key = f"chargers_{port}"
        if key in config_values:
            port_overrides[port] = {"chargers": config_values.pop(key)}
            
    config_values["port_overrides"] = port_overrides
    config_values["seed"] = seed
    
    config = ScenarioConfig(**config_values)
    config.validate()
    return config


def calculate_cost(components, weights=COST_WEIGHTS):
    #calculates cost duh
    return (weights["passenger_wait_min"] * components["total_wait_min"]
            + weights["uncompleted_passenger"] * components["uncompleted_passengers"] 
            + weights["flight"] * components["flights"] 
            + weights["aircraft_hour"] * components["aircraft_hours"])


def score_run(simulation, weights=COST_WEIGHTS):
    #returns cost components and model metrics for a completed simulation
    config = simulation.config
    metrics = measurements(simulation)
    end = simulation.env.now
    
    total_wait_min = sum(
        max(
            0.0,
            float(
                (passenger.departed if passenger.departed is not None else end)
                - passenger.arrived
            )
            * config.time_scale_min,
        )
        for passenger in simulation.passengers
    )
    
    components = {
        "total_wait_min": total_wait_min,
        "uncompleted_passengers": metrics["arrivals"] - metrics["completed"],
        "flights": metrics["flights"],
        "aircraft_hours": config.fleet * config.horizon_min / 60.0,
    }
    
    return {
        "cost": calculate_cost(components, weights),
        **components,
        "arrivals": metrics["arrivals"],
        "completed": metrics["completed"],
        "mean_wait_min": metrics["mean_wait_min"],
        "p95_wait_min": metrics["p95_wait_min"],
        "served_fraction": metrics["served_fraction"],
        "timely_fraction": metrics["timely_fraction"],
        "reposition_flights": metrics["reposition_flights"],
        "safety_failures": metrics["safety_failures"],
    }


def get_sweep_values(parameter, values, multipliers, baseline):

    #determines the sequence of values to sweep.

    if values is not None:
        selected = list(values)
    else:
        selected = [baseline[parameter] * factor for factor in multipliers]

    if parameter in INTEGER_PARAMETERS:
        selected = [int(round(v)) for v in selected]
        
    return list(dict.fromkeys(selected))


def get_seeds(seeds, value):
    """returns the ensemble seeds for a given parameter value"""
    if isinstance(seeds, dict):
        return tuple(seeds.get(value, []))
    return tuple(seeds)


def run_parameter_sweep(
    parameter,
    values=None,
    *,
    multipliers=DEFAULT_MULTIPLIERS,
    seeds=(1, 2, 3),
    baseline=BASELINE_PARAMETERS,
    cost_weights=COST_WEIGHTS,
    csv_path="cost_sweep_results.csv",
    make_plot=True,
    show_progress=True,
):
    """runs an ensemble per value, saves the scored runs, and plots"""
    parameter_values = get_sweep_values(parameter, values, multipliers, baseline)
    seed_plan = [(value, get_seeds(seeds, value)) for value in parameter_values]
    
    total_runs = sum(len(s) for _, s in seed_plan)
        
    if show_progress:
        print(f"Starting {total_runs} simulations for '{parameter}'.", flush=True)

    rows = []
    completed = 0

    for value, selected_seeds in seed_plan:
        for seed in selected_seeds:
            run_parameters = deepcopy(baseline)
            run_parameters[parameter] = value
            
            sim = Simulation(build_config(run_parameters, seed)).run()
            rows.append({
                "parameter": parameter,
                "parameter_value": value,
                "seed": seed,
                **score_run(sim, cost_weights),
            })
            
            completed += 1
            if show_progress:
                print(f"\rProgress: {completed}/{total_runs} | {parameter}={value} | seed={seed}", end="", flush=True)

    if show_progress:
        print(flush=True)

    write_csv(rows, csv_path)
    
    if make_plot and rows:
        plot_sweep_results(rows, parameter)
        plt.show()
        
    return rows


def write_csv(rows, csv_path):
    """writes one csv row per simulation replication."""
    if not rows:
        return
        
    path = Path(csv_path)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep_results(rows, parameter, ax=None):
    """plots the mean cost by parameter value, with std bars."""
    grouped = {}
    for row in rows:
        val = float(row["parameter_value"])
        cost = float(row["cost"])
        grouped.setdefault(val, []).append(cost)
        
    x_values = sorted(grouped)
    means = [mean(grouped[v]) for v in x_values]
    deviations = [pstdev(grouped[v]) for v in x_values]

    if ax is None:
        _, ax = plt.subplots()
        
    ax.errorbar(x_values, means, yerr=deviations, marker="o", capsize=3)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Cost")
    ax.set_title(f"cost by {parameter}")
    ax.grid(True, alpha=0.3)
    return ax


def plot_from_csv(csv_path):
    """reads an existing csv file and produceses the cost plot without rerunning simulations."""
    path = Path(csv_path)
    if not path.exists():
        print(f"File {csv_path} not found.")
        return
        
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    if not rows:
        print("file is empty.")
        return
        
    parameter_name = rows[0].get("parameter", "unknown Parameter")
    plot_sweep_results(rows, parameter_name)
    plt.show()


if __name__ == "__main__":
    
    mode = input("type 1 to run a new simulation sweep, or 2 to plot from an existing csv: ").strip()
    
    if mode == '2':
        target_file = input("enter the csv filename. example file is [stands_sweep.csv]: ").strip()
        plot_from_csv(target_file)
    else:
        # example scenario
        # in this example the cost function decreases SIGNIFICANTLY going from 2 stands per port to 3.
        rows = run_parameter_sweep(
            parameter="stands_per_port",
            values=np.linspace(2, 8, 7),
            seeds=range(1,31),
            csv_path="stands_sweep.csv",
            make_plot=True,
        )
        sweeps = [("fleet", range(4,13), "fleet_sweep.csv"),
                  ("stands_per_port", [1,2,3,4], "stands_sweep.csv"),
        ]

        for parameter, values, path in sweeps:
            run_parameter_sweep(
                parameter=parameter,
                values=values,
                seeds=range(1,31),
                csv_path=path,
                make_plot=True,
            )


        