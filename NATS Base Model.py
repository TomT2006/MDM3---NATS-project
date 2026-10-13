import simpy
import random
import csv
import statistics
import math
import os
import matplotlib.pyplot as plt


# simulation settings
sim_time = 960
charge_rate = 2
reserve = 25
chargers_per_port = 4
passengers_per_flight = 4
unload_time = 3
board_time = 5
planned_ground_allowance = 20

ports = ["A", "B", "C"]

# route - flight time and battery used
routes = {
    ("A", "B"): (15, 20),
    ("B", "A"): (15, 20),
    ("A", "C"): (25, 30),
    ("C", "A"): (25, 30),
    ("B", "C"): (35, 40),
    ("C", "B"): (35, 40)
}
# FCFS - First come first served, LBF - Lowest Battery First,
# NFBM- Next Flight batter margin SPT - Shortest processing time (shortest charging time to required)
methods = ["FCFS", "LBF", "NFBM", "SPT"]
targets = ["full", "minimum"]
fleet_sizes = [12, 24, 36, 48]
repeats = 30

output_folder = "evtol_graphs"


def percentile(values, percent):
    if not values:
        return 0

    values = sorted(values)
    position = (len(values) - 1) * percent / 100

    low = math.floor(position)
    high = math.ceil(position)

    return values[low] + (values[high] - values[low]) * (position - low)


def mean_or_zero(values):
    if not values:
        return 0

    return statistics.mean(values)


# same routes and arrival/departure timetable for each charging method so that they
are comparible
def make_flight_plans(fleet_size, seed):
    plans = {}

    for i in range(fleet_size):
        rng = random.Random(seed * 100000 + i)

        port = ports[i % 3]
        start_delay = (i * 7 % 11) / 2
        scheduled_departure = start_delay + board_time

        journeys = []

        for _ in range(100):
            choices = [p for p in ports if p != port]
            destination = rng.choice(choices)

            flight_minutes, battery_use = routes[(port, destination)]

            journeys.append({
                "origin": port,
                "destination": destination,
                "flight_minutes": flight_minutes,
                "battery_use": battery_use,
                "scheduled_departure": scheduled_departure
            })

            # longer routes have later planned departure times
            scheduled_departure += (
                flight_minutes
                + unload_time
                + board_time
                + planned_ground_allowance
            )

            port = destination

        plans[i] = {
            "start_delay": start_delay,
            "journeys": journeys
        }

    return plans

# simpy gives priority to lower numbers!
# I am assuming that aircraft already charging are not interrupted

def charging_priority(method, battery, required, charge_minutes):
    if method == "LBF":
        return battery

    if method == "NFBM":
        return battery - required

    if method == "SPT":
        return charge_minutes

    return 0


def charge_aircraft(env, aircraft, port, chargers, method, target, results):
    battery = aircraft["battery"]
    required = aircraft["required"]

    if target == "full":
        target_battery = 100
    else:
        target_battery = required

    charge_minutes = (target_battery - battery) / charge_rate

    priority = charging_priority(
        method, battery, required, charge_minutes
    )

    request_time = env.now

    results["requests"] += 1
    results["queue_entries"][port] += 1

    if method == "FCFS":
        request = chargers[port].request()
    else:
        request = chargers[port].request(priority=priority)

    with request as req:
        yield req

        wait = env.now - request_time

        results["waits"].append(wait)
        results["waits_by_port"][port].append(wait)
        results["charging_sessions"] += 1

        results["charging_starts"].append(
            (port, env.now, charge_minutes)
        )

        yield env.timeout(charge_minutes)

        aircraft["battery"] = target_battery


def evtol(env, aircraft, chargers, method, target, results):
    yield env.timeout(aircraft["start_delay"])

    port = aircraft["port"]
    battery = aircraft["battery"]
    arrival_time = None

    for journey in aircraft["flight_plan"]:
        destination = journey["destination"]
        flight_minutes = journey["flight_minutes"]
        battery_use = journey["battery_use"]
        scheduled_departure = journey["scheduled_departure"]

        if arrival_time is not None:
            yield env.timeout(unload_time)

        required = battery_use + reserve

        aircraft["required"] = required
        aircraft["battery"] = battery

        # only charge if there is not enough battery for the next flight
        if battery < required:
            yield env.process(
                charge_aircraft(
                    env,
                    aircraft,
                    port,
                    chargers,
                    method,
                    target,
                    results
                )
            )

            battery = aircraft["battery"]

        yield env.timeout(board_time)

        # wait for the scheduled departure if ready early
        if env.now < scheduled_departure:
            yield env.timeout(scheduled_departure - env.now)

        # don't start flights that cannot finish before closing
        if env.now + flight_minutes >= sim_time:
            break

        departure_time = env.now
        departure_delay = max(0, departure_time - scheduled_departure)

        results["departure_delays"].append(departure_delay)
        results["scheduled_departures_completed"] += 1

        if departure_delay <= 0.000001:
            results["on_time_departures"] += 1

        if arrival_time is not None:
            results["turnarounds"].append(
                departure_time - arrival_time
            )

        results["departures"].append(
            (
                aircraft["id"],
                port,
                departure_time,
                scheduled_departure
            )
        )

        yield env.timeout(flight_minutes)

        battery -= battery_use
        port = destination
        arrival_time = env.now

        results["flights"] += 1
        results["flights_by_port"][port] += 1
        results["flights_per_aircraft"][aircraft["id"]] += 1


def run_simulation(method, target, fleet_size, seed, flight_plans=None):
    if flight_plans is None:
        flight_plans = make_flight_plans(fleet_size, seed)

    env = simpy.Environment()

    if method == "FCFS":
        resource_type = simpy.Resource
    else:
        resource_type = simpy.PriorityResource

    chargers = {
        p: resource_type(env, capacity=chargers_per_port)
        for p in ports
    }

    results = {
        "flights": 0,
        "requests": 0,
        "charging_sessions": 0,
        "waits": [],
        "turnarounds": [],
        "charging_starts": [],
        "departures": [],
        "departure_delays": [],
        "on_time_departures": 0,
        "scheduled_departures_completed": 0,
        "queue_entries": {p: 0 for p in ports},
        "waits_by_port": {p: [] for p in ports},
        "flights_by_port": {p: 0 for p in ports},
        "flights_per_aircraft": {
            i: 0 for i in range(fleet_size)
        }
    }

    for i in range(fleet_size):
        aircraft = {
            "id": i,
            "port": ports[i % 3],
            "battery": 100,
            "flight_plan": flight_plans[i]["journeys"],
            "start_delay": flight_plans[i]["start_delay"]
        }

        env.process(
            evtol(
                env,
                aircraft,
                chargers,
                method,
                target,
                results
            )
        )

    env.run(until=sim_time)

    # calculate how much each vertiport's chargers were used
    occupied = {p: 0 for p in ports}

    for port, start, duration in results["charging_starts"]:
        occupied[port] += max(
            0,
            min(duration, sim_time - start)
        )

    utilisation = {
        p: 100 * occupied[p] / (chargers_per_port * sim_time)
        for p in ports
    }

    pending = {
        p: len(chargers[p].queue)
        for p in ports
    }

    waits = results["waits"]
    turnarounds = results["turnarounds"]
    delays = results["departure_delays"]

    departures = results["scheduled_departures_completed"]

    if departures > 0:
        on_time_percent = (
            100 * results["on_time_departures"] / departures
        )
    else:
        on_time_percent = 0

    return {
        "method": method,
        "target": target,
        "fleet": fleet_size,
        "seed": seed,
        "flights": results["flights"],
        "passengers": results["flights"] * passengers_per_flight,
        "flights_per_aircraft": results["flights"] / fleet_size,
        "mean_wait": mean_or_zero(waits),
        "p95_wait": percentile(waits, 95),
        "max_wait": max(waits, default=0),
        "mean_turnaround": mean_or_zero(turnarounds),
        "on_time_percent": on_time_percent,
        "mean_departure_delay": mean_or_zero(delays),
        "p95_departure_delay": percentile(delays, 95),
        "on_time_departures": results["on_time_departures"],
        "scheduled_departures_completed": departures,
        "charging_sessions_started": results["charging_sessions"],
        "charging_requests": results["requests"],
        "pending_charger_requests": sum(pending.values()),
        "util_A": utilisation["A"],
        "util_B": utilisation["B"],
        "util_C": utilisation["C"],
        "util_imbalance": (
            max(utilisation.values()) - min(utilisation.values())
        )
    }


def save_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


def average_result(rows, method, target, fleet, column):
    values = [
        r[column]
        for r in rows
        if r["method"] == method
        and r["target"] == target
        and r["fleet"] == fleet
    ]

    return statistics.mean(values)


# save tables with a thicker line after each fleet size
def save_table_png(filename, title, headers, rows, group_size=None):
    fig_height = max(2.0, 0.43 * len(rows) + 1.25)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=18
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D9DFE7")
        cell.set_linewidth(0.6)

        if r == 0:
            cell.set_facecolor("#243B53")
            cell.set_text_props(
                color="white",
                weight="bold"
            )

        elif r % 2 == 0:
            cell.set_facecolor("#EFF3F8")

    if group_size is not None:
        fig.canvas.draw()

        last_col = len(headers) - 1

        for row_number in range(
            group_size,
            len(rows),
            group_size
        ):
            cell = table[(row_number, 0)]

            y = cell.get_y()
            x_left = cell.get_x()

            right_cell = table[(row_number, last_col)]
            x_right = (
                right_cell.get_x() + right_cell.get_width()
            )

            ax.plot(
                [x_left, x_right],
                [y, y],
                color="#243B53",
                linewidth=2.5,
                transform=ax.transAxes,
                clip_on=False,
                zorder=10
            )

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=190,
        bbox_inches="tight"
    )

    plt.close(fig)


def make_graphs(all_results):
    os.makedirs(output_folder, exist_ok=True)

    # remove old generated pngs so only the current results remain
    old_graphs = [
        "average_wait_full.png",
        "average_wait_minimum.png",
        "p95_wait_full.png",
        "p95_wait_minimum.png",
        "completed_flights_full.png",
        "completed_flights_minimum.png",
        "paired_flight_comparison.png",
        "charger_utilisation_table.png"
    ]

    for filename in old_graphs:
        path = os.path.join(output_folder, filename)

        if os.path.exists(path):
            os.remove(path)

    method_colours = {
        "FCFS": "tab:blue",
        "LBF": "tab:orange",
        "NFBM": "tab:green",
        "SPT": "tab:red"
    }

    # make one combined graph for each waiting time measurement
    graph_settings = [
        (
            "mean_wait",
            "Average charger wait (minutes)",
            "average_wait_combined"
        ),
        (
            "p95_wait",
            "P95 charger wait (minutes)",
            "p95_wait_combined"
        )
    ]

    for column, ylabel, filename in graph_settings:
        plt.figure(figsize=(10, 6))

        for method in methods:
            for target in targets:
                values = [
                    average_result(
                        all_results,
                        method,
                        target,
                        fleet,
                        column
                    )
                    for fleet in fleet_sizes
                ]

                if target == "full":
                    line_style = "-"
                else:
                    line_style = "--"

                plt.plot(
                    fleet_sizes,
                    values,
                    marker="o",
                    linestyle=line_style,
                    color=method_colours[method],
                    label=method + " - " + target.upper()
                )

        plt.title(
            ylabel + " - FULL vs MINIMUM charging"
        )

        plt.xlabel("Number of aircraft")
        plt.ylabel(ylabel)
        plt.xticks(fleet_sizes)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=9, ncol=2)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_folder,
                filename + ".png"
            ),
            dpi=180
        )

        plt.close()

    # compare completed flights for all eight combinations
    plt.figure(figsize=(10, 6))

    for method in methods:
        for target in targets:
            values = [
                average_result(
                    all_results,
                    method,
                    target,
                    fleet,
                    "flights"
                )
                for fleet in fleet_sizes
            ]

            if target == "full":
                line_style = "-"
            else:
                line_style = "--"

            plt.plot(
                fleet_sizes,
                values,
                marker="o",
                linestyle=line_style,
                color=method_colours[method],
                label=method + " - " + target.upper()
            )

    plt.title(
        "Completed flights - FULL vs MINIMUM charging"
    )

    plt.xlabel("Number of aircraft")
    plt.ylabel("Average completed flights")
    plt.xticks(fleet_sizes)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            "full_vs_minimum.png"
        ),
        dpi=180
    )

    plt.close()

    # save the average results for every method and fleet size
    summary_columns = [
        "flights",
        "passengers",
        "mean_wait",
        "p95_wait",
        "mean_turnaround",
        "on_time_percent",
        "mean_departure_delay",
        "p95_departure_delay",
        "util_A",
        "util_B",
        "util_C",
        "pending_charger_requests"
    ]

    summary = []

    for fleet in fleet_sizes:
        for target in targets:
            for method in methods:
                row = {
                    "fleet": fleet,
                    "target": target,
                    "method": method
                }

                for column in summary_columns:
                    row[column] = average_result(
                        all_results,
                        method,
                        target,
                        fleet,
                        column
                    )

                summary.append(row)

    save_csv("evtol_summary.csv", summary)

    # show all methods together in the full and minimum tables
    headers = [
        "Fleet",
        "Method",
        "Flights",
        "Mean wait",
        "P95 wait",
        "On time %",
        "Dep. delay",
        "P95 delay",
        "Pending"
    ]

    for target in targets:
        table_rows = []

        for fleet in fleet_sizes:
            for index, method in enumerate(methods):
                r = next(
                    x for x in summary
                    if x["fleet"] == fleet
                    and x["target"] == target
                    and x["method"] == method
                )

                # only show the fleet number on the first row
                fleet_label = (
                    str(fleet) if index == 0 else ""
                )

                table_rows.append([
                    fleet_label,
                    method,
                    f'{r["flights"]:.1f}',
                    f'{r["mean_wait"]:.1f}',
                    f'{r["p95_wait"]:.1f}',
                    f'{r["on_time_percent"]:.1f}',
                    f'{r["mean_departure_delay"]:.1f}',
                    f'{r["p95_departure_delay"]:.1f}',
                    f'{r["pending_charger_requests"]:.1f}'
                ])

        save_table_png(
            os.path.join(
                output_folder,
                f"results_table_{target}.png"
            ),
            (
                f"Simulation results - {target.upper()} charging "
                f"({repeats}-run averages)"
            ),
            headers,
            table_rows,
            group_size=4
        )

    print("\nSaved 3 graphs and 2 results tables.")
    print("Saved evtol_summary.csv.")
    print("Wait times exclude aircraft still queued at closing.")
    print("On-time percentage is based on flights that departed.")


def run_experiments():
    # 4 methods x 2 targets x 4 fleets x 30 repeats = 960 runs
    all_results = []

    for fleet in fleet_sizes:
        for seed in range(1, repeats + 1):

            # generate once so all methods use the same timetable
            flight_plans = make_flight_plans(
                fleet,
                seed
            )

            for target in targets:
                for method in methods:
                    result = run_simulation(
                        method,
                        target,
                        fleet,
                        seed,
                        flight_plans
                    )

                    all_results.append(result)

    save_csv(
        "evtol_runs.csv",
        all_results
    )

    print(
        "Method  Target    Fleet   Flights  Avg wait  P95 wait  "
        "On time %  Avg delay  P95 delay  Pending"
    )

    for fleet in fleet_sizes:
        for target in targets:
            for method in methods:
                rows = [
                    r for r in all_results
                    if r["fleet"] == fleet
                    and r["target"] == target
                    and r["method"] == method
                ]

                def avg(key):
                    return statistics.mean(
                        r[key] for r in rows
                    )

                print(
                    f"{method:5}   "
                    f"{target:7}  "
                    f"{fleet:5}  "
                    f"{avg('flights'):7.1f}  "
                    f"{avg('mean_wait'):8.1f}  "
                    f"{avg('p95_wait'):8.1f}  "
                    f"{avg('on_time_percent'):9.1f}  "
                    f"{avg('mean_departure_delay'):9.1f}  "
                    f"{avg('p95_departure_delay'):9.1f}  "
                    f"{avg('pending_charger_requests'):7.1f}"
                )

    print(
        "\nSaved all",
        len(all_results),
        "runs to evtol_runs.csv"
    )

    make_graphs(all_results)

    print(
        "Graphs folder:",
        os.path.abspath(output_folder)
    )


if __name__ == "__main__":
    run_experiments()



