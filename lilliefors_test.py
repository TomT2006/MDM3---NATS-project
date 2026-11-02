import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors


model_path = Path(__file__).with_name("NATS Base Model.py")
spec = importlib.util.spec_from_file_location("nats_base_model", model_path)

if spec is None or spec.loader is None:
    raise ImportError(f"could not find simulation model from {model_path}")

model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def main():
    gaps_by_port = {port: [] for port in model.ports}

    for seed in range(1, model.repeats + 1):
        result = model.run_simulation(
            "FCFS",
            "minimum",
            24,
            seed,
            record_arrivals=True
        )

        for port, arrival_times in result["arrivals_by_port"].items():
            ordered_times = sorted(arrival_times)
            gaps_by_port[port].extend(
                later - earlier
                for earlier, later in zip(
                    ordered_times,
                    ordered_times[1:]
                )
            )

    figure, axes = plt.subplots(
        1,
        len(model.ports),
        figsize=(15, 4.5),
        sharey=True
    )
    figure.suptitle(
        "arrival gap CDFs: actual vs fitted exponential"
    )
    difference_figure, difference_axes = plt.subplots(
        1,
        len(model.ports),
        figsize=(15, 4.5),
        sharey=True
    )
    difference_figure.suptitle(
        "abs difference between observed and fitted CDFs"
    )

    for port, gaps in gaps_by_port.items():
        port_index = model.ports.index(port)
        axis = axes[port_index]
        difference_axis = difference_axes[port_index]

        if len(gaps) < 5:
            print(f"Port {port}: too few arrival gaps to test (n={len(gaps)})")
            continue

        statistic, p_value = lilliefors(gaps, dist="exp")
        print(
            f"Port {port}: n={len(gaps)}, "
            f"D={statistic:.4f}, p={p_value:.4g}"
        )

        ordered_gaps = sorted(gaps)
        gap_count = len(ordered_gaps)
        mean_gap = sum(ordered_gaps) / gap_count
        empirical_probabilities = [
            index / gap_count
            for index in range(1, gap_count + 1)
        ]

        axis.step(
            [0] + ordered_gaps,
            [0] + empirical_probabilities,
            where="post",
            label="actual gaps"
        )

        max_gap = ordered_gaps[-1]
        x_values = [
            max_gap * index / 200
            for index in range(201)
        ]
        fitted_probabilities = [
            1 - math.exp(-x / mean_gap)
            for x in x_values
        ]
        axis.plot(
            x_values,
            fitted_probabilities,
            linestyle="--",
            label="Fitted exponential"
        )
        axis.set_title(f"Pad {port} (p={p_value:.3g})")
        axis.set_xlabel("time between arrivals (min)")
        axis.grid(alpha=0.3)
        axis.legend()

        fitted_at_gaps = [
            1 - math.exp(-gap / mean_gap)
            for gap in ordered_gaps
        ]
        difference_x = [
            gap for gap in ordered_gaps for _ in range(2)
        ]
        difference_y = []
        for index, fitted_probability in enumerate(fitted_at_gaps):
            difference_y.extend([
                abs(index / gap_count - fitted_probability),
                abs((index + 1) / gap_count - fitted_probability)
            ])

        difference_axis.plot(
            difference_x,
            difference_y,
            label="abs CDF difference"
        )
        difference_axis.axhline(
            statistic,
            color="tab:red",
            linestyle="--",
            label=f"Lilliefors D = {statistic:.4f}"
        )
        difference_axis.set_title(f"Pad {port}")
        difference_axis.set_xlabel("time between arrivals (min)")
        difference_axis.grid(alpha=0.3)
        difference_axis.legend()

    axes[0].set_ylabel("cumulative probability")
    figure.tight_layout()
    difference_axes[0].set_ylabel("abs cumulative probability difference")
    difference_figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
