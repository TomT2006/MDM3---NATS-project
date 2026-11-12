"""
Viability-constrained optimiser for the eVTOL infrastructure sim.

What it does
  1. Defines viability constraints (completion, p95 queue length, utilisation).
  2. Uses Optuna's TPE sampler WITH constraints, so infeasible trials teach the
     sampler where the feasible region is (it learns from previous runs).
  3. Logs every evaluation to CSV (config, raw components, constraint margins,
     z-contributions, score, viable flag, best-so-far) for plotting.
  4. For each fleet size, finds the minimum value of EACH infrastructure
     parameter that keeps the system viable, and turns that into report
     sentences like "pads: 1 sufficient for fleet 2-8".

Replace run_sim() with your SimPy wrapper. It must return a dict with the keys
in COMPONENT_KEYS (per-run, one seed).
"""
import itertools
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------------------------------------------------------
# 1. SETTINGS
# ----------------------------------------------------------------------------
INFRA = ['pads', 'stands', 'chargers']                  # searched per fleet size
RANGES = {'fleet': (2, 30), 'pads': (1, 6), 'stands': (1, 12), 'chargers': (1, 12)}
DEMAND = 200                                            # passengers per operating day
SEEDS = list(range(8))                                  # common random numbers

# Viability = hard requirements. Edit thresholds and justify them in the report.
LIMITS = dict(min_completion=0.95,    # >= 95% of passengers completed
              max_p95_queue=5,        # p95 queue length at any node <= 5 aircraft
              max_util=0.85)          # busiest resource < 85% utilised

# Importance weights (normalised to sum to 1 below)
RAW_WEIGHTS = dict(total_wait_min=0.35, uncompleted_passengers=0.35,
                   flights=0.10, aircraft_hours=0.10, stand_hours=0.10)
W = {k: v / sum(RAW_WEIGHTS.values()) for k, v in RAW_WEIGHTS.items()}
COMPONENT_KEYS = list(W) + ['p95_queue_len', 'max_utilisation']

# Anchors: 0 = ideal (nothing bottlenecked), 1 = baseline (sensible guess)
BASELINE = dict(fleet=8, pads=1, stands=3, chargers=3)
IDEAL = dict(fleet=8, pads=6, stands=12, chargers=12)
GENEROUS = {p: RANGES[p][1] for p in INFRA}             # "others not limiting"

# Objective once viable: smallest infrastructure, ties broken by service score.
SIZE_WEIGHTS = dict(pads=1.0, stands=1.0, chargers=1.0)  # change if items differ in effort
TIEBREAK = 0.1


# ----------------------------------------------------------------------------
# 2. SIMULATION WRAPPER  (STAND-IN, replace with your SimPy model)
# ----------------------------------------------------------------------------
def run_sim(cfg, seed):
    """Stand-in so the script runs end to end. Replace with your sim."""
    rng = np.random.default_rng(seed)
    f, p, s, c = cfg['fleet'], cfg['pads'], cfg['stands'], cfg['chargers']
    rho = {'pad': f * 0.10 / p, 'stand': f * 0.20 / s, 'charger': f * 0.25 / c}
    rho = {k: v * rng.normal(1, 0.03) for k, v in rho.items()}
    q = {k: (min(v, 0.99) / (1 - min(v, 0.99))) for k, v in rho.items()}
    blocked = sum(max(0, v - 0.95) for v in rho.values())
    unc = DEMAND * min(1, 0.6 * blocked + max(0, 1 - f / 6) * 0.3)
    return dict(total_wait_min=40 * sum(q.values()) + rng.normal(0, 5),
                uncompleted_passengers=unc,
                flights=DEMAND / 4 + 0.5 * f + 2 * blocked,
                aircraft_hours=f * 6 + rng.normal(0, 0.5),
                stand_hours=s * 10,
                p95_queue_len=2.5 * max(q.values()),
                max_utilisation=min(1.0, max(rho.values())))


# ----------------------------------------------------------------------------
# 3. EVALUATION: components -> constraints -> z -> score  (with cache + log)
# ----------------------------------------------------------------------------
_cache, LOG = {}, []


def mean_components(cfg):
    key = tuple(sorted(cfg.items()))
    if key not in _cache:                                # sims are the expensive bit
        runs = pd.DataFrame([run_sim(cfg, s) for s in SEEDS])
        _cache[key] = runs[COMPONENT_KEYS].mean().to_dict()
    return _cache[key]


REF = mean_components(BASELINE)
IDL = mean_components(IDEAL)


# Terms where baseline == ideal have no span to normalise by (z undefined), so they cannot
# distinguish designs. Drop them and re-spread their weight so baseline scores exactly 1.
_active = {k for k in W if abs(REF[k] - IDL[k]) > 1e-9}
if _active != set(W):
    print('Note: terms with baseline == ideal, excluded from score:', sorted(set(W) - _active))
_tot = sum(W[k] for k in _active)
W = {k: (W[k] / _tot if k in _active else 0.0) for k in W}


def z_value(k, x):
    """Anchored: ideal -> 0, baseline -> 1, worse than baseline -> above 1."""
    span = REF[k] - IDL[k]
    return (x - IDL[k]) / span if abs(span) > 1e-9 else 0.0


def constraint_margins(comp):
    """<= 0 means satisfied. Positive = size of violation (what TPE learns from)."""
    completion = 1 - comp['uncompleted_passengers'] / DEMAND
    return [LIMITS['min_completion'] - completion,
            comp['p95_queue_len'] - LIMITS['max_p95_queue'],
            comp['max_utilisation'] - LIMITS['max_util']]


def evaluate(cfg, tag=''):
    comp = mean_components(cfg)
    margins = constraint_margins(comp)
    contrib = {f'c_{k}': W[k] * z_value(k, comp[k]) for k in W}
    size = sum(SIZE_WEIGHTS[p] * cfg[p] for p in INFRA)
    row = dict(tag=tag, **cfg, **comp, **contrib,
               score=sum(contrib.values()),
               m_completion=margins[0], m_queue=margins[1], m_util=margins[2],
               viable=all(m <= 0 for m in margins), infra_size=size)
    row['objective'] = size + TIEBREAK * row['score']
    row['eval_id'] = len(LOG)
    LOG.append(row)
    return row


# ----------------------------------------------------------------------------
# 4. OPTIMISER: per fleet size, learns from previous trials
# ----------------------------------------------------------------------------
def optimise_for_fleet(fleet, n_trials=60, seed=0):
    # Infeasible trials inform TPE. Optuna >=5 uses trial.set_constraint; older uses constraints_func.
    new_api = hasattr(optuna.trial.Trial, 'set_constraint')
    kw = {} if new_api else dict(constraints_func=lambda t: t.user_attrs['constraints'])
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=15, **kw)

    def objective(trial):
        cfg = {'fleet': fleet, **{p: trial.suggest_int(p, *RANGES[p]) for p in INFRA}}
        row = evaluate(cfg, tag=f'opt_fleet{fleet}')
        margins = [row['m_completion'], row['m_queue'], row['m_util']]
        if new_api:
            for name, m in zip(['completion', 'queue', 'util'], margins):
                trial.set_constraint(name, m)
        else:
            trial.set_user_attr('constraints', margins)
        return row['objective']

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    df = pd.DataFrame([r for r in LOG if r['tag'] == f'opt_fleet{fleet}'])
    ok = df[df.viable]
    return None if ok.empty else ok.sort_values('objective').iloc[0].to_dict()


# ----------------------------------------------------------------------------
# 5. PER-PARAMETER THRESHOLDS (the report sentences)
# ----------------------------------------------------------------------------
def single_param_thresholds(fleets):
    """For each fleet size and each parameter: the smallest value that is viable
    when every OTHER infrastructure item is at its maximum (non-limiting)."""
    rows = []
    for f in fleets:
        for p in INFRA:
            need = None
            for v in range(RANGES[p][0], RANGES[p][1] + 1):
                cfg = {**GENEROUS, 'fleet': f, p: v}
                if evaluate(cfg, tag=f'scan_{p}')['viable']:
                    need = v
                    break
            rows.append(dict(fleet=f, param=p, min_required=need))
    return pd.DataFrame(rows)


def report_sentences(th):
    out = []
    for p, g in th.groupby('param'):
        g = g.sort_values('fleet')
        g = g.assign(min_required=g.min_required.astype('object').where(g.min_required.notna(), None))
        for need, grp in itertools.groupby(g.itertuples(), key=lambda r: r.min_required):
            fl = [r.fleet for r in grp]
            span = f'{fl[0]}' if len(fl) == 1 else f'{fl[0]} to {fl[-1]}'
            if need is None:
                out.append(f'{p}: not viable even at {RANGES[p][1]} for fleet {span} '
                           f'(another resource or the fleet itself is limiting)')
            else:
                out.append(f'{p}: {int(need)} sufficient for fleet {span} '
                           f'(with other resources non-limiting)')
    return out


# ----------------------------------------------------------------------------
# 6. PLOTS
# ----------------------------------------------------------------------------
def plot_convergence(df, fleet):
    d = df[df.tag == f'opt_fleet{fleet}'].reset_index(drop=True)
    best = d.where(d.viable).objective.cummin()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(d.index[~d.viable], d.objective[~d.viable], c='tab:red', s=14, label='infeasible')
    ax.scatter(d.index[d.viable], d.objective[d.viable], c='tab:green', s=14, label='viable')
    ax.step(d.index, best, c='k', where='post', label='best viable so far')
    ax.set(xlabel='trial', ylabel='objective (infra size + tiebreak x score)',
           title=f'Optimiser learning, fleet = {fleet}')
    ax.legend()
    plt.show()


def plot_thresholds(th):
    fig, ax = plt.subplots(figsize=(7, 4))
    for p, g in th.groupby('param'):
        g = g.sort_values('fleet')
        ax.step(g.fleet, g.min_required.fillna(RANGES[p][1] + 1), where='post', label=p, marker='o', ms=3)
    ax.set(xlabel='fleet size', ylabel='minimum viable count (top = not viable)',
           title='Minimum of each item needed to stay viable')
    ax.legend()
    plt.show()


# ----------------------------------------------------------------------------
# 6b. INDIVIDUAL PARAMETER ANALYSIS (one-at-a-time sweeps)
# ----------------------------------------------------------------------------
SWEEP_FLEETS = [6, 12, 18]                       # fleet sizes to overlay for infra sweeps
MODERATE = dict(pads=2, stands=4, chargers=4)    # held infra when sweeping fleet
CONSTRAINT_NAMES = {'m_completion': 'completion', 'm_queue': 'p95 queue', 'm_util': 'utilisation'}


def sweep_parameter(param):
    """Vary ONE parameter over its full range, holding the others fixed.
    Infra items: others at max (non-limiting), one line per fleet size.
    Fleet: infra held at max and at a moderate level, one line each."""
    lo, hi = RANGES[param]
    if param == 'fleet':
        series = {'infra = max': GENEROUS, 'infra = moderate': MODERATE,
                  'infra = baseline': {p: BASELINE[p] for p in INFRA}}
        cfgs = [(lab, v, {**h, 'fleet': v}) for lab, h in series.items() for v in range(lo, hi + 1)]
    else:
        cfgs = [(f'fleet = {f}', v, {**GENEROUS, 'fleet': f, param: v})
                for f in SWEEP_FLEETS for v in range(lo, hi + 1)]
    rows = []
    for lab, v, cfg in cfgs:
        r = evaluate(cfg, tag=f'sweep_{param}')
        rows.append(dict(r, param=param, series=lab, value=v))
    return pd.DataFrame(rows)


def sweep_summary(sw):
    """Per parameter and series: smallest viable value, the constraint that fails one
    unit below it (the binding constraint), and where the score gain levels off."""
    out = []
    for (param, series), g in sw.groupby(['param', 'series'], sort=False):
        g = g.sort_values('value')
        ok = g[g.viable]
        rec = dict(param=param, series=series, min_viable=None, binding_constraint=None,
                   value_for_95pct_of_gain=None)
        if not ok.empty:
            vmin = int(ok.value.min())
            rec['min_viable'] = vmin
            below = g[g.value == vmin - 1]
            if not below.empty:
                m = below[list(CONSTRAINT_NAMES)].iloc[0]
                rec['binding_constraint'] = CONSTRAINT_NAMES[m.idxmax()]
        gain = g.score.iloc[0] - g.score.min()
        if gain > 1e-9:
            rec['value_for_95pct_of_gain'] = int(g[(g.score.iloc[0] - g.score) >= 0.95 * gain].value.min())
        out.append(rec)
    return pd.DataFrame(out)


def plot_sweep(sw, param):
    """2x3 panel on plain linear axes with units on every label.
    Green band = viable zone. Axes that would blow up are capped (stated in the title);
    a line sitting on the top edge means 'off the scale'."""
    QCAP = 4 * LIMITS['max_p95_queue']          # queue axis cap
    SCAP = 3.0                                  # score axis cap (1 = baseline)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    series = list(dict.fromkeys(sw.series))
    for lab in series:
        g = sw[sw.series == lab].sort_values('value')
        axs[0, 0].plot(g.value, 100 * (1 - g.uncompleted_passengers / DEMAND), marker='o', ms=4, label=lab)
        axs[0, 1].plot(g.value, g.p95_queue_len.clip(upper=QCAP), marker='o', ms=4, label=lab)
        axs[0, 2].plot(g.value, 100 * g.max_utilisation, marker='o', ms=4, label=lab)
        axs[1, 0].plot(g.value, g.score.clip(upper=SCAP), marker='o', ms=4, label=lab)

    # viable zones + limit lines
    lo, hi = RANGES[param]
    axs[0, 0].axhspan(100 * LIMITS['min_completion'], 100, color='tab:green', alpha=0.12)
    axs[0, 0].axhline(100 * LIMITS['min_completion'], c='k', ls='--', lw=1)
    axs[0, 0].set_ylim(0, 102)
    axs[0, 1].axhspan(0, LIMITS['max_p95_queue'], color='tab:green', alpha=0.12)
    axs[0, 1].axhline(LIMITS['max_p95_queue'], c='k', ls='--', lw=1)
    axs[0, 1].set_ylim(0, QCAP * 1.03)
    axs[0, 2].axhspan(0, 100 * LIMITS['max_util'], color='tab:green', alpha=0.12)
    axs[0, 2].axhline(100 * LIMITS['max_util'], c='k', ls='--', lw=1)
    axs[0, 2].set_ylim(0, 105)
    axs[1, 0].axhline(1, c='grey', ls=':', lw=1.2)
    axs[1, 0].text(hi, 1.03, 'baseline = 1', ha='right', fontsize=8, color='grey')
    axs[1, 0].set_ylim(0, SCAP * 1.03)

    meta = [(axs[0, 0], 'Passengers completed', '% of daily demand', f'green = at least {100*LIMITS["min_completion"]:.0f}%'),
            (axs[0, 1], f'p95 queue length (axis capped at {QCAP:g})', 'aircraft in queue', f'green = at most {LIMITS["max_p95_queue"]}'),
            (axs[0, 2], 'Busiest resource utilisation', '% of time busy', f'green = below {100*LIMITS["max_util"]:.0f}%'),
            (axs[1, 0], f'Score (axis capped at {SCAP:g})', '0 = ideal, 1 = baseline', 'lower is better')]
    for ax, title, ylab, note in meta:
        ax.set_title(f'{title}\n({note})', fontsize=10)
        ax.set_ylabel(ylab)
        ax.set_xlabel(f'number of {param}')
    axs[0, 0].legend(fontsize=8, loc='lower right')
    axs[1, 0].legend(fontsize=7, loc='upper left')

    mid = series[len(series) // 2]
    g = sw[sw.series == mid].sort_values('value')
    bottom = np.zeros(len(g))
    for k in W:
        vals = g[f'c_{k}'].values
        axs[1, 1].bar(g.value, vals, bottom=bottom, label=k, width=0.8)
        bottom += vals
    axs[1, 1].set_ylim(0, SCAP * 1.03)
    axs[1, 1].axhline(1, c='grey', ls=':', lw=1.2)
    axs[1, 1].set_title(f'Where the score comes from ({mid})\n(axis capped at {SCAP:g}; bar on top edge = off scale)', fontsize=10)
    axs[1, 1].set_ylabel('score contribution (weight x z)')
    axs[1, 1].set_xlabel(f'number of {param}')
    axs[1, 1].legend(fontsize=7)

    d = g.score.diff().fillna(0).clip(-SCAP, SCAP)
    axs[1, 2].bar(g.value, d, color=np.where(d <= 0, 'tab:green', 'tab:red'))
    axs[1, 2].axhline(0, c='k', lw=0.8)
    lo_, hi_ = min(d.min(), 0), max(d.max(), 0)
    pad = 0.15 * ((hi_ - lo_) or 1)
    axs[1, 2].set_ylim(lo_ - pad, hi_ + pad)       # 0 sits inside the plot, both directions visible
    axs[1, 2].set_title(f'Change in score from adding one more {param} ({mid})\n'
                        f'(green = score improves, red = score worsens; capped at +/-{SCAP:g})', fontsize=10)
    axs[1, 2].set_ylabel('change in score')
    axs[1, 2].set_xlabel(f'number of {param}')

    for ax in axs.flat:
        ax.grid(alpha=0.3)
        ax.margins(x=0.02)
    fig.suptitle(f'Individual effect of {param} (other items held non-limiting)', fontsize=13)
    fig.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# 7. RUN
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    FLEETS = list(range(2, 21, 2))

    joint = []
    for f in FLEETS:
        best = optimise_for_fleet(f, n_trials=60)
        joint.append(dict(fleet=f, **({p: best[p] for p in INFRA} if best else {})))
    joint = pd.DataFrame(joint)
    print('\nJoint minimal viable infrastructure per fleet size:\n', joint.to_string(index=False))

    th = single_param_thresholds(FLEETS)
    print('\nReport sentences:')
    for s in report_sentences(th):
        print(' -', s)

    sweeps = pd.concat([sweep_parameter(p) for p in ['fleet'] + INFRA], ignore_index=True)
    summary = sweep_summary(sweeps)
    print('\nIndividual parameter sweeps (min viable value, binding constraint just below it):')
    print(summary.to_string(index=False))
    sweeps.to_csv('parameter_sweeps.csv', index=False)
    summary.to_csv('parameter_sweep_summary.csv', index=False)
    for p in ['fleet'] + INFRA:
        plot_sweep(sweeps[sweeps.param == p], p)

    log = pd.DataFrame(LOG)
    log['best_viable_so_far'] = log.where(log.viable).groupby('tag').objective.cummin()
    log.to_csv('optimiser_log.csv', index=False)
    joint.to_csv('joint_minimal_config.csv', index=False)
    th.to_csv('single_param_thresholds.csv', index=False)
    plot_convergence(log, FLEETS[len(FLEETS) // 2])
    plot_thresholds(th)
   