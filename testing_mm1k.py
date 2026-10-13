import numpy as np
import matplotlib.pyplot as plt
from MM1K_model import mm1k          
mu = 1.0                     
K = 5                          
horizon = 20000                
arrival_rates = np.linspace(0.1, 2.0, 25)

results = [mm1k(arrival_rate=l, service_rate=mu, K=K,
                horizon_min=horizon, warmup_min=horizon // 10) for l in arrival_rates]
throughput = [r["lam_eff_simulated"] for r in results]
blocking = [r["P_block_simulated"] for r in results]

fig, (ax_thr, ax_blk) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_thr.plot(arrival_rates, throughput)
ax_thr.set(xlabel="Arrival rate (aircraft per minute)",
           ylabel="Throughput (accepted aircraft per minute)",
           title=f"Throughput vs arrival rate, K = {K}")

ax_blk.plot(arrival_rates, blocking)
ax_blk.set(xlabel="Arrival rate (aircraft per minute)",
           ylabel="Blocking probability",
           title=f"Blocking probability vs arrival rate, K = {K}")

fig.tight_layout()
plt.show()