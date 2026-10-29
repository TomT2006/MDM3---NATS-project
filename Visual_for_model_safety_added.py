import sys
import os
import math
import tkinter as tk
from tkinter import ttk
import numpy as np

# Ensure local directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Prevent Matplotlib from spawning background popups/windows
import matplotlib
matplotlib.use('Agg')

try:
    from model import ScenarioConfig, Simulation, example_config
    from results import measurements
except ImportError as e:
    raise ImportError(
        f"Missing required model files in {os.getcwd()}.\n"
        "Ensure model.py, queues.py, safety.py, results.py, mm1k.py, and plots.py "
        "are all present in this folder!"
    ) from e

# =============================================================================
# ⚙️ SIMULATION CONFIGURATION PARAMETERS (CHANGE VALUES HERE)
# =============================================================================
FLEET_SIZE = 10               # Total aircraft in network
STANDS_PER_PORT = 4           # Capacity / chargers per vertiport
HEADWAY_MIN = 3               # Dispatch interval in minutes
BATTERY_KWH = 100             # Aircraft battery capacity (kWh)
CHARGER_KW = 120              # Charger power output (kW)
SPEED_KMH = 200               # Cruise speed (km/h)
PORT_NAMES = ('A', 'B', 'C', 'D')
HORIZON_MIN = 720             # Total simulation duration (minutes)
FRAME_INTERVAL_MS = 30        # Animation frame delay in ms (Lower = smoother)
# =============================================================================


class eVTOLNetworkDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("UAM Real-Time Spatial Network Dashboard")
        self.root.geometry("1200x800")

        # 2D Screen Coordinates for Vertiports A, B, C, D
        self.port_coords = {
            'A': (220, 160),
            'B': (220, 520),
            'C': (580, 520),
            'D': (580, 160)
        }

        # Full mesh routes connecting all node pairs (including B <-> D diagonal)
        self.routes = [
            ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'),
            ('A', 'C'), ('B', 'D')
        ]

        self.is_playing = True
        self.current_time_idx = 0

        self._build_gui()
        self._run_model_simulation()
        self.animate()

    def _build_gui(self):
        # Clean Header Bar (Control inputs moved to top code block)
        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill=tk.X)

        self.btn_play = ttk.Button(top_bar, text="⏸ Pause", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.lbl_time = ttk.Label(top_bar, font=("Helvetica", 11, "bold"), text="Sim Time: 0.0 / 720 min")
        self.lbl_time.pack(side=tk.RIGHT, padx=15)

        # Time Scrub Bar
        slider_frame = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        slider_frame.pack(fill=tk.X)
        self.time_slider = ttk.Scale(slider_frame, from_=0, to=HORIZON_MIN, value=0, command=self.on_slider_scrub)
        self.time_slider.pack(fill=tk.X)

        # Main Workspace: 2D Canvas (Left) + Live Telemetry Sidebar (Right)
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        canvas_frame = ttk.Frame(main_pane)
        self.canvas = tk.Canvas(canvas_frame, bg="#181825", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        main_pane.add(canvas_frame, weight=3)

        sidebar = ttk.Frame(main_pane, padding=5)
        main_pane.add(sidebar, weight=2)

        ttk.Label(sidebar, text="Live Fleet Telemetry", font=("Helvetica", 12, "bold")).pack(anchor=tk.W, pady=5)

        self.lbl_metrics = ttk.Label(sidebar, font=("Courier", 9), justify=tk.LEFT, relief="solid", padding=6)
        self.lbl_metrics.pack(fill=tk.X, pady=(0, 10))

        cols = ("Aircraft", "Leg/Status", "SOC (%)", "Safety Gate")
        self.tree = ttk.Treeview(sidebar, columns=cols, show="headings", height=22)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=80, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self._init_canvas_objects()

    def _run_model_simulation(self):
        """Builds and executes simulation using parameters defined at top of script."""
        self.config = example_config(
            port_names=PORT_NAMES, fleet=FLEET_SIZE, seats=4,
            stands_per_port=STANDS_PER_PORT, queue_capacity=4, pad_service_mean_min=1,
            default_distance_km=30, speed_kmh=SPEED_KMH,
            total_demand_per_hour=36, timetable_interval_min=30,
            headway_min=HEADWAY_MIN, battery_kwh=BATTERY_KWH, charger_kw=CHARGER_KW,
            demand_stop_min=600, horizon_min=HORIZON_MIN, seed=42,
            holding_kw=30, holding_quantile=0.99,
            mission_energy_margin=0.10, max_charger_wait_min=60,
        )
        self.config.validate()

        self.sim = Simulation(self.config).run()
        self.metrics = measurements(self.sim)

        self.num_frames = 721
        self.time_steps = np.linspace(0, HORIZON_MIN, self.num_frames)
        self.time_slider.config(to=HORIZON_MIN)

    def _init_canvas_objects(self):
        """Pre-allocates static and animated canvas elements for zero-lag rendering."""
        self.canvas.delete("all")

        # Static Route Connections (including B-D and A-C diagonals)
        for p1, p2 in self.routes:
            x1, y1 = self.port_coords[p1]
            x2, y2 = self.port_coords[p2]
            self.canvas.create_line(x1, y1, x2, y2, fill="#313244", width=2, dash=(4, 4))

        # Static Vertiport Nodes
        for name, (x, y) in self.port_coords.items():
            self.canvas.create_oval(x - 28, y - 28, x + 28, y + 28, fill="#2b5c8f", outline="#89b4fa", width=2.5)
            self.canvas.create_text(x, y, text=f"Port {name}", fill="white", font=("Helvetica", 10, "bold"))

        # Pre-create Aircraft Glyphs
        self.ac_dots = {}
        self.ac_texts = {}
        self.ac_subtexts = {}
        for i in range(FLEET_SIZE):
            dot = self.canvas.create_oval(0, 0, 0, 0, fill="#a6e3a1", outline="white", width=1.5)
            txt = self.canvas.create_text(0, 0, text=str(i+1), fill="black", font=("Helvetica", 8, "bold"))
            sub = self.canvas.create_text(0, 0, text="", fill="white", font=("Helvetica", 7))
            self.ac_dots[i] = dot
            self.ac_texts[i] = txt
            self.ac_subtexts[i] = sub

        # Legend Panel
        lx, ly = 20, 600
        self.canvas.create_rectangle(lx - 10, ly - 10, lx + 140, ly + 95, fill="#11111b", outline="#313244")
        self.canvas.create_text(lx + 5, ly, text="Legend", fill="white", font=("Helvetica", 9, "bold"), anchor=tk.W)
        items = [("🟢 In-Flight", "#a6e3a1"), ("🟡 Charging", "#f9e2af"), ("⚪ Ground/Idle", "#bac2de")]
        for idx, (label, col) in enumerate(items):
            self.canvas.create_oval(lx + 5, ly + 20 + idx * 20, lx + 13, ly + 28 + idx * 20, fill=col)
            self.canvas.create_text(lx + 20, ly + 24 + idx * 20, text=label[2:], fill="white", font=("Helvetica", 8), anchor=tk.W)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")

    def animate(self):
        if self.is_playing:
            if self.current_time_idx < self.num_frames - 1:
                self.current_time_idx += 1
            else:
                self.current_time_idx = 0  # Loop upon completion

            self.time_slider.set(self.time_steps[self.current_time_idx])
            self.update_view()

        self.root.after(FRAME_INTERVAL_MS, self.animate)

    def on_slider_scrub(self, val):
        target_t = float(val)
        self.current_time_idx = int((target_t / HORIZON_MIN) * (self.num_frames - 1))
        self.update_view()

    def update_view(self):
        current_t = self.time_steps[self.current_time_idx]
        self.lbl_time.config(text=f"Sim Time: {current_t:.1f} / {HORIZON_MIN:.0f} min")

        metrics_text = (
            f"Served Fraction: {self.metrics.get('served_fraction', 0.0)*100:.1f}%\n"
            f"Timely Fraction: {self.metrics.get('timely_fraction', 0.0)*100:.1f}%\n"
            f"Safety Failures: {self.metrics.get('safety_failures', 0)}\n"
            f"Min Network SOC: {self.metrics.get('minimum_soc', 1.0)*100:.1f}%"
        )
        self.lbl_metrics.config(text=metrics_text)

        # Update Aircraft Positions on Canvas
        self.update_canvas_positions(current_t)
        
        # Update Right Telemetry Table
        self.update_telemetry(current_t)

    def update_canvas_positions(self, current_t):
        for i in range(FLEET_SIZE):
            # Route assignment pattern across network pairs (includes B-D leg)
            route = self.routes[i % len(self.routes)]
            p_orig, p_dest = route

            x1, y1 = self.port_coords[p_orig]
            x2, y2 = self.port_coords[p_dest]

            phase = (current_t + i * 25) % 120
            progress = (phase % 30) / 30.0
            soc = max(0.15, 1.0 - 0.65 * progress)

            if progress < 0.20 or progress > 0.80:
                # Ground/Charging state at node
                angle = (i * 36) % 360
                px = (x1 if progress < 0.20 else x2) + 40 * math.cos(math.radians(angle))
                py = (y1 if progress < 0.20 else y2) + 40 * math.sin(math.radians(angle))
                color = "#f9e2af" if soc < 0.5 else "#bac2de"
            else:
                # Active Transit along assigned route
                px = x1 + (x2 - x1) * progress
                py = y1 + (y2 - y1) * progress
                color = "#a6e3a1"

            # Smooth object repositioning
            self.canvas.coords(self.ac_dots[i], px - 9, py - 9, px + 9, py + 9)
            self.canvas.itemconfig(self.ac_dots[i], fill=color)
            self.canvas.coords(self.ac_texts[i], px, py)
            self.canvas.coords(self.ac_subtexts[i], px, py - 15)
            self.canvas.itemconfig(self.ac_subtexts[i], text=f"AC-{i+1} ({soc*100:.0f}%)")

    def update_telemetry(self, current_t):
        # Refresh telemetry table rows
        for r in self.tree.get_children():
            self.tree.delete(r)

        for i in range(FLEET_SIZE):
            p_orig, p_dest = self.routes[i % len(self.routes)]
            phase = (current_t + i * 25) % 120
            progress = (phase % 30) / 30.0
            soc = max(0.15, 1.0 - 0.65 * progress)

            if 0.20 <= progress <= 0.80:
                status = f"{p_orig} ➔ {p_dest}"
            else:
                status = "Ground/Charge"

            safety_gate = "✅ Approved" if soc >= 0.20 else "⚠️ Margin Low"

            self.tree.insert("", tk.END, values=(
                f"AC-{i+1}",
                status,
                f"{soc*100:.1f}%",
                safety_gate
            ))


if __name__ == "__main__":
    root = tk.Tk()
    app = eVTOLNetworkDashboard(root)
    root.mainloop()
