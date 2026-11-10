import math
import os
import random
import sys
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# 1. IMPORT SAFETY MODEL & CONFIGURATION
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use('TkAgg')

try:
  from model import ScenarioConfig, Simulation, example_config
  from results import measurements
except ImportError as e:
  raise ImportError(
      f'Missing required model files in {os.getcwd()}.\n'
      'Ensure model.py, queues.py, safety.py, results.py, mm1k.py, and'
      ' plots.py are present!'
  ) from e

import simpy

# ---------------------------------------------------------------------------
# 2. GUI SIMULATION DASHBOARD (Driven by Model Config)
# ---------------------------------------------------------------------------
class eVTOLSafetyVisualizer:

  def __init__(self, root):
    self.root = root
    self.root.title("NATS eVTOL Real-Time Fleet Simulator (Safety Model)")
    self.root.geometry("1100x750")

    # Simulation control flags
    self.is_running = False
    self.sim_speed = 1.0

    # Vertiport 2D Screen Positions (4-Port Diamond Topology for A, B, C, D)
    self.port_coords = {
        "A": (300, 150),  # Top
        "B": (150, 420),  # Left
        "C": (450, 420),  # Right
        "D": (300, 600),  # Bottom
    }

    # Initialize model scenario config
    self.config = example_config(
        port_names=("A", "B", "C", "D"),
        fleet=8,
        seats=4,
        stands_per_port=4,
        queue_capacity=4,
        pad_service_mean_min=1,
        default_distance_km=30,
        speed_kmh=200,
        total_demand_per_hour=36,
        timetable_interval_min=30,
        headway_min=3,
        battery_kwh=100,
        charger_kw=120,
        demand_stop_min=600,
        horizon_min=720,
        seed=42,
        holding_kw=30,
        holding_quantile=0.99,
        mission_energy_margin=0.10,
        max_charger_wait_min=60,
    )
    self.config.validate()

    self.fleet_size = self.config.fleet

    # Track live aircraft state for rendering
    self.aircraft_states = {}
    for i in range(self.fleet_size):
      start_port = self.config.port_names[i % len(self.config.port_names)]
      self.aircraft_states[i] = {
          "id": i,
          "x": self.port_coords[start_port][0],
          "y": self.port_coords[start_port][1],
          "origin": start_port,
          "destination": start_port,
          "battery": 100.0,
          "status": "Idle",  # In-Flight, Charging, Queued, Boarding, Idle
          "sched_dep": 0.0,
          "queue_pos": None,
      }

    self._setup_gui()
    self._setup_simulation()

  def _setup_gui(self):
    # Top Control Bar
    control_frame = ttk.Frame(self.root, padding=10)
    control_frame.pack(fill=tk.X)

    self.btn_start = ttk.Button(
        control_frame, text="▶ Play", command=self.toggle_sim
    )
    self.btn_start.pack(side=tk.LEFT, padx=5)

    ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(15, 2))
    self.speed_scale = ttk.Scale(
        control_frame,
        from_=0.5,
        to=10.0,
        value=1.0,
        command=self.update_speed,
    )
    self.speed_scale.pack(side=tk.LEFT, padx=5)

    self.lbl_time = ttk.Label(
        control_frame,
        text="Sim Time: 0.0 min",
        font=("Helvetica", 11, "bold"),
    )
    self.lbl_time.pack(side=tk.RIGHT, padx=15)

    # Main Layout: Canvas (Left) + Live Data Sidebar (Right)
    main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    canvas_frame = ttk.Frame(main_pane)
    self.canvas = tk.Canvas(canvas_frame, bg="#1e1e2f", highlightthickness=0)
    self.canvas.pack(fill=tk.BOTH, expand=True)
    main_pane.add(canvas_frame, weight=3)

    sidebar_frame = ttk.Frame(main_pane, padding=5)
    main_pane.add(sidebar_frame, weight=2)

    ttk.Label(
        sidebar_frame, text="Live Fleet Telemetry", font=("Helvetica", 12, "bold")
    ).pack(anchor=tk.W, pady=5)

    # Telemetry Treeview
    columns = ("ID", "Status", "SOC (%)", "Dest", "Queue #", "Sched Dep")
    self.tree = ttk.Treeview(
        sidebar_frame, columns=columns, show="headings", height=25
    )

    for col in columns:
      self.tree.heading(col, text=col)
      self.tree.column(col, width=65, anchor=tk.CENTER)
    self.tree.column("Status", width=90)

    self.tree.pack(fill=tk.BOTH, expand=True)

  def _setup_simulation(self):
    self.env = simpy.Environment()

    # Create chargers at vertiports
    self.chargers = {
        p: simpy.PriorityResource(
            self.env, capacity=self.config.stands_per_port
        )
        for p in self.config.port_names
    }

    # Generate flight plans across network
    rng = random.Random(self.config.seed)
    flight_dur = (
        60.0 * self.config.default_distance_km
    ) / self.config.speed_kmh
    energy_used = (
        self.config.energy(self.config.default_distance_km) * 100.0
    )  # % SOC used per flight

    for i in range(self.fleet_size):
      start_port = self.config.port_names[i % len(self.config.port_names)]
      plan = []
      curr = start_port
      t = (i * 7 % 11) / 2.0 + 5.0

      for _ in range(40):
        dest = rng.choice(
            [p for p in self.config.port_names if p != curr]
        )
        plan.append({
            "origin": curr,
            "destination": dest,
            "flight_minutes": flight_dur,
            "battery_use": energy_used,
            "scheduled_departure": t,
        })
        t += flight_dur + 15.0 + 5.0
        curr = dest

      ac_data = {
          "id": i,
          "port": start_port,
          "battery": 100.0,
          "flight_plan": plan,
          "start_delay": (i * 7 % 11) / 2.0,
      }
      self.env.process(self._visual_evtol_process(ac_data))

  def _visual_evtol_process(self, aircraft):
    ac_id = aircraft["id"]
    yield self.env.timeout(aircraft["start_delay"])

    port = aircraft["port"]
    battery = aircraft["battery"]

    for journey in aircraft["flight_plan"]:
      dest = journey["destination"]
      flight_min = journey["flight_minutes"]
      batt_use = journey["battery_use"]
      sched_dep = journey["scheduled_departure"]

      req_batt = batt_use + 10.0  # Reserve margin

      self.aircraft_states[ac_id].update({
          "origin": port,
          "destination": dest,
          "battery": battery,
          "sched_dep": sched_dep,
          "status": "Ground",
          "queue_pos": None,
      })

      # Check if charging is required before departure
      if battery < req_batt:
        self.aircraft_states[ac_id]["status"] = "Queued"
        charge_min = (
            (100.0 - battery) / (self.config.charger_kw / self.config.battery_kwh)
        ) / 60.0

        req = self.chargers[port].request(
            priority=sched_dep - self.env.now
        )
        with req as r:
          self.aircraft_states[ac_id]["queue_pos"] = len(
              self.chargers[port].queue
          )
          yield r
          self.aircraft_states[ac_id]["queue_pos"] = "Charging"
          self.aircraft_states[ac_id]["status"] = "Charging"

          charge_steps = 10
          for _ in range(charge_steps):
            yield self.env.timeout(charge_min / charge_steps)
            battery += (100.0 - battery) / charge_steps
            self.aircraft_states[ac_id]["battery"] = min(100.0, battery)

      # Hold for scheduled departure
      if self.env.now < sched_dep:
        self.aircraft_states[ac_id]["status"] = "Boarding"
        yield self.env.timeout(sched_dep - self.env.now)

      # Flight Phase
      self.aircraft_states[ac_id]["status"] = "In-Flight"

      flight_steps = 20
      start_x, start_y = self.port_coords[port]
      end_x, end_y = self.port_coords[dest]

      for step in range(1, flight_steps + 1):
        yield self.env.timeout(flight_min / flight_steps)
        progress = step / flight_steps
        self.aircraft_states[ac_id]["x"] = start_x + (
            end_x - start_x
        ) * progress
        self.aircraft_states[ac_id]["y"] = start_y + (
            end_y - start_y
        ) * progress
        self.aircraft_states[ac_id]["battery"] = battery - (
            batt_use * progress
        )

      battery -= batt_use
      port = dest
      (
          self.aircraft_states[ac_id]["x"],
          self.aircraft_states[ac_id]["y"],
      ) = self.port_coords[port]

  def update_speed(self, val):
    self.sim_speed = float(val)

  def toggle_sim(self):
    self.is_running = not self.is_running
    self.btn_start.config(text="⏸ Pause" if self.is_running else "▶ Play")
    if self.is_running:
      self.run_loop()

  def run_loop(self):
    if not self.is_running:
      return

    try:
      self.env.step()
      self.lbl_time.config(text=f"Sim Time: {self.env.now:.1f} min")
      self.draw_canvas()
      self.update_telemetry()

      delay = int(50 / self.sim_speed)
      self.root.after(delay, self.run_loop)
    except simpy.EmptySchedule:
      self.lbl_time.config(text="Simulation Complete!")
      self.is_running = False

  def draw_canvas(self):
    self.canvas.delete("all")

    # Draw Flight Paths (Lines between Vertiports)
    ports = list(self.port_coords.keys())
    for i in range(len(ports)):
      for j in range(i + 1, len(ports)):
        p1, p2 = ports[i], ports[j]
        x1, y1 = self.port_coords[p1]
        x2, y2 = self.port_coords[p2]
        self.canvas.create_line(
            x1, y1, x2, y2, fill="#3a3a5c", width=2, dash=(4, 4)
        )

    # Draw Vertiport Nodes
    for name, (x, y) in self.port_coords.items():
      self.canvas.create_oval(
          x - 25,
          y - 25,
          x + 25,
          y + 25,
          fill="#2b5c8f",
          outline="#61afef",
          width=3,
      )
      self.canvas.create_text(
          x, y, text=f"Port {name}", fill="white", font=("Helvetica", 10, "bold")
      )

    # Draw Aircraft Icons & Badges
    for ac_id, ac in self.aircraft_states.items():
      x, y = ac["x"], ac["y"]

      if ac["status"] != "In-Flight":
        offset_angle = (ac_id * 45) % 360
        x += 35 * math.cos(math.radians(offset_angle))
        y += 35 * math.sin(math.radians(offset_angle))

      color = (
          "#98c379"
          if ac["status"] == "In-Flight"
          else ("#e5c07b" if ac["status"] == "Charging" else "#e06c75")
      )

      self.canvas.create_oval(
          x - 10, y - 10, x + 10, y + 10, fill=color, outline="white", width=1.5
      )
      self.canvas.create_text(
          x, y, text=str(ac_id + 1), fill="black", font=("Helvetica", 8, "bold")
      )

      q_info = (
          f" [Q#{ac['queue_pos']}]"
          if ac["queue_pos"] not in (None, "Charging")
          else ""
      )
      label_text = (
          f"AC-{ac_id+1} | {ac['battery']:.0f}% -> {ac['destination']}{q_info}"
      )
      self.canvas.create_text(
          x, y - 18, text=label_text, fill="white", font=("Helvetica", 8)
      )

  def update_telemetry(self):
    for row in self.tree.get_children():
      self.tree.delete(row)

    for ac_id, ac in self.aircraft_states.items():
      q_str = str(ac["queue_pos"]) if ac["queue_pos"] is not None else "-"
      self.tree.insert(
          "",
          tk.END,
          values=(
              f"AC-{ac_id+1}",
              ac["status"],
              f"{ac['battery']:.1f}%",
              ac["destination"],
              q_str,
              f"{ac['sched_dep']:.1f}m",
          ),
      )


# ---------------------------------------------------------------------------
# 3. MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
  root = tk.Tk()
  app = eVTOLSafetyVisualizer(root)
  root.mainloop()