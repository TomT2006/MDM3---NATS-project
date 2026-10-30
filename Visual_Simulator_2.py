import importlib.util
import math
import os
import random
import sys
import tkinter as tk
from tkinter import ttk

# ---------------------------------------------------------------------------
# 1. IMPORT BASE MODEL & TIMETABLE GENERATOR
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_NAME = "NATS Base Model.py"
base_model_path = os.path.join(SCRIPT_DIR, BASE_MODEL_NAME)

if not os.path.exists(base_model_path):
  base_model_path = os.path.join(os.getcwd(), BASE_MODEL_NAME)

spec = importlib.util.spec_from_file_location("base_model", base_model_path)
base_model = importlib.util.module_from_spec(spec)
sys.modules["base_model"] = base_model
spec.loader.exec_module(base_model)

import simpy

MAX_ALLOWED_DELAY = 30.0  # Flight is cancelled if departure delay > 30 minutes


# ---------------------------------------------------------------------------
# 2. GUI SIMULATION DASHBOARD
# ---------------------------------------------------------------------------
class eVTOLVisualizer:

  def __init__(
      self,
      root,
      fleet_size=12,
      method="SLB",
      target="minimum",
      max_delay=30.0,
  ):
    self.root = root
    self.root.title("NATS eVTOL Real-Time Fleet Simulator")
    self.root.geometry("1150x780")

    self.fleet_size = fleet_size
    self.method = method
    self.target = target
    self.max_delay = max_delay

    # Simulation control flags
    self.is_running = False
    self.sim_speed = 1.0  # Speed multiplier

    # Vertiport 2D Screen Positions (Triangle Topology)
    self.port_coords = {
        "A": (300, 150),  # Top
        "B": (150, 500),  # Bottom Left
        "C": (450, 500),  # Bottom Right
    }

    # Track live aircraft state for rendering
    self.aircraft_states = {}
    for i in range(fleet_size):
      self.aircraft_states[i] = {
          "id": i,
          "x": self.port_coords["A"][0],
          "y": self.port_coords["A"][1],
          "origin": "A",
          "destination": "A",
          "battery": 100.0,
          "status": "Idle",  # In-Flight, Charging, Queued, Boarding, Cancelled
          "sched_dep": 0.0,
          "eta": 0.0,
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
        control_frame, text="Sim Time: 0.0 min", font=("Helvetica", 11, "bold")
    )
    self.lbl_time.pack(side=tk.RIGHT, padx=15)

    # Main Layout: Canvas (Left) + Live Data Sidebar (Right)
    main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Canvas Frame
    canvas_frame = ttk.Frame(main_pane)
    self.canvas = tk.Canvas(canvas_frame, bg="#1e1e2f", highlightthickness=0)
    self.canvas.pack(fill=tk.BOTH, expand=True)
    main_pane.add(canvas_frame, weight=3)

    # Sidebar Frame (Telemetry Table)
    sidebar_frame = ttk.Frame(main_pane, padding=5)
    main_pane.add(sidebar_frame, weight=2)

    ttk.Label(
        sidebar_frame, text="Live Fleet Telemetry", font=("Helvetica", 12, "bold")
    ).pack(anchor=tk.W, pady=5)

    # Telemetry Treeview
    columns = ("ID", "Status", "Battery", "Dest", "Queue #", "Sched Dep")
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
    resource_type = (
        simpy.Resource if self.method == "FCFS" else simpy.PriorityResource
    )
    self.chargers = {
        p: resource_type(self.env, capacity=base_model.chargers_per_port)
        for p in base_model.ports
    }
    self.results = {
        "flights": 0,
        "cancelled_flights": 0,
        "requests": 0,
        "charging_sessions": 0,
        "waits": [],
        "turnarounds": [],
        "charging_starts": [],
        "departures": [],
        "departure_delays": [],
        "on_time_departures": 0,
        "scheduled_departures_completed": 0,
        "queue_entries": {p: 0 for p in base_model.ports},
        "waits_by_port": {p: [] for p in base_model.ports},
        "flights_by_port": {p: 0 for p in base_model.ports},
        "flights_per_aircraft": {i: 0 for i in range(self.fleet_size)},
    }

    rng = random.Random(42)
    for i in range(self.fleet_size):
      start_port = base_model.ports[i % 3]
      plan = []
      curr = start_port
      t = (i * 7 % 11) / 2.0 + base_model.board_time
      for _ in range(50):
        dest = rng.choice([p for p in base_model.ports if p != curr])
        flight_time, batt = base_model.routes[(curr, dest)]
        plan.append({
            "origin": curr,
            "destination": dest,
            "flight_minutes": flight_time,
            "battery_use": batt,
            "scheduled_departure": t,
        })
        t += (
            flight_time
            + base_model.unload_time
            + base_model.board_time
            + base_model.planned_ground_allowance
        )
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

      req_batt = batt_use + base_model.reserve

      self.aircraft_states[ac_id].update({
          "origin": port,
          "destination": dest,
          "battery": battery,
          "sched_dep": sched_dep,
          "status": "Ground",
          "queue_pos": None,
      })

      # Charge if battery is low
      if battery < req_batt:
        self.aircraft_states[ac_id]["status"] = "Queued"
        charge_min = (
            (100.0 - battery) / base_model.charge_rate
            if self.target == "full"
            else (req_batt - battery) / base_model.charge_rate
        )

        req = (
            self.chargers[port].request()
            if self.method == "FCFS"
            else self.chargers[port].request(priority=sched_dep - self.env.now)
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
            battery += (
                100.0 - battery if self.target == "full" else req_batt - battery
            ) / charge_steps
            self.aircraft_states[ac_id]["battery"] = min(100.0, battery)

      # Hold for scheduled departure
      if self.env.now < sched_dep:
        self.aircraft_states[ac_id]["status"] = "Boarding"
        yield self.env.timeout(sched_dep - self.env.now)

      # Check departure delay and apply >30 min cancellation rule
      departure_delay = max(0, self.env.now - sched_dep)
      if departure_delay > self.max_delay:
        self.aircraft_states[ac_id]["status"] = "Cancelled"
        self.aircraft_states[ac_id]["queue_pos"] = "Cancelled"
        self.results["cancelled_flights"] += 1
        break  # Cancel flight and terminate schedule for this aircraft

      # Flight Phase
      self.aircraft_states[ac_id]["status"] = "In-Flight"
      self.aircraft_states[ac_id]["queue_pos"] = None
      self.aircraft_states[ac_id]["eta"] = self.env.now + flight_min

      flight_steps = 20
      start_x, start_y = self.port_coords[port]
      end_x, end_y = self.port_coords[dest]

      for step in range(1, flight_steps + 1):
        yield self.env.timeout(flight_min / flight_steps)
        progress = step / flight_steps
        self.aircraft_states[ac_id]["x"] = (
            start_x + (end_x - start_x) * progress
        )
        self.aircraft_states[ac_id]["y"] = (
            start_y + (end_y - start_y) * progress
        )
        self.aircraft_states[ac_id]["battery"] = battery - (
            batt_use * progress
        )

      battery -= batt_use
      port = dest
      self.aircraft_states[ac_id]["x"], self.aircraft_states[ac_id]["y"] = (
          self.port_coords[port]
      )

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

    # Draw Flight Paths
    ports = list(self.port_coords.keys())
    for i in range(len(ports)):
      for j in range(i + 1, len(ports)):
        p1, p2 = ports[i], ports[j]
        x1, y1 = self.port_coords[p1]
        x2, y2 = self.port_coords[p2]
        self.canvas.create_line(
            x1, y1, x2, y2, fill="#3a3a5c", width=2, dash=(4, 4)
        )

    # Draw Vertiports
    for name, (x, y) in self.port_coords.items():
      self.canvas.create_oval(
          x - 25, y - 25, x + 25, y + 25, fill="#2b5c8f", outline="#61afef", width=3
      )
      self.canvas.create_text(
          x, y, text=f"Port {name}", fill="white", font=("Helvetica", 10, "bold")
      )

    # Draw Aircraft Icons & Badges
    for ac_id, ac in self.aircraft_states.items():
      x, y = ac["x"], ac["y"]

      if ac["status"] != "In-Flight":
        offset_angle = (ac_id * 30) % 360
        x += 35 * math.cos(math.radians(offset_angle))
        y += 35 * math.sin(math.radians(offset_angle))

      # Color coding based on status
      if ac["status"] == "Cancelled":
        color = "#7f8c8d"  # Grey
      elif ac["status"] == "In-Flight":
        color = "#98c379"  # Green
      elif ac["status"] == "Charging":
        color = "#e5c07b"  # Yellow
      elif ac["status"] == "Queued":
        color = "#e06c75"  # Red
      else:
        color = "#abb2bf"  # Light Grey (Idle/Ground)

      self.canvas.create_oval(
          x - 10, y - 10, x + 10, y + 10, fill=color, outline="white", width=1.5
      )
      self.canvas.create_text(
          x, y, text=str(ac_id), fill="black", font=("Helvetica", 8, "bold")
      )

      # Label text
      if ac["status"] == "Cancelled":
        label_text = f"AC-{ac_id} | CANCELLED"
      else:
        q_info = (
            f" [Q#{ac['queue_pos']}]"
            if ac["queue_pos"] not in (None, "Charging")
            else ""
        )
        label_text = (
            f"AC-{ac_id} | {ac['battery']:.0f}% -> {ac['destination']}{q_info}"
        )

      self.canvas.create_text(
          x, y - 18, text=label_text, fill="white", font=("Helvetica", 8)
      )

    # Draw Canvas Color Legend
    legend_items = [
        ("🟢 In-Flight", "#98c379"),
        ("🟡 Charging", "#e5c07b"),
        ("🔴 Queued", "#e06c75"),
        ("⚪ Idle/Ground", "#abb2bf"),
        ("🔘 Cancelled", "#7f8c8d"),
    ]
    lx, ly = 20, 580
    self.canvas.create_rectangle(
        lx - 10,
        ly - 10,
        lx + 130,
        ly + 115,
        fill="#181825",
        outline="#313244",
        width=1.5,
    )
    self.canvas.create_text(
        lx + 5,
        ly,
        text="Status Legend",
        fill="white",
        font=("Helvetica", 9, "bold"),
        anchor=tk.W,
    )
    for idx, (label, col) in enumerate(legend_items):
      self.canvas.create_oval(
          lx + 5,
          ly + 20 + idx * 18,
          lx + 13,
          ly + 28 + idx * 18,
          fill=col,
          outline="white",
      )
      self.canvas.create_text(
          lx + 20,
          ly + 24 + idx * 18,
          text=label[2:],
          fill="white",
          font=("Helvetica", 8),
          anchor=tk.W,
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
              f"AC-{ac_id}",
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
  app = eVTOLVisualizer(root, fleet_size=12, method="SLB", target="minimum")
  root.mainloop()