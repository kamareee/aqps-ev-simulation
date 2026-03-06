from typing import List, Dict
import numpy as np
from .queue_manager import QueueManager
from .tou_optimization import TOUOptimizer
from .data_structures import Vehicle


class AQPSScheduler:
    def __init__(self, config: Dict):
        """
        Implementation of the Hybrid Two-Layer AQPC Control Strategy.
        Uses BESS and PV as exogenous inputs.
        """
        self.config = config
        self.num_chargers = config["station"]["num_chargers"]
        self.grid_limit = config["grid"]["transformer_limit_kw"]

        # Layer 1: Discrete Event Queue Manager
        self.queue_manager = QueueManager(self.num_chargers)

        # Layer 2: Continuous Convex Optimizer
        self.optimizer = TOUOptimizer(
            transformer_cap=self.grid_limit, dt_hours=config["simulation"]["dt_hours"]
        )

    def step(
        self,
        time_idx: int,
        arriving_vehicles: List[Vehicle],
        departing_vehicle_ids: List[str],
        tou_forecast: np.array,
        pv_forecast: np.array,
        bess_forecast: np.array = None,
    ) -> Dict:
        """
        Executes one control step of the AQPC algorithm.

        Args:
            bess_forecast: Array of BESS power available (fed into algo)
        """

        # Handle default BESS input (0 if not provided)
        if bess_forecast is None:
            bess_forecast = np.zeros_like(pv_forecast)

        # Get instantaneous values for this step
        current_pv = pv_forecast[0] if len(pv_forecast) > 0 else 0.0
        current_bess = bess_forecast[0] if len(bess_forecast) > 0 else 0.0

        # --- Step 1: Update System State ---
        self.queue_manager.process_departures(departing_vehicle_ids)
        self.queue_manager.process_arrivals(arriving_vehicles, time_idx)

        # --- Step 2: Layer 1 - Feasibility & Assignment ---
        # Calculate TOTAL current capacity available for Priority Subsistence
        current_total_capacity = self.grid_limit + current_pv + current_bess

        active_vehicles = self.queue_manager.get_active_assignment(
            current_time=time_idx, current_system_capacity=current_total_capacity
        )

        # Prepare data for Optimizer
        opt_input_evs = []
        for v in active_vehicles:
            opt_input_evs.append(
                {
                    "id": v.id,
                    "energy_needed": v.energy_needed,
                    "max_power": v.max_charging_power_kw,
                    "dep_time_idx": v.departure_time_idx,
                }
            )

        # --- Step 3: Layer 2 - MPC Optimization ---
        horizon = self.config["simulation"]["planning_horizon_steps"]

        results = self.optimizer.solve(
            current_time_idx=time_idx,
            horizon=horizon,
            active_evs=opt_input_evs,
            tou_prices=tou_forecast,
            pv_generation=pv_forecast,
            bess_generation=bess_forecast,
        )

        # --- Step 4: Actuation ---
        power_cmds = {}
        for vid, power_profile in results.schedule.items():
            p_cmd = power_profile[0] if power_profile else 0.0
            power_cmds[vid] = p_cmd
            self.queue_manager.update_vehicle_energy(
                vid, p_cmd * self.config["simulation"]["dt_hours"]
            )

        return {
            "commands": power_cmds,
            "grid_power": results.grid_power[0] if results.grid_power else 0.0,
            "cost": results.total_cost,
            "active_count": len(active_vehicles),
            "queue_size": self.queue_manager.get_queue_size(),
        }
