"""
CarbonEdge Standalone Module
============================
A standalone implementation of Algorithm 1 from the CarbonEdge paper:
"CarbonEdge: Leveraging Mesoscale Spatial Carbon-Intensity Variations for Low Carbon Edge Computing"

This module implements the carbon-aware incremental placement algorithm for edge computing
without requiring Kubernetes or other infrastructure dependencies.

Algorithm 1 Overview:
1. Compute application-server latency
2. Filter servers exceeding latency constraints (retain feasible candidates)
3. Retrieve server telemetry (capacity, base power, power state) and carbon intensity
4. Solve optimization to minimize carbon emissions
5. Commit resource allocation and power state transitions

The optimization minimizes:
- Application operational emissions: sum(x_ij * E_ij * I_j)
- Server activation emissions: sum((y_j - y_j_curr) * B_j * I_j)

Subject to constraints:
- Resource constraint: demands <= capacity
- Latency constraint: latency <= limit
- Placement constraint: each app assigned to exactly one server
- Power state consistency: active servers cannot be powered off
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s: %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class GeoLocation:
    """Geographic coordinates for latency calculation."""
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'GeoLocation') -> float:
        """Calculate distance in kilometers using Haversine formula."""
        R = 6371  # Earth radius in km
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


@dataclass
class CartesianLocation:
    """
    Cartesian location in km, matching the environ.py approach.
    Uses uniform random placement in a square area.
    """
    x: float  # km
    y: float  # km
    
    def distance_to(self, other: 'CartesianLocation') -> float:
        """Calculate Euclidean distance in km."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Application:
    """Represents an edge application to be placed."""
    id: str
    location: int  # Node the application where the app is connecting
    latency_limit_ms: float  # Maximum acceptable latency in milliseconds
    resource_demands: dict[str, float]  # Resource type -> demand (e.g., {"cpu": 2, "memory": 4096})
    energy_consumption: float  # Total energy consumption in kWh when running (over entire job duration, not per-timeslot. CarbonEdge considered only 1 time slot
    duration: int # Job duration in time slots


@dataclass
class Server:
    """Represents an edge server in a data center."""
    id: str
    location: CartesianLocation
    carbon_intensity: float  # gCO2/kWh from the local grid
    base_power: float  # Base power consumption in kW when idle
    capacity: dict[str, float]  # Resource type -> available capacity
    is_active: bool = False  # Current power state


@dataclass
class PlacementResult:
    """Result of the placement algorithm."""
    placements: dict[str, str]  # app_id -> server_id
    server_power_states: dict[str, bool]  # server_id -> is_active
    total_carbon_emissions: float  # Total carbon emissions in gCO2
    success: bool
    message: str = ""


# =============================================================================
# CarbonEdge Algorithm 1 Implementation
# =============================================================================

class CarbonEdgeHandler:
    def __init__(self, env, exp_info, ts_data):
        self.environment    = env
        self.ts_data        = ts_data
        self.POWER_SCALE    = exp_info.get("POWER_SCALE", 0.1)           # Adjust based on power units
        self.TS_H           = float(exp_info.get("TS_H", 1/60))                 # Duration of time slot in hours
        self.MAX_CARBON_INT = exp_info.get("MAX_CARBON_INTENSITY", 600)  # gCO2/kWh for 0 green power
        self.MIN_CARBON_INT = exp_info.get("MIN_CARBON_INTENSITY", 20)   # gCO2/kWh for max green power

        self.prop_matrix_delay = self.environment.matrix_propagation_delay

        self.servers = self._transform_servers_to_CE_servers()
        self.apps    = self._transform_jobs_to_CE_applications()
        self.engine  = CarbonEdgePlacement(self.servers, self.prop_matrix_delay)

        self.hasOptimizationBeenDone = False
        self.placements = {}
        self.carbon_emissions = 0.0


    def _transform_servers_to_CE_servers(self) -> list[Server]:
        """
        Transform GreenRL servers to CarbonEdge servers.
        """
        # Base coordinates for mapping 
        base_lat, base_lon = 0.0, 0.0
        scenario_km_side = getattr(self.environment, 'SIZE_KM_SETTING_SIDE', 50)
        
        # Get power parameters
        alpha = getattr(self.environment, 'alpha', 0.5)
        gamma = getattr(self.environment, 'gamma', 0.1)
        
        # --- Convert Servers ---
        servers = []
        max_green = np.max(self.ts_data.greenPW) if np.max(self.ts_data.greenPW) > 0 else 1
        
        # Get current state from running jobs
        current_power = self.environment.getPowerConsumption()  # Power from running jobs per server
        
        for n in range(self.environment.N_SERVERS):
            # Get server location
            if hasattr(self.environment, 'serverLocations') and len(self.environment.serverLocations) > n:
                server_loc = self.environment.serverLocations[n]
                lat = base_lat + (server_loc[1] / scenario_km_side) * 0.5
                lon = base_lon + (server_loc[0] / scenario_km_side) * 0.5
            else:
                # Default grid layout
                lat = base_lat + (n // 2) * 0.1
                lon = base_lon + (n % 2) * 0.1
            
            # Convert green power to carbon intensity (inverse relationship)
            green_level = self.ts_data.greenPW[n] if n < len(self.ts_data.greenPW) else 0
            carbon_intensity = self.MAX_CARBON_INT - \
                (green_level / max_green) * (self.MAX_CARBON_INT - self.MIN_CARBON_INT)
            
            # Get REMAINING capacity (accounting for existing jobs)
            max_pc = self.environment.MAX_PC_SERVERS[n] if hasattr(self.environment, 'MAX_PC_SERVERS') else 16
            used_pc = sum(self.environment.Node_List[n]) if self.environment.Node_List[n] else 0
            remaining_pc = max_pc - used_pc
            max_storage = self.environment.MAX_STORAGE_SERVER[n] if hasattr(self.environment, 'MAX_STORAGE_SERVER') else 1000
            
            servers.append(Server(
                id=f"server-{n}",
                location=CartesianLocation(lat, lon),
                carbon_intensity=carbon_intensity,
                base_power=(gamma + current_power[n]) * self.POWER_SCALE,  # kW, includes running jobs
                capacity={
                    "cpu": remaining_pc,  # Remaining capacity, not max
                    "memory": max_storage
                },
                is_active=(used_pc > 0)  # True if jobs are running on this server
            ))

        return servers

    def _transform_jobs_to_CE_applications(self) -> list[Application]:
        """
        Transform GreenRL jobs to CarbonEdge applications.
        """
        # --- Convert Jobs to Applications ---
        # Get power parameters
        alpha = getattr(self.environment, 'alpha', 0.5)
        gamma = getattr(self.environment, 'gamma', 0.1)

        # Calculate max one-way delay from environment parameters
        deadline_delay     = getattr(self.environment, 'DEADLINE_DELAY', 33.33)
        transmission_delay = getattr(self.environment, 'TRASMISSION_DELAY', 1.0)
        computation_delay  = getattr(self.environment, 'COMPUTATION_DELAY', 5.0)
        max_1way_delay     = (deadline_delay - 2 * transmission_delay - computation_delay) / 2
        
        applications = []
        c_jn = getattr(self.environment, 'c_jn', 1.0)  # Processing cycles per job
        job_storage = getattr(self.environment, 'JOB_STORAGE', 100)
        
        for j in range(len(self.ts_data.jobs_pc)):
            # Get job parameters
            arrival_node = self.ts_data.jobs_arrivNd[j] if j < len(self.ts_data.jobs_arrivNd) else 0
            wireless_delay = self.ts_data.jobs_wdlay[j] if j < len(self.ts_data.jobs_wdlay) else 0
            
            # Calculate total energy consumption over entire job duration
            # P = α·c_j + γ, then E = P * time * job_duration
            power_normalized   = alpha * c_jn + gamma
            job_duration_ts    = self.ts_data.jobs_duration[j] if j < len(self.ts_data.jobs_duration) else 1
            job_duration_hours = job_duration_ts * self.TS_H
            energy_kwh         = power_normalized * self.POWER_SCALE * job_duration_hours    # Total energy over job duration
            
            # Adjust latency limit based on wireless delay
            adjusted_latency = max(max_1way_delay - wireless_delay, 1.0)
            
            applications.append(Application(
                id=f"job-{j}",
                location=arrival_node,  # Use node index for carbonedge.py (not GeoLocation)
                latency_limit_ms=adjusted_latency,
                resource_demands={
                    "cpu": c_jn,
                    "memory": job_storage
                },
                energy_consumption=energy_kwh,
                duration=job_duration_ts
            ))

        return applications
                
    def optimize(self) -> tuple[dict[str, int], float]:
        """
        Run CarbonEdge placement for a batch of applications.
        
        Returns:
            placements: Dict mapping app_id -> server_index (0-based) or -1 if rejected
            total_carbon: Total carbon emissions for this placement
        """
        if self.hasOptimizationBeenDone:
            logger.warning("CarbonEdge optimization already done, returning cached results")
        elif not self.apps:
            logger.info("No jobs to run this time slot. Skipping CarbonEdge optimization.")
        else:        
            self.result = self.engine.place_applications(self.apps)
            
            for app in self.apps:
                if app.id in self.result.placements and self.result.placements[app.id] is not None:
                    # Find server index from server id
                    server_id = self.result.placements[app.id]
                    for idx, server in enumerate(self.servers):
                        if server.id == server_id:
                            self.placements[app.id] = idx
                            break
                else:
                    self.placements[app.id] = -1  # Rejected
            
            self.carbon_emissions = self.result.total_carbon_emissions
        
        return None

    def get_carbon_emissions(self):
        return self.carbon_emissions
    
    def get_placements(self):
        return self.placements
    
    def return_job_action(self, job_index):
        app_id = f"job-{job_index}"

        if app_id in self.placements and self.placements[app_id] >= 0:
            # CarbonEdge placed this job - use server index + 1 (1-indexed action)
            action = self.placements[app_id] + 1
        else:
            # CarbonEdge rejected this job
            action = 0
        return action

    @staticmethod
    def carbonedge_retrieveDecision(carbonedge_handler, job_index):
        return carbonedge_handler.return_job_action(job_index)

# =============================================================================
# CarbonEdge Algorithm 1 Implementation
# =============================================================================

class CarbonEdgePlacement:
    """
    Implements Algorithm 1: CarbonEdge Incremental Placement.
    
    This class solves the carbon-aware placement optimization problem
    to minimize carbon emissions while meeting latency and resource constraints.
    """
    
    def __init__(self, servers: list[Server], prop_matrix_delay: np.ndarray):
        """
        Initialize with available servers.
        
        Args:
            servers: List of edge servers available for placement
        """
        self.servers = {s.id: s for s in servers}
        self._server_list = servers
        self.propagation_matrix = prop_matrix_delay

    def place_applications(
        self,
        applications: list[Application],
        carbon_weight: float = 1.0,
        energy_weight: float = 0.0,
    ) -> PlacementResult:
        """
        Execute Algorithm 1: Carbon-aware incremental placement.
        
        Args:
            applications: Batch of applications to place
            carbon_weight: Weight for carbon emissions in objective (default 1.0)
            energy_weight: Weight for energy usage in objective (default 0.0)
            
        Returns:
            PlacementResult with optimal placements and power states
        """
        if not applications:
            return PlacementResult(
                placements={},
                server_power_states={s.id: s.is_active for s in self._server_list},
                total_carbon_emissions=0.0,
                success=True,
                message="No applications to place"
            )
        
        n_apps = len(applications)
        n_servers = len(self._server_list)
        
        # Step 1-8: Compute latency and filter servers for each application
        # L[i][j] = latency from app i to server j
        # feasible[i][j] = True if server j can serve app i within latency limit
        latency_matrix  = np.zeros((n_apps, n_servers))
        feasible_matrix = np.zeros((n_apps, n_servers), dtype=bool)
        
        for i, app in enumerate(applications):
            for j, server in enumerate(self._server_list):
                latency               = self.propagation_matrix[app.location, j]
                latency_matrix[i, j]  = latency
                feasible_matrix[i, j] = latency <= app.latency_limit_ms
        
        # Store apps with no feasible server (adaptation not in CARBONEDGE)
        apps_feasible     = []
        apps_infeasible   = []
        apps_feasible_idx = []
        for i, app in enumerate(applications):
            if  feasible_matrix[i].any():
                apps_feasible.append(app)
                apps_feasible_idx.append(i)
            else:
                apps_infeasible.append(app)
            
        feasible_matrix = feasible_matrix[apps_feasible_idx,:]
        
        # Step 9: Retrieve server telemetry and carbon intensity
        # Already available in self._server_list
        
        # Step 10: Solve optimization problem (Equation 7)
        result = self._solve_optimization(
            apps_feasible,
            feasible_matrix,
            carbon_weight,
            energy_weight
        )
        
        if result is None:
            return PlacementResult(
                placements={},
                server_power_states={s.id: s.is_active for s in self._server_list},
                total_carbon_emissions=0.0,
                success=False,
                message="Optimization failed to find a feasible solution"
            )
        
        placements, power_states, total_emissions = result
        
        # add that placement of unfeasible apps is none.
        for app in apps_infeasible:
            placements[app.id] = None

        # Step 11: Update server states
        self._commit_placements(placements, power_states, apps_feasible)
        
        return PlacementResult(
            placements=placements,
            server_power_states=power_states,
            total_carbon_emissions=total_emissions,
            success=True,
            message=f"Successfully placed {len(apps_feasible)} applications"
        )
    
    def _solve_optimization(
        self,
        applications: list[Application],
        feasible_matrix: np.ndarray,
        carbon_weight: float,
        energy_weight: float,
    ) -> Optional[tuple[dict[str, str], dict[str, bool], float]]:
        """
        Solve the carbon-aware placement optimization problem.
        
        Decision variables:
        - x[i,j] ∈ {0,1}: placement of app i on server j
        - y[j] ∈ {0,1}: power state of server j - server is on or off
        
        Objective: Minimize carbon emissions
        - Application operational: sum(x_ij * E_ij * I_j)
        - Server activation: sum((y_j - y_j_curr) * B_j * I_j)
        
        Constraints:
        1. Resource: sum(x_ij * j^k) <= y_j * C_j^k for all j, k
        2. Latency: x_ij = 0 if L_ij > l_i for all i,j (enforced via feasible_matrix)
        3. Placement: sum(x_ij) = 1 for all i
        4. Power consistency: y_j >= y_j_curr for all j (can't power off active)
        5. Assignment requires power: x_ij <= y_j for all i, j
        """
        n_apps    = len(applications)
        n_servers = len(self._server_list)
        
        # Variable layout: [x_00, x_01, ..., x_0m, x_10, ..., x_nm, y_0, y_1, ..., y_m]
        # Total variables: n_apps * n_servers + n_servers
        n_x = n_apps * n_servers
        n_y = n_servers
        n_vars = n_x + n_y
        
        # Build objective function coefficients
        # Minimize: sum(x_ij * E_ij *  I_j) + sum((y_j - y_j_curr) * B_j * I_j)
        c = np.zeros(n_vars)
        
        for i, app in enumerate(applications):
            for j, server in enumerate(self._server_list):
                idx = i * n_servers + j
                # Application operational carbon = energy * carbon_intensity
                app_carbon = app.energy_consumption * server.carbon_intensity
                c[idx] = carbon_weight * app_carbon + energy_weight * app.energy_consumption
        
        
        # Server activation carbon (only for newly activated servers). 
        # Subtract the constant term for currently active servers (handled in objective offset)
        constant_offset = 0.0

        for j, server in enumerate(self._server_list):
            idx = n_x + j            
            activation_carbon = server.base_power * server.carbon_intensity
            c[idx] = carbon_weight * activation_carbon + energy_weight * server.base_power
            if server.is_active:
                constant_offset -= c[idx]

        # Build constraints
        constraints = []
        
        # Constraint 1: Resource constraints
        # For each server j and resource type k: sum_i(x_ij * R_ik) <= C_jk
        # y_j is not necessary because Constraint 5 already sets that if y_j = 0->x_ij=0
        resource_types = set()
        for app in applications:
            resource_types.update(app.resource_demands.keys())
        
        for j, server in enumerate(self._server_list):
            for resource in resource_types:
                capacity = server.capacity.get(resource, 0)
                A_row = np.zeros(n_vars)
                for i, app in enumerate(applications):
                    demand = app.resource_demands.get(resource, 0)
                    idx = i * n_servers + j
                    A_row[idx] = demand
                
                constraints.append(LinearConstraint(A_row, lb=-np.inf, ub=capacity))
        
        # Constraint 2: Latency constraints (enforced by setting infeasible x to 0)
        # This is handled via bounds
        
        # Constraint 3: Placement constraints
        # For each app i: sum_j(x_ij) = 1
        for i in range(n_apps):
            A_row = np.zeros(n_vars)
            for j in range(n_servers):
                idx = i * n_servers + j
                A_row[idx] = 1
            constraints.append(LinearConstraint(A_row, lb=1, ub=1))
        
        # Constraint 4: Power consistency (can't power off active servers)
        # y_j >= y_j_curr, handled via bounds
        
        # Constraint 5: Assignment requires power
        # x_ij <= y_j for all i, j
        # Equivalently: x_ij - y_j <= 0
        for i in range(n_apps):
            for j in range(n_servers):
                A_row = np.zeros(n_vars)
                x_idx = i * n_servers + j
                y_idx = n_x + j
                A_row[x_idx] = 1
                A_row[y_idx] = -1
                constraints.append(LinearConstraint(A_row, lb=-np.inf, ub=0))
        
        # Variable bounds
        lb = np.zeros(n_vars)
        ub = np.ones(n_vars)
        
        # Enforce latency constraints by setting upper bound to 0 for infeasible pairs (Constraint 2)
        for i in range(n_apps):
            for j in range(n_servers):
                if not feasible_matrix[i, j]:
                    idx = i * n_servers + j
                    ub[idx] = 0
        
        # Power consistency: y_j >= y_j_curr
        for j, server in enumerate(self._server_list):
            if server.is_active:
                lb[n_x + j] = 1
        
        bounds = Bounds(lb, ub)
        
        # Define integrality (all binary)
        integrality = np.ones(n_vars, dtype=int)
        
        # Solve MILP
        try:
            result = milp(
                c=c,
                constraints=constraints,
                bounds=bounds,
                integrality=integrality,
            )
            
            if not result.success:
                return None
            
            x_sol = result.x
            
            # Extract placements
            placements = {}
            for i, app in enumerate(applications):
                for j, server in enumerate(self._server_list):
                    idx = i * n_servers + j
                    if x_sol[idx] > 0.5:  # Binary rounding
                        placements[app.id] = server.id
                        break
            
            # Extract power states
            power_states = {}
            for j, server in enumerate(self._server_list):
                y_idx = n_x + j
                power_states[server.id] = x_sol[y_idx] > 0.5
            
            # Calculate total emissions
            total_emissions = result.fun + constant_offset
            
            return placements, power_states, total_emissions
            
        except Exception as e:
            logger.exception("Optimization error: %s", e)
            return None
    
    def _commit_placements(
        self,
        placements: dict[str, str],
        power_states: dict[str, bool],
        applications: list[Application]
    ):
        """
        Commit placements and update server states.
        Updates server capacity and power states for the next iteration.
        """
        # Update power states
        for server_id, is_active in power_states.items():
            self.servers[server_id].is_active = is_active
        
        # Update server capacities
        app_dict = {app.id: app for app in applications}
        for app_id, server_id in placements.items():
            app = app_dict[app_id]
            if server_id is None:
                continue
            else:
                server = self.servers[server_id]
                for resource, demand in app.resource_demands.items():
                    if resource in server.capacity:
                        server.capacity[resource] -= demand


if __name__ == "__main__":
    print("what")
