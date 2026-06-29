import json
import math
import os
import sys
from simpy.core import Environment
import libsumo
import networkx as nx
from tqdm import trange

import settings
import mintedge
from mintedge.allocation_strategy import MintEDGEAllocationError


def get_nodes_sorted_by_demand_with_distribution(
    infr, peak_demand, demand_weight=0.7, dispersion_factor=1.0
):
    """Sort the nodes combining local demand and betweenness centrality, then
    greedily spread the selection over the map.

    The base score weights two normalized metrics:
    - demand_weight: demand (minimize link transmission)
    - 1 - demand_weight: betweenness (avoid bottlenecks)
    Candidates are then reordered favouring those far from the already
    selected nodes, controlled by dispersion_factor.

    Args:
        infr (Infrastructure): Infrastructure whose nodes are ranked
        peak_demand (Dict[str, Dict[str, int]]): Peak demand matrix
        demand_weight (float): Weight of the demand metric in [0, 1]
        dispersion_factor (float): Exponent applied to the distance to the
            nearest selected node (0 = no dispersion, higher = more spread)

    Returns:
        List[Tuple[BaseStation, float]]: Nodes sorted by score (best first)
    """
    betweenness_weight = 1.0 - demand_weight  # if demand is 0.7, betweenness is 0.3

    # Aggregate the demand of every service per node
    node_demand_dict = {}
    for node in infr.nxgraph.nodes:
        node_demand_dict[node] = sum(
            peak_demand.get(node.name, {}).get(service, 0)
            for service in peak_demand.get(node.name, {})
        )

    # Betweenness is expensive, so compute it only once
    betweenness_scores = nx.betweenness_centrality(infr.nxgraph)

    # Normalize both metrics to the [0, 1] range
    max_demand = max(node_demand_dict.values(), default=1) or 1
    max_betweenness = max(betweenness_scores.values(), default=1) or 1

    weighted_centrality = {}
    for node in infr.nxgraph.nodes:
        norm_demand = node_demand_dict[node] / max_demand
        norm_betweenness = betweenness_scores.get(node, 0) / max_betweenness

        weighted_centrality[node] = (
            norm_demand * demand_weight +
            norm_betweenness * betweenness_weight
        )

    initial_candidates = sorted(
        weighted_centrality.items(), key=lambda x: x[1], reverse=True
    )
    final_sorted = []

    if not initial_candidates:
        return final_sorted

    # The best scored node is always selected first
    first_node, first_score = initial_candidates.pop(0)
    final_sorted.append((first_node, first_score))

    while initial_candidates:
        best_idx = -1
        best_adjusted_score = -1

        for i, (node, base_score) in enumerate(initial_candidates):
            # Distance to the nearest already selected node
            min_dist = min(
                node.location.distance(picked_node.location)
                for picked_node, _ in final_sorted
            )
            min_dist_km = min_dist / 1000.0

            # 0.0 = no dispersion (all together)
            # 0.5 = soft dispersion (looks for a balance)
            # 1.0 = linear dispersion (fairly spread)
            # 2.0 = extreme dispersion (nodes drift to the map corners)
            adjusted_score = base_score * (min_dist_km ** dispersion_factor)

            if adjusted_score > best_adjusted_score:
                best_adjusted_score = adjusted_score
                best_idx = i

        # Append this round's winner to the final list
        winner_node, winner_score = initial_candidates.pop(best_idx)
        final_sorted.append((winner_node, winner_score))

    return final_sorted

def compute_peak_demand(infr, mob_mngr, sim_time):
    """Sweep the whole mobility trace to find the peak demand per base station
    and service.

    Args:
        infr (Infrastructure): Infrastructure with the base stations and services
        mob_mngr (MobilityManager): Mobility manager providing the user steps
        sim_time (int): Simulation horizon in seconds

    Returns:
        Dict[str, Dict[str, int]]: Peak demand for each base station and service
    """
    # Peak demand matrix, initialized to zero
    max_demand = {bs.name: {a.name: 0 for a in infr.services.values()} for bs in infr.bss.values()}
    print("Advancing SUMO to find the real demand peak over the whole day...")

    # Fast-forward through the simulated time
    for t in trange(0, sim_time, settings.ORCHESTRATOR_INTERVAL):
        users = mob_mngr._get_next_step()

        current_demand = {bs.name: {a.name: 0 for a in infr.services.values()} for bs in infr.bss.values()}

        for user_id, loc in users.items():
            # Closest base station to the user
            closest_bs = min(infr.bss.values(), key=lambda b: loc.distance(b.location))

            for serv in infr.services.values():
                # Match the user type with the services assigned in settings
                if user_id.startswith("car") and serv.name in settings.CAR_SERVICES:
                    current_demand[closest_bs.name][serv.name] += serv.arrival_rate

                elif user_id.startswith("person") and serv.name in settings.PEDESTRIAN_SERVICES:
                    current_demand[closest_bs.name][serv.name] += serv.arrival_rate

                elif user_id.startswith("stationary") and serv.name in settings.STATIONARY_SERVICES:
                    current_demand[closest_bs.name][serv.name] += serv.arrival_rate

        # Keep the per (base station, service) peak over the whole simulation
        for bs in infr.bss.values():
            for serv in infr.services.values():
                if current_demand[bs.name][serv.name] > max_demand[bs.name][serv.name]:
                    max_demand[bs.name][serv.name] = current_demand[bs.name][serv.name]

    return max_demand




def check_delay_feasibility(infr, assig_matrix, alloc_matrix):
    """Check that, with the allocated CPU, every (base station, service, server)
    triple meets its delay budget.

    The base algorithm is optimistic because _get_cand_servers computes t_c
    assuming a fully dedicated server (100%). Here t_c is the real one,
    workload / (alloc * max_cap).

    Args:
        infr (Infrastructure): Infrastructure with the placed servers
        assig_matrix (Dict[str, Dict[str, Dict[str, float]]]): Request assignation
        alloc_matrix (Dict[str, Dict[str, float]]): Resource allocation in servers

    Returns:
        Tuple[bool, str]: (True, "") if feasible, (False, reason) otherwise
    """
    strategy = mintedge.AllocationStrategy(infr)

    for dst in infr.bss.values():
        if dst.server is None:
            continue
        for serv in infr.services.values():
            alloc = alloc_matrix[serv.name][dst.name]
            if alloc <= 0:
                continue
            # Real compute time with the CPU fraction actually allocated
            t_c = serv.workload / (alloc * dst.server.max_cap)

            for src_name in infr.bss:
                fraction = assig_matrix[src_name][serv.name][dst.name]
                if fraction <= 0:
                    continue
                src = infr.bss[src_name]
                t_transport = strategy._calculate_transport_delay(src, serv, dst)
                if t_transport + t_c > serv.max_delay:
                    return False, (
                        f"{serv.name}: {src_name}→{dst.name}  "
                        f"delay={( t_transport + t_c)*1000:.2f}ms "
                        f"> budget={serv.max_delay*1000:.0f}ms"
                    )
    return True, ""


def main():
    print("Starting OFFLINE Server Placement computation...")
    # Task id: from SLURM_ARRAY_TASK_ID, or argv[1], or 1 when run standalone
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID",
                                 sys.argv[1] if len(sys.argv) > 1 else 1))
    print(f"task_id = {task_id} (used as seed and in the output file name)")
    sim_time = 3600*24  # Full day to capture the real demand peak
    env = Environment()
    mintedge.SIMULATION_TIME = sim_time

    sim = mintedge.Simulation(sim_time, "dummy.csv", task_id)

    # Initialize mobility
    if settings.RANDOM_ROUTES:
        mob_mngr = mintedge.RandomMobilityManager(env)
    else:
        mob_mngr = mintedge.MobilityManager(env)

    # Build the infrastructure without any pre-placed servers
    original = settings.USE_PEAK_BASED_PLACEMENT
    settings.USE_PEAK_BASED_PLACEMENT = False
    try:
        infr = sim.create_infrastructure(env, mob_mngr)
    finally:
        settings.USE_PEAK_BASED_PLACEMENT = original

    sim.clear_servers(infr)  # Clear in case something was already placed

    peak_demand = compute_peak_demand(infr, mob_mngr, sim_time)

    libsumo.close()

    # Peak-demand based placement
    infr.find_all_paths()
    total_peak_ops = sum(
        peak_demand[bs][serv] * infr.services[serv].workload
        for bs in peak_demand
        for serv in peak_demand[bs]
    )
    print(f"\nTotal peak ops computed: {total_peak_ops}")

    best_server = sim.get_best_server_settings()
    # Lower bound on the number of servers needed by raw capacity
    start_k = max(1, math.ceil(total_peak_ops / best_server["MAX_CAPACITY"]))

    sorted_nodes = [node for node, _ in get_nodes_sorted_by_demand_with_distribution(infr, peak_demand)]

    final_placement = []
    # Incrementally add servers until the allocation is capacity and delay feasible
    for k in range(start_k, len(sorted_nodes) + 1):
        sim.clear_servers(infr)
        for node in sorted_nodes[:k]:
            node.set_edge_server(
                mintedge.EdgeServer(
                    env, node.name, best_server["MAX_CAPACITY"],
                    idle_power=best_server["IDLE_POWER"],
                    max_power=best_server["MAX_POWER"],
                    boot_time=best_server["BOOT_TIME"],
                )
            )
        try:
            _, assig_matrix, alloc_matrix = mintedge.AllocationStrategy(infr).get_allocation(peak_demand)
            feasible, reason = check_delay_feasibility(infr, assig_matrix, alloc_matrix)
            if not feasible:
                print(f"  K={k}: capacity OK but delay infeasible -> {reason}")
                raise MintEDGEAllocationError(reason)
            print(f"Feasible configuration found with {k} servers!")
            final_placement = [node.name for node in sorted_nodes[:k]]
            break
        except MintEDGEAllocationError:
            continue

    if not final_placement:
        print("No feasible placement found. Using all the central nodes.")
        final_placement = [node.name for node in sorted_nodes]

    # Save the results next to this script, where the simulation expects it
    out_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "offline_placement.json")
    with open(out_name, "w") as f:
        json.dump({"servers": final_placement}, f, indent=4)

    print(f"Placement successfully saved to '{out_name}'.")

if __name__ == "__main__":
    main()