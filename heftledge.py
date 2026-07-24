import os
import numpy as np
import json
import math
import sys

from optimizer import OptimizerParams, START, END, sample_params

EPSILON = 1e-9

class HEFTless:


    def __init__ (self, verbose=False):
        self.verbose = False

    def augment_workflow (self, p: OptimizerParams):
        """
        Add virtual start and end tasks.
        """
        real_start_tasks = set(p.T)
        real_end_tasks = []

        for t in p.T:
            if len(p.adj[t]) == 0:
                real_end_tasks.append(t)
            for dest,_ in p.adj[t]:
                if dest in real_start_tasks:
                    real_start_tasks.remove(dest)

        # Add virtual tasks
        assert(not START in p.T)
        assert(not END in p.T)
        p.T.add(START)
        p.T.add(END)


        p.adj[START] = [(t, 1.0) for t in real_start_tasks]
        for t in real_end_tasks:
            p.adj[t].append((END, 1.0))
        p.adj[END] = []

        p.output_size[START] = p.input_size
        p.output_size[END] = 0
        p.task_memory[START] = 0
        p.task_memory[END] = 0
        p.task_cpus[START] = 0
        p.task_cpus[END] = 0
        for n in p.all_nodes():
            p.exectime[(START, n)] = 0.0
            p.exectime[(END, n)] = 0.0
            p.init_time[(START, n)] = 0.0
            p.init_time[(END, n)] = 0.0

    def compute_task_exec_probabilities (self, p: OptimizerParams):
        q = [START]
        task_prob = {t: 1 for t in p.T}
        visited = set()
        while len(q) > 0:
            t = q.pop()
            visited.add(t)
            for dest,prob_edge in p.adj[t]:
                task_prob[dest] *= task_prob[t] * prob_edge
                if not dest in visited:
                    q.append(dest)
        return task_prob

    def run (self, p: OptimizerParams):


        # Avoid key errors
        for t in p.T:
            if not t in p.adj:
                p.adj[t] = []

        # Remove unknown edges
        for t in p.T:
            adj_list = p.adj[t]
            for x in adj_list:
                dest,_ = x
                if not dest in p.T:
                    print(f"Found {dest} in adjlist of {t}, but {dest} is not in the list of tasks!")
                    adj_list.remove(x)

        # Shorter keys for LP compatibility
        old_params = p
        p, node_key_translator = p.to_shorter_keys()

        self.augment_workflow(p)


        all_nodes = p.all_nodes()

        # ---------------------------------
        wMakespan = p.obj_weights[0]
        wCost = p.obj_weights[2]
        # NOTE: ignoring data transfers and reclaimed memory

        predecessors = {}
        predecessors[START] = []
        for t in p.T:
            for succ, _ in p.adj[t]:
                # t -> succ
                if not succ in predecessors:
                    predecessors[succ] = set()
                predecessors[succ].add(t)


        ordered_tasks = []
        frontier = [START]
        while len(frontier) > 0:
            t = frontier.pop(0)

            ok = True
            for prev in predecessors[t]:
                if not prev in ordered_tasks:
                    ok = False
                    break
            if not ok:
                frontier.append(t)
                continue

            ordered_tasks.append(t)

            for succ, _ in p.adj[t]:
                if not succ in ordered_tasks and not succ in frontier:
                    frontier.append(succ)


        rank = {}

        for t in reversed(ordered_tasks):
            avg_exec = np.mean([p.exectime[(t, n)] for n in all_nodes])

            if len(p.adj[t]) == 0:
                rank[t] = avg_exec
            else:
                rank[t] = avg_exec + max([rank[succ] for succ, _ in p.adj[t]])

        topological_index = {t: i for i, t in enumerate(ordered_tasks)}

        COST_NORMALIZER = 0.0
        MAKESPAN_NORMALIZER = 0.0
        for t in p.T:
            wcet = max([p.exectime[(t,n)] for n in all_nodes])
            wcit = max([p.init_time[(t,n)] for n in all_nodes])
            MAKESPAN_NORMALIZER += wcet + wcit
            COST_NORMALIZER += (wcet + wcit)*max([p.cost[n] for n in all_nodes])


        ordered_tasks = sorted(ordered_tasks, key=lambda x: (rank[x], -topological_index[x]), reverse=True)

        compl_time = {}
        task_assignment = {}

        # Timeline initialization: tracks (start, end, memory_used, cpu_used)
        node_resource_timeline = {n: [] for n in all_nodes}


        for t in ordered_tasks:
            obj = float("inf")
            best_start = 0
            best_end = 0
            best_locality = -1

            for n in all_nodes:
                if t == START:
                    start_time = 0
                    prep_time = p.exectime[(t,n)] + p.init_time[(t,n)]

                    locality_score = 1 if n == p.handling_node else 0
                else:
                    coord = p.handling_node

                    # Check if all predecessor tasks are scheduled on the current node 'n'
                    all_local = True
                    for prev in predecessors[t]:
                        if task_assignment[prev] != n:
                            all_local = False
                            break

                    if all_local:
                        locality_score = 2  # Highest preference: same node
                    elif n == coord:
                        locality_score = 1  # Medium preference: coordinator node
                    else:
                        locality_score = 0  # Low preference: remote worker

                    if all_local:
                        # Case 1: Continuous execution on node 'n'; no offloading.
                        start_time = max([compl_time[prev] for prev in predecessors[t]])
                    else:
                        # Case 2: Serverledge stateless offloading. Data generated on 'n' also
                        # returns to the coordinator and is bundled into the new request.
                        coord_ready_time = 0    # Time at which the coordinator has received all necessary remote data
                        payload_to_send = 0     # Total size of the data to be bundled in the payload

                        for prev in predecessors[t]:
                            prev_node = task_assignment[prev]

                            # Transfer to the coordinator, unless prev_node is the coordinator
                            if prev_node == coord:
                                arrive_coord = compl_time[prev]
                            else:
                                arrive_coord = compl_time[prev] + p.output_size[prev] / p.node_bandwidth[(prev_node, coord)] / 10**6 + p.node_latency[(prev_node, coord)]

                            coord_ready_time = max(coord_ready_time, arrive_coord)
                            payload_to_send += p.output_size[prev]

                        # The coordinator bundles all data into a single offload request
                        if n == coord:
                            start_time = coord_ready_time
                        else:
                            transfer_to_n = payload_to_send / p.node_bandwidth[(coord, n)] / 10**6 + p.node_latency[(coord, n)]
                            start_time = coord_ready_time + transfer_to_n

                    prep_time = p.exectime[(t,n)] + p.init_time[(t,n)]

                end_time = start_time + prep_time

                mem_used_in_interval = 0
                cpu_used_in_interval = 0

                for (sched_start, sched_end, sched_mem, sched_cpu) in node_resource_timeline[n]:
                    # Check for time overlap with the scheduled task.
                    # If they overlap, sum the resources.
                    if not (end_time <= sched_start or start_time >= sched_end):
                        mem_used_in_interval += sched_mem
                        cpu_used_in_interval += sched_cpu

                has_memory = (not n in p.node_available_memory) or (p.node_available_memory[n] - mem_used_in_interval >= p.task_memory[t])
                has_cpu = (not n in p.node_available_cpus) or (p.node_available_cpus[n] - cpu_used_in_interval >= p.task_cpus[t])

                if has_memory and has_cpu: # NOTE: Heftless checks CPU, concurrency, bandwidth and memory
                    task_cost = (p.exectime[(t,n)] + p.init_time[(t,n)]) * p.cost[n]
                    compl_time_on_n = end_time
                    print(f"Completion time of {t} on {n}: {compl_time_on_n}")

                    if compl_time_on_n <= p.deadline:
                        _obj = wCost*task_cost/COST_NORMALIZER + wMakespan*compl_time_on_n/MAKESPAN_NORMALIZER

                        is_better = _obj < (obj - EPSILON)
                        is_tie = abs(_obj - obj) <= EPSILON

                        if is_better or (is_tie and locality_score > best_locality):
                            obj = _obj
                            best_locality = locality_score
                            task_assignment[t] = n
                            compl_time[t] = compl_time_on_n
                            best_start = start_time
                            best_end = end_time

            if not t in task_assignment:
                print("UNFEASIBLE")
                raise RuntimeError("Unfeasible solution!")

            chosen_node = task_assignment[t]

            node_resource_timeline[chosen_node].append((best_start, best_end, p.task_memory[t], p.task_cpus[t]))

        # Fix node identifiers
        for t in p.T:
            task_assignment[t] = node_key_translator[task_assignment[t]]

        return task_assignment


if __name__ == "__main__":
    params = sample_params()
    assignment = HEFTless().run(params)
    print(assignment)
