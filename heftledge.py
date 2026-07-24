import os
import numpy as np
import json
import math
import sys

from optimizer import OptimizerParams, START, END, sample_params



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

        task_prob = self.compute_task_exec_probabilities(p)

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

        # Inizializzazione della timeline: traccia (start, end, memoria_usata, cpu_usata)
        node_resource_timeline = {n: [] for n in all_nodes}


        for t in ordered_tasks:
            obj = float("inf")
            best_start = 0
            best_end = 0

            for n in all_nodes:
                if t == START:
                    start_time = 0
                    prep_time = p.exectime[(t,n)] + p.init_time[(t,n)]
                else:
                    coord = p.handling_node

                    coord_ready_time = 0      # Quando il coord ha ricevuto tutti i dati remoti
                    max_local_finish = 0      # Quando finisce l'ultimo task predecessore già locale
                    payload_to_send = 0       # Somma dei dati da impacchettare
                    needs_remote_transfer = False

                    for prev in predecessors[t]:
                        prev_node = task_assignment[prev]

                        if prev_node == n:
                            # Stesso nodo: il dato è già in memoria locale. Costo = 0
                            max_local_finish = max(max_local_finish, compl_time[prev])
                        else:
                            # Trasferimento remoto: i dati devono arrivare al coordinatore
                            needs_remote_transfer = True

                            if prev_node == coord:
                                arrive_coord = compl_time[prev]
                            else:
                                arrive_coord = compl_time[prev] + p.output_size[prev] / p.node_bandwidth[(prev_node, coord)] / 10**6 + p.node_latency[(prev_node, coord)]

                            # Il coordinatore deve aspettare il branch più lento
                            coord_ready_time = max(coord_ready_time, arrive_coord)

                            payload_to_send += p.output_size[prev]

                    # Calcola quando il task 't' può effettivamente iniziare su 'n'
                    if not needs_remote_transfer:
                        # Tutti i predecessori erano sullo stesso nodo 'n'.
                        start_time = max_local_finish
                    else:
                        # Il coordinatore deve fare la richiesta di offload impacchettata
                        if n == coord:
                            # Il target è il coordinatore stesso: i dati sono già lì, deve solo
                            # aspettare che l'ultimo dato arrivi e che i suoi task locali finiscano.
                            start_time = max(coord_ready_time, max_local_finish)
                        else:
                            # Il coordinatore invia il pacchetto di dati al nodo remoto 'n'
                            transfer_to_n = payload_to_send / p.node_bandwidth[(coord, n)] / 10**6 + p.node_latency[(coord, n)]

                            # Il task parte quando arriva il pacchetto dal coord e
                            # i task precedenti locali hanno finito di elaborare
                            start_time = max(coord_ready_time + transfer_to_n, max_local_finish)

                    prep_time = p.exectime[(t,n)] + p.init_time[(t,n)]

                end_time = start_time + prep_time

                mem_used_in_interval = 0
                cpu_used_in_interval = 0

                for (sched_start, sched_end, sched_mem, sched_cpu) in node_resource_timeline[n]:
                    # C'è sovrapposizione se la fine del nuovo task non precede l'inizio del vecchio task
                    # e l'inizio del nuovo task non segue la fine del vecchio task
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
                        if _obj < obj:
                            obj = _obj
                            task_assignment[t] = n
                            compl_time[t] = compl_time_on_n
                            # Salva i tempi esatti associati alla scelta migliore
                            best_start = start_time
                            best_end = end_time

            if not t in task_assignment:
                print("UNFEASIBLE")
                raise RuntimeError("Unfeasible solution!")

            chosen_node = task_assignment[t]

            node_resource_timeline[chosen_node].append(
                (best_start, best_end, p.task_memory[t], p.task_cpus[t])
            )

        # Fix node identifiers
        for t in p.T:
            task_assignment[t] = node_key_translator[task_assignment[t]]

        return task_assignment


if __name__ == "__main__":
    params = sample_params()
    assignment = HEFTless().run(params)
    print(assignment)
