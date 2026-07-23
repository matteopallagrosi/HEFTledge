import os
import json
import math
import sys
import pulp as pl
import pickle

START="_start"
END="_end"

class OptimizerParams:

    def __init__ (self):
        self.edge_nodes = set()
        self.cloud_nodes = set()
        self.T = set()
        self.adj = {}
        self.exectime = {}
        self.init_time = {}
        self.deadline = 0
        self.obj_weights = [0.3, 0.3, 0.3, 0.1]
        self.cost = {}
        self.output_size = {}
        self.node_available_memory = {}
        self.node_free_memory = {}
        self.node_available_cpus = {}
        self.task_memory = {}
        self.task_cpus = {}
        self.input_size = 10
        self.handling_node = None
        self.node_labels = {}
        self.task_labels = {}
        self.ds_bandwidth = {}
        self.ds_latency = {}
        self.node_latency = {}
        self.node_bandwidth = {}

    def all_nodes (self):
        return self.edge_nodes.union(self.cloud_nodes)

    def _encode_tuple_dict(self, d):
        return {json.dumps(k): v for k, v in d.items()}

    def _decode_tuple_dict(self, d):
        return {tuple(json.loads(k)): v for k, v in d.items()}

    def _encode_set_values(self, d):
        return {k: list(v) if isinstance(v, set) else v for k, v in d.items()}

    def _decode_set_values(self, d):
        return {k: set(v) if isinstance(v, list) else v for k, v in d.items()}

    def to_json(self):
        return json.dumps({
            "cloud_nodes": list(self.cloud_nodes),
            "edge_nodes": list(self.edge_nodes),
            "T": list(self.T),
            "adj": self.adj,
            "exectime": self._encode_tuple_dict(self.exectime),
            "init_time": self._encode_tuple_dict(self.init_time),
            "deadline": self.deadline,
            "cost": self.cost,
            "obj_weights": self.obj_weights,
            "output_size": self.output_size,
            "node_available_memory": self.node_available_memory,
            "node_free_memory": self.node_free_memory,
            "node_available_cpus": self.node_available_cpus,
            "task_memory": self.task_memory,
            "task_cpus": self.task_cpus,
            "input_size": self.input_size,
            "handling_node": self.handling_node,
            "node_labels": self._encode_set_values(self.node_labels),
            "task_labels": self._encode_set_values(self.task_labels),
            "ds_bandwidth": self.ds_bandwidth,
            "ds_latency": self.ds_latency,
            "node_latency": self._encode_tuple_dict(self.node_latency),
            "node_bandwidth": self._encode_tuple_dict(self.node_bandwidth),
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        obj = cls()
        obj.cloud_nodes = set(data["cloud_nodes"])
        obj.edge_nodes = set(data["edge_nodes"])
        obj.T = set(data["T"])
        obj.adj = data["adj"]
        for k,v in obj.adj.items():
            newlist = []
            for entry in v:
                if isinstance(entry, list):
                    pair = tuple(entry)
                else:
                    pair = tuple(json.loads(entry))
                assert(len(pair) == 2)
                casted_pair = (pair[0], float(pair[1]))
                newlist.append(casted_pair)
            obj.adj[k] = newlist
        obj.exectime = obj._decode_tuple_dict(data["exectime"])
        obj.init_time = obj._decode_tuple_dict(data["init_time"])
        obj.deadline = data["deadline"]
        obj.cost = data["cost"]
        obj.obj_weights = data["obj_weights"]
        assert(len(obj.obj_weights) == 4)
        obj.output_size = data["output_size"]
        obj.node_available_memory = data["node_available_memory"]
        obj.node_free_memory = data["node_free_memory"]
        obj.node_available_cpus = data["node_available_cpus"]
        obj.task_memory = data["task_memory"]
        obj.task_cpus = data["task_cpus"]
        obj.input_size = data["input_size"]
        obj.handling_node = data["handling_node"]
        obj.node_labels = obj._decode_set_values(data["node_labels"])
        obj.task_labels = obj._decode_set_values(data["task_labels"])
        obj.ds_bandwidth = data["ds_bandwidth"]
        obj.ds_latency = data["ds_latency"]
        obj.node_latency = obj._decode_tuple_dict(data["node_latency"])
        obj.node_bandwidth = obj._decode_tuple_dict(data.get("node_bandwidth", {}))
        return obj

    def __str__(self):
        return (
            f"OptimizerParams(\n"
            f"  edge_nodes={self.edge_nodes},\n"
            f"  cloud_nodes={self.cloud_nodes},\n"
            f"  T={self.T},\n"
            f"  adj={self.adj},\n"
            f"  exectime={self.exectime},\n"
            f"  init_time={self.init_time},\n"
            f"  deadline={self.deadline},\n"
            f"  cost={self.cost},\n"
            f"  obj_weights={self.obj_weights},\n"
            f"  output_size={self.output_size},\n"
            f"  node_available_memory={self.node_available_memory},\n"
            f"  node_free_memory={self.node_free_memory},\n"
            f"  node_available_cpus={self.node_available_cpus},\n"
            f"  task_memory={self.task_memory},\n"
            f"  task_cpus={self.task_cpus},\n"
            f"  input_size={self.input_size},\n"
            f"  handling_node={self.handling_node},\n"
            f"  node_labels={self.node_labels},\n"
            f"  task_labels={self.task_labels},\n"
            f"  ds_bandwidth={self.ds_bandwidth},\n"
            f"  ds_latency={self.ds_latency},\n"
            f"  node_latency={self.node_latency}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()

    def to_shorter_keys (self):
        translator = {}
        back_translator = {}

        p = OptimizerParams()
        p.T = self.T
        p.adj = self.adj
        p.deadline = self.deadline
        p.obj_weights = self.obj_weights
        p.output_size = self.output_size
        p.task_memory = self.task_memory
        p.task_cpus = self.task_cpus
        p.input_size = self.input_size
        p.task_labels = self.task_labels

        p.edge_nodes = set()
        for i,n in enumerate(self.edge_nodes):
            key = f"e{i}"
            p.edge_nodes.add(key)
            translator[n] = key
            back_translator[key] = n
        p.cloud_nodes = set()
        for i,n in enumerate(self.cloud_nodes):
            key = f"c{i}"
            p.cloud_nodes.add(key)
            translator[n] = key
            back_translator[key] = n

        p.exectime = {}
        for k,v in self.exectime.items():
            t,n = k
            p.exectime[(t, translator[n])] = v

        p.init_time = {}
        for k,v in self.init_time.items():
            t,n = k
            p.init_time[(t, translator[n])] = v

        p.node_free_memory = {}
        for k,v in self.node_free_memory.items():
            p.node_free_memory[translator[k]] = v
        p.node_available_memory = {}
        for k,v in self.node_available_memory.items():
            p.node_available_memory[translator[k]] = v
        p.node_available_cpus = {}
        for k,v in self.node_available_cpus.items():
            p.node_available_cpus[translator[k]] = v

        p.cost = {}
        for k,v in self.cost.items():
            p.cost[translator[k]] = v

        p.handling_node = translator[self.handling_node]

        p.node_labels = {}
        for k,v in self.node_labels.items():
            p.node_labels[translator[k]] = v

        p.ds_bandwidth = {}
        for k,v in self.ds_bandwidth.items():
            p.ds_bandwidth[translator[k]] = v
        p.ds_latency = {}
        for k,v in self.ds_latency.items():
            p.ds_latency[translator[k]] = v

        p.node_latency = {}
        for k,v in self.node_latency.items():
            n1,n2 = k
            p.node_latency[(translator[n1], translator[n2])] = v

        p.node_bandwidth = {}
        for k,v in self.node_bandwidth.items():
            n1,n2 = k
            p.node_bandwidth[(translator[n1], translator[n2])] = v

        return p, back_translator



# (A) -0.5-- (B) -- (C)
#     \-0.5- (D)
def sample_params () -> OptimizerParams:
    params = OptimizerParams()
    params.edge_nodes = set(["edge2", "edge1"])
    params.cloud_nodes = set(["cloud"])
    params.T = ["A", "B", "C", "D"]
    params.adj = {t: [] for t in params.T}
    params.adj["A"] = [("B", 0.5), ("D", 0.5)]
    params.adj["B"] = [("C", 1)]
    params.exectime = {(t,n): 1 for t in params.T for n in params.all_nodes()}
    params.init_time = {(t,n): 0.1 for t in params.T for n in params.all_nodes()}
    params.output_size = {t: 0.5 for t in params.T}
    params.deadline = 30
    params.task_memory = {t: 512 for t in params.T}
    params.task_cpus = {t: 1 for t in params.T}
    params.node_free_memory = {n: 756 for n in params.edge_nodes}
    params.node_available_memory = {n: 756 for n in params.edge_nodes}
    params.node_available_cpus = {n: 4 for n in params.edge_nodes}
    params.cost = {n: 0.001 for n in params.edge_nodes}
    params.cost["cloud"] = 0.01
    params.task_labels = {t: set() for t in params.T}
    params.node_labels = {n: set() for n in params.all_nodes()}


    params.task_labels["B"].add("GPU")
    params.task_labels["D"].add("EDGE")
    params.node_labels["cloud"].add("GPU")
    params.node_labels["edge1"].add("EDGE")
    params.node_labels["edge2"].add("EDGE")

    params.handling_node = "edge1"

    for n1 in params.all_nodes():
        params.ds_bandwidth[n1] = 1000*1000
        params.ds_latency[n1] = 0.050

        for n2 in params.all_nodes():
            if n1 == n2:
                params.node_latency[(n1,n2)] = 0
            else:
                params.node_latency[(n1,n2)] = 0.050

    json_str = params.to_json()
    with open("sample.json", "w") as of:
        of.write(json_str)

    new_params = OptimizerParams.from_json(json_str)
    return new_params
