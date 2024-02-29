#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[11]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from bn_utils import *
IN_PATH = ".\inputs"
OUT_PATH = ".\outputs"

def exact_inference(query, evidence, cpts, graph):
    new_cpts = []
    parents = graph['parents_nodes']
    for i, cpt in enumerate(cpts):
        tb = []
        for row in cpt:
            if evidence.get(i) and evidence[i] != row[i]:
                continue
            flag = True
            for j in parents[i]:
                if evidence.get(j) and row[j] != evidence[j]:
                    flag = False
            if flag:
                tb.append(row)
        new_cpts.append(tb)
    #two lines below added    
    temp = {names[k] if isinstance(k,str) else k:v for k,v in query.items()}
    query = temp
    return variable_elimination(evidence, query, new_cpts)

def load_model(path):
    
    with open(path, 'r') as file:
        main_graph = dict()
        V = int(file.readline())
        names = dict()
        cpts = list()
        temp_graph = [list() for _ in range(V)]
        main_graph["parents_nodes"] = [list() for _ in range(V)]
        main_graph["children_nodes"] = [list() for _ in range(V)]
        
        for i in range(V):
            data = file.readline()
            data = data.strip('\n')
            data = data.rstrip()
            names[data] = i
            data = file.readline()
            data = data.strip('\n')
            data = data.rstrip()
            try:
                if float(data) or int(data):
                    cpt = list()
                    row = {i: True, 'Prob': float(data)}
                    cpt.append(row)
                    row = {i: False, 'Prob': 1 - float(data)}
                    cpt.append(row)
                    cpts.append(cpt)
            except:
                
                parents = [_ for _ in data.split(" ")]
                
                for parent in parents:
                    temp_graph[i].append(parent)
                cpt = list()
                
                for j in range(2 ** (len(parents))):
                    read_file = file.readline()
                    read_file = read_file.strip('\n')
                    read_file = read_file.rstrip()
                    data = [float(x) for x in read_file.split(" ")]
                    row = {i: True, 'Prob': data[len(data) - 1]}
                    for k in range(len(parents)):
                        row[parents[k]] = bool(data[k])
                    cpt.append(row)
                    new_row = row.copy()
                    new_row[i] = False
                    new_row['Prob'] = 1 - new_row["Prob"]
                    cpt.append(new_row)
                cpts.append(cpt) 
                
        for i in range(V):
            
            for node in temp_graph[i]:
                main_graph["children_nodes"][names[node]].append(i)
                main_graph["parents_nodes"][i].append(names[node])   
                
        new_cpts = list()
        
        for cpt in cpts:
            new_cpt = list()
            
            for row in cpt:
                temp_row = dict()
                
                for key in row.keys():
                    if key in names.keys():
                        temp_row[names[key]] = row[key]
                        
                    else:
                        temp_row[key] = row[key]
                        
                new_cpt.append(temp_row)
            new_cpts.append(new_cpt) 
            
    return main_graph, V, names, cpts, new_cpts

def prior_sample(query, evidences, new_cpt, graph):
    nodes, values = list(), list()
    evidence = [-1 for _ in range(V)]
    
    for key in evidences.keys():
        evidence[names[key]] = bool(evidences[key])
        
    for key in query.keys():
        nodes.append(names[key])
        values.append(query[key])
        
    sorted_vertex = topological_sort(graph)
    samples = list()
    
    for i in range(10000):
        value = [-1 for _ in range(V)]
        
        for vertex in sorted_vertex:
            value[vertex] = sample_vertex(vertex, value, new_cpt)
        flag = 1
        
        for j in range(len(evidence)):
            if evidence[j] != -1 and value[j] != evidence[j]:
                flag = 0
                break
        if flag == 1:
            samples.append(value)
    good_sample = 0
    
    for sample in samples:
        flag = 1
        
        for i in range(len(nodes)):
            if bool(values[i]) != sample[nodes[i]]:
                flag = 0
                break
        if flag == 1:
            good_sample += 1
    return good_sample / len(samples)

def rejection_sample(query, evidences, new_cpt, graph):
    nodes, values = list(), list()
    evidence = [-1 for _ in range(V)]
    
    for key in evidences.keys():
        evidence[names[key]] = bool(evidences[key])
        
    for key in query.keys():
        nodes.append(names[key])
        values.append(query[key])
        
    sort_vertex = topological_sort(graph)
    samples = list()
    
    for i in range(10000):
        value = [-1 for _ in range(V)]
        flag = 1
        
        for vertex in sort_vertex:
            holder = sample_vertex(vertex, value, new_cpt)
            if evidence[vertex] != -1 and holder != evidence[vertex]:
                flag = 0
                break
            value[vertex] = holder
        if flag == 1:
            samples.append(value)
    good_sample = 0
    
    for sample in samples:
        flag = 1
        
        for i in range(len(nodes)):
            if bool(values[i]) != sample[nodes[i]]:
                flag = 0
                break
        if flag == 1:
            good_sample += 1
    return good_sample / len(samples)

def likelihood_sample(query, evidences, new_cpt, graph):
    nodes, values = list(), list()
    evidence = [-1 for _ in range(V)]
    
    for key in evidences.keys():
        evidence[names[key]] = bool(evidences[key])
        
    for key in query.keys():
        nodes.append(names[key])
        values.append(query[key])
        
    sorted_vertex = topological_sort(graph)
    samples = list()
    
    for i in range(10000):
        value = [-1 for _ in range(V)]
        weight = 1
        
        for vertex in sorted_vertex:
            if evidence[vertex] == -1:
                value[vertex] = sample_vertex(vertex, value, new_cpt)
            else:
                value[vertex] = evidence[vertex]
                weight *= find_row(new_cpt[vertex], value)
        samples.append([value, weight])
    good_sample = 0
    samples_sum = 0
    
    for sample in samples:
        flag = 1
        
        for i in range(len(nodes)):
            if bool(values[i]) != sample[0][nodes[i]]:
                flag = 0
                break
        samples_sum += sample[1]
        if flag == 1:
            good_sample += sample[1]
    return good_sample / samples_sum

def gibbs_sample(query, evidences, new_cpt, graph):
    nodes, values = list(), list()
    evidence = [-1 for _ in range(V)]
    
    for key in evidences.keys():
        evidence[names[key]] = bool(evidences[key])
        
    for key in query.keys():
        nodes.append(names[key])
        values.append(query[key])
        
    sorted_vertex = topological_sort(graph)
    samples = list()
    value = [-1 for _ in range(V)]
    
    for i in range(V):
        if evidence[i] != -1:
            value[i] = evidence[i]
        else:
            if np.random.random() < 0.5:
                value[i] = True
            else:
                value[i] = False
                
    for i in range(10000):
        next_value = [-1 for _ in range(V)]
        
        for vertex in sorted_vertex:
            if evidence[vertex] == -1:
                value[vertex] = -1
                value[vertex] = sample_vertex(vertex, value, new_cpt)
                next_value[vertex] = value[vertex]
            else:
                next_value[vertex] = value[vertex]
        samples.append(next_value)
    good_sample = 0
    
    for sample in samples:
        flag = 1
        
        for i in range(len(nodes)):
            if bool(values[i]) != sample[nodes[i]]:
                flag = 0
                break
        if flag == 1:
            good_sample += 1
    return good_sample / len(samples)

def read_queries(path):
    with open(path,'r') as file:
        data = file.readline()
        loaded_data = json.loads(data)
        
    queries = list()
    evidences = list()
    for query in loaded_data:
        queries.append(query[0])
        evidences.append(query[1])
    
    return queries, evidences

graph, V, names, cpts, new_cpts = load_model(IN_PATH + "\model.txt")
queries, evidences = read_queries(IN_PATH + "\queries.txt")

prior_ae_vals = []
rejection_ae_vals = []
likelihood_ae_vals = []
gibbs_ae_vals = []

for i in range(len(queries)):
    exact_val = exact_inference(queries[i], evidences[i], new_cpts, graph)
    prior = prior_sample(queries[i], evidences[i], new_cpts, graph)
    rejection = rejection_sample(queries[i], evidences[i], new_cpts, graph)
    likelihood = likelihood_sample(queries[i], evidences[i], new_cpts, graph)
    gibbs = gibbs_sample(queries[i], evidences[i], new_cpts, graph)
    
    prior_AE = abs(exact_val - prior)
    rejection_AE = abs(exact_val - rejection)
    likelihood_AE = abs(exact_val - likelihood)
    gibbs_AE = abs(exact_val - gibbs)
    prior_ae_vals.append(prior_AE)
    rejection_ae_vals.append(rejection_AE)
    likelihood_ae_vals.append(likelihood_AE)
    gibbs_ae_vals.append(gibbs_AE)
draw_plot(prior_ae_vals, rejection_ae_vals, likelihood_ae_vals, gibbs_ae_vals, "Graph")
   