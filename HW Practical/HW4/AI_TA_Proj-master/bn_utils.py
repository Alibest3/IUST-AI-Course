#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def mult(cpt1, cpt2):
    cpt = []
    keys1 = cpt1[0].keys()
    keys2 = cpt2[0].keys()
    share_keys = []
    not_share_keys = []
    for key in keys1:
        if key in keys2 and key != 'Prob':
            share_keys.append(key)
        else:
            not_share_keys.append(key)
    for row1 in cpt1:
        for row2 in cpt2:
            flag = 1
            for key in share_keys:
                if row1[key] != row2[key]:
                    flag = 0
            if flag == 0:
                continue
            new_row = row2.copy()
            for key in not_share_keys:
                if key != 'Prob':
                    new_row[key] = row1[key]
                else:
                    new_row['Prob'] = row1['Prob'] * row2['Prob']
            cpt.append(new_row)
    return cpt


# In[ ]:


def find_joint(probability):
    cpt = probability[0]
    for i in range(1, len(probability)):
        cpt = mult(cpt, probability[i])
    return cpt


# In[ ]:


def elimination(joint, node):
    del_cpt = []
    id_tracker = {}
    for row2 in joint:
        row = row2.copy()
        id_string = ""
        for key in row.keys():
            if key == node or key == "Prob":
                continue
            if row[key]:
                id_string += "T"
            else:
                id_string += "F"
        if id_string in id_tracker.keys():
            id_tracker[id_string]['Prob'] += row['Prob']
        else:
            id_tracker[id_string] = row
    for key in id_tracker.keys():
        row = id_tracker[key]
        del row[node]
        del_cpt.append(row.copy())
    return del_cpt


# In[ ]:


def variable_elimination(evidence, query, cpts):
    new_cpts = cpts
    for i in range(len(cpts)):
        if evidence.get(i) or query.get(i):
            continue
        after_joint_prob = []
        joint_prob = []
        for cpt in new_cpts:
            if i in cpt[0].keys():
                joint_prob.append(cpt)
            else:
                after_joint_prob.append(cpt)
        joint = find_joint(joint_prob)
        new_cpts = after_joint_prob
        new_cpts.append(elimination(joint, i))
    final_joint = find_joint(new_cpts)
    sum_prob = 0
    holder = 0
    for joint in final_joint:
        sum_prob += joint['Prob']
        flag = 1
        for node, value in query.items():
            if joint[node] != bool(value):
                flag = 0
                break
        if flag == 1:
            holder = joint['Prob']
    return holder / sum_prob


# In[ ]:


def find_row(cpt, values):
    holder_row = {}
    for row in cpt:
        flag = 1
        for key in row.keys():
            if key != 'Prob' and values[key] != -1 and row[key] != values[key]:
                flag = 0
                break
        if flag == 1:
            holder_row = row.copy()
            break
    return holder_row['Prob']


# In[ ]:


def topological_sort_util(v, visited, stack, graph):
    visited[v] = True
    for i in graph[v]:
        if not visited[i]:
            topological_sort_util(i, visited, stack, graph)

    stack.append(v)

def topological_sort(graph):
    ch_nodes = graph['children_nodes']
    visited = [False] * len(ch_nodes)
    stack = []

    for i in range(len(ch_nodes)):
        if not visited[i]:
            topological_sort_util(i, visited, stack, ch_nodes)
    return stack[::-1]


# In[ ]:


def sample_vertex(node, values, cpts):
    cpt = cpts[node]
    holder_row = {}
    for row in cpt:
        flag = 1
        for key in row.keys():
            if key != 'Prob' and values[key] != -1 and row[key] != values[key]:
                flag = 0
                break
        if flag == 1:
            holder_row = row.copy()
            break
    x = holder_row['Prob']
    if np.random.random() < x:
        return True
    return False


# In[ ]:


def draw_plot(prior, reject, likelihood, gibbs, title):
    X = [1, 2, 3]

    # Plotting both the curves simultaneously
    plt.plot(X, prior, color='r', label='Prior')
    plt.plot(X, reject, color='g', label='Rejection')
    plt.plot(X, likelihood, color='b', label='Likelihood')
    plt.plot(X, gibbs, color='y', label='Gibbs')

    plt.xlabel("#Q")
    plt.ylabel("AE")
    plt.title(title)
    plt.legend()
    # path = OUT_PATH
    # path = os.path.join(path, "out.png")
    # plt.savefig(path)
    plt.show()

