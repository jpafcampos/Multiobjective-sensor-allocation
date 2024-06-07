import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from IPython.display import Image
from IPython.core.display import HTML
from params import *

def compute_total_entropy(G):
    '''
    Compute the total entropy of the system
    '''
    total_entropy = 0
    for node in G.nodes:
        failure_prob = G.nodes[node]['failure_prob']
        if failure_prob != 0 and failure_prob != 1:
            total_entropy += failure_prob * np.log(failure_prob) + (1-failure_prob) * np.log(1-failure_prob)
        else:
            total_entropy += 0

    return -total_entropy
    
def compute_entropy(G, node):
    '''
    Compute the entropy of a node
    '''
    failure_prob = G.nodes[node]['failure_prob']
    entropy = failure_prob * np.log(failure_prob) + (1-failure_prob) * np.log(1-failure_prob)

    return -entropy

def compute_total_cost(G):
    '''
    Compute the total cost of the system
    '''
    sensors_data = load_sensors_data()
    total_cost = 0
    for node in G.nodes:
        allocated_sensor = G.nodes[node]['allocated_sensor']
        total_cost += sensors_data[allocated_sensor][1]

    return total_cost

def update_nodes_probabilities(G, sensors, use_bayes = True):
    '''

    '''
    sensors_data = load_sensors_data()
    subsystem2idx, idx2subsystem  = load_cdst_system_data()

    new_G = G.copy()  # create a new graph to avoid altering the original attributes in G

    if use_bayes:
        for i in range(len(sensors)):
            node = idx2subsystem[i]
            allocated_sensor = sensors[i]
            if allocated_sensor != 0:
                R = sensors_data[allocated_sensor][0]
                R_bar = 1 - R
                F_prior = new_G.nodes[node]['failure_prob']
                F_bar_prior = 1 - F_prior
                new_failure_prob = (R_bar * F_prior) / (R_bar * F_prior + R * F_bar_prior)
                new_G.nodes[node]['failure_prob'] = new_failure_prob
    
    else:
        for i in range(len(sensors)):
            node = idx2subsystem[i]
            allocated_sensor = sensors[i]
            if allocated_sensor != 0:
                new_failure_prob = 1 - sensors_data[allocated_sensor][0]
                new_G.nodes[node]['failure_prob'] = new_failure_prob

    # Update probabilities of the descendants, which are not monitored by sensors (OR gates)
    for node in new_G.nodes:
        if len(list(new_G.predecessors(node))) > 0:
            parents = list(new_G.predecessors(node))
            failure_probs = [new_G.nodes[parent]['failure_prob'] for parent in parents]
            new_G.nodes[node]['failure_prob'] = np.sum(failure_probs) - np.prod(failure_probs)

    return new_G

def total_entropy_with_sensors(G, sensors):
    '''
    Compute the total entropy of the system, given the sensors
    '''
    new_G = update_nodes_probabilities(G, sensors)
    #print(new_G.nodes.data())
    total_entropy = compute_total_entropy(new_G)

    return total_entropy

def allocate_sensors(G, sensors):
    '''
    Allocate sensors to the nodes of the graph
    '''
    new_G = G.copy()
    # Update probabilities
    new_G = update_nodes_probabilities(G, sensors)
    subsystem2idx, idx2subsystem  = load_cdst_system_data()
    for i in range (len(sensors)):
        node = idx2subsystem[i]
        allocated_sensor = sensors[i]
        new_G.nodes[node]['allocated_sensor'] = allocated_sensor
    return new_G


def draw_solution(sensors, name):
    '''
    Draw the solution
    '''
    # Load the failure tree
    G = load_failure_tree_cdst()

    sensors_data = load_sensors_data()

    # Update the failure tree with the individual's genes
    new_G = allocate_sensors(G, sensors)

    # Draw the graph
    pos = graphviz_layout(new_G, prog='dot')

    # Assign a color to each node, depending on the allocated sensor
    colors = []
    for node in new_G.nodes:
        allocated_sensor = new_G.nodes[node]['allocated_sensor']
        if allocated_sensor == 0:
            colors.append('silver')
        elif allocated_sensor == 1:
            colors.append('lawngreen')
        elif allocated_sensor == 2:
            colors.append('cyan')
        elif allocated_sensor == 3:
            colors.append('lightyellow')
    
    # Draw the graph
    
    plt.figure()
    nx.draw(new_G, pos, with_labels=False, arrows=True, node_color=colors, node_size=900)

    # Add node probabilities as labels
    labels = {}
    for node in new_G.nodes:
        probability = new_G.nodes[node]['failure_prob']
        labels[node] = f'{node}\nP: {probability:.2f}'

    nx.draw_networkx_labels(new_G, pos, labels=labels, font_size=12)
    # add a legend for colors on the left
    plt.plot([],[], color='silver', label='No sensor')
    plt.plot([],[], color='lawngreen', label='Sensor 1' + '; (R: ' + str(sensors_data[1][0]) + '; C: ' + str(sensors_data[1][1]) + ')' )
    plt.plot([],[], color='cyan', label='Sensor 2' + '; (R: ' + str(sensors_data[2][0]) + '; C: ' + str(sensors_data[2][1]) + ')' )
    plt.plot([],[], color='lightyellow', label='Sensor 3' + '; (R: ' + str(sensors_data[3][0]) + '; C: ' + str(sensors_data[3][1]) + ')' )

    # add a second legend for entropy and cost on the right
    total_entropy = compute_total_entropy(new_G)
    total_cost = compute_total_cost(new_G)
    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
    plt.plot([],[], color='white', label=f'Total entropy: {total_entropy:.2f}')
    plt.plot([],[], color='white', label=f'Total cost: {total_cost:.2f}')
    plt.legend(prop={'size': 10})
    plt.savefig('images/' + name + '.png')
