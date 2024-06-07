import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from IPython.display import Image
from IPython.core.display import HTML

def load_sensors_data():
    '''
    There are three types of sensors
    each sensor has 2 features: the reliability and the cost
    sensor 0: no sensor is allocated
    '''
    sensor_dict = {}
    sensor_dict[0] = [0, 0]
    sensor_dict[1] = [1.0, 25]
    sensor_dict[2] = [0.90, 5]
    sensor_dict[3] = [0.85, 2]

    return sensor_dict


def load_cdst_system_data():

    subsystem2idx = {}
    subsystem2idx['CSH'] = 0
    subsystem2idx['CC'] = 1
    subsystem2idx['C'] = 2
    subsystem2idx['SH'] = 3
    subsystem2idx['BR'] = 4
    subsystem2idx['OL'] = 5
    subsystem2idx['CY'] = 6
    subsystem2idx['HM'] = 7
    subsystem2idx['SYS'] = 8

    idx2subsystem = {}
    idx2subsystem[0] = 'CSH'
    idx2subsystem[1] = 'CC'
    idx2subsystem[2] = 'C'
    idx2subsystem[3] = 'SH'
    idx2subsystem[4] = 'BR'
    idx2subsystem[5] = 'OL'
    idx2subsystem[6] = 'CY'
    idx2subsystem[7] = 'HM'
    idx2subsystem[8] = 'SYS'

    return subsystem2idx, idx2subsystem


class Subsystem:
    def __init__(self, name, failure_prob, allocated_sensor):
        self.name = name
        self.failure_prob = failure_prob
        self.allocated_sensor = allocated_sensor
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f'{self.name}'
    

def load_failure_tree():

    # Initialize subsystems
    HM = Subsystem('HM', 0.334, 0)
    O = Subsystem('O', 0.18, 0)
    EHIT = Subsystem('EHIT', 0.188, 0)
    C = Subsystem('C', 0.195, 0)
    L = Subsystem('L', 0.383, 0)
    CY = Subsystem('CY', 0.361, 0)
    ECS = Subsystem('ECS', 0.148, 0)
    PQ = Subsystem('PQ', 0.25, 0)
    S = Subsystem('S', 0.127, 0)
    HMH = Subsystem('HMH', 0.187, 0)
    PQH = Subsystem('PQH', 0.05, 0)
    OEH = Subsystem('OEH', 0.029, 0)
    IHI = Subsystem('IHI', 0.046, 0)
    LC = Subsystem('LC', 0.062, 0)
    SW = Subsystem('SW', 0.21, 0)
    GR = Subsystem('GR', 0.26, 0)
    FR = Subsystem('FR', 0.236, 0)
    LCC = Subsystem('LCC', 0.09, 0)
    IT = Subsystem('IT', 0.086, 0)
    BU = Subsystem('BU', 0.06, 0)
    #ELVTR = Subsystem('ELVTR', 0.9, 0)

    # Add children
    ECS.add_child(CY)
    PQ.add_child(CY)
    O.add_child(HM)
    EHIT.add_child(HM)
    PQH.add_child(HMH)
    IHI.add_child(HMH)
    OEH.add_child(HMH)
    LC.add_child(HMH)
    LCC.add_child(FR)
    IT.add_child(FR)
    BU.add_child(FR)
    #CY.add_child(ELVTR)
    #HMH.add_child(ELVTR)
    #HM.add_child(ELVTR)
    #FR.add_child(ELVTR)
    #C.add_child(ELVTR)
    #L.add_child(ELVTR)
    #S.add_child(ELVTR)
    #SW.add_child(ELVTR)
    #GR.add_child(ELVTR)

    # Create graph
    G = nx.DiGraph()
    for subsystem in [O, L, S, PQ, HM, CY, C, ECS, EHIT, HMH, PQH, OEH, IHI, LC, SW, GR, FR, LCC, IT, BU]:
        G.add_node(subsystem.name, failure_prob=subsystem.failure_prob, allocated_sensor=subsystem.allocated_sensor)
        for child in subsystem.children:
            G.add_edge(subsystem.name, child.name)

    return G

def load_failure_tree_cdst():

    # Initialize subsystems
    CSH = Subsystem('CSH', 0.075, 0)
    CC = Subsystem('CC', 0.042, 0)
    C = Subsystem('C', 0.26, 0)
    SH = Subsystem('SH', 0.075, 0)
    BR = Subsystem('BR', 0.06, 0)
    OL = Subsystem('OL', 0.16, 0)
    CY = Subsystem('CY', 0.11385, 0)
    HM = Subsystem('HM', 0.2942, 0)
    SYS = Subsystem('SYS', 0.660, 0)

    # Add children
    CSH.add_child(CY)
    CC.add_child(CY)
    C.add_child(SYS)
    SH.add_child(HM)
    BR.add_child(HM)
    OL.add_child(HM)
    HM.add_child(SYS)
    CY.add_child(SYS)

    # Create graph
    G = nx.DiGraph()
    for subsystem in [CSH, CC, C, SH, BR, OL, CY, HM, SYS]:
        G.add_node(subsystem.name, failure_prob=subsystem.failure_prob, allocated_sensor=subsystem.allocated_sensor)
        for child in subsystem.children:
            G.add_edge(subsystem.name, child.name)

    return G


def load_elvtr_system_data():
    
    subsystem2idx = {}
    subsystem2idx['O'] = 0
    subsystem2idx['EHIT'] = 1
    subsystem2idx['C'] = 2
    subsystem2idx['L'] = 3
    subsystem2idx['ECS'] = 4
    subsystem2idx['PQ'] = 5
    subsystem2idx['S'] = 6
    subsystem2idx['PQH'] = 7
    subsystem2idx['OEH'] = 8
    subsystem2idx['IHI'] = 9
    subsystem2idx['LC'] = 10
    subsystem2idx['SW'] = 11
    subsystem2idx['GR'] = 12
    subsystem2idx['LCC'] = 13
    subsystem2idx['IT'] = 14
    subsystem2idx['BU'] = 15
    subsystem2idx['CY'] = 16
    subsystem2idx['HMH'] = 17
    subsystem2idx['HM'] = 18
    subsystem2idx['FR'] = 19
    #subsystem2idx['ELVTR'] = 20

    idx2subsystem = {}
    idx2subsystem[0] = 'O'
    idx2subsystem[1] = 'EHIT'
    idx2subsystem[2] = 'C'
    idx2subsystem[3] = 'L'
    idx2subsystem[4] = 'ECS'
    idx2subsystem[5] = 'PQ'
    idx2subsystem[6] = 'S'
    idx2subsystem[7] = 'PQH'
    idx2subsystem[8] = 'OEH'
    idx2subsystem[9] = 'IHI'
    idx2subsystem[10] = 'LC'
    idx2subsystem[11] = 'SW'
    idx2subsystem[12] = 'GR'
    idx2subsystem[13] = 'LCC'
    idx2subsystem[14] = 'IT'
    idx2subsystem[15] = 'BU'
    idx2subsystem[16] = 'CY'
    idx2subsystem[17] = 'HMH'
    idx2subsystem[18] = 'HM'
    idx2subsystem[19] = 'FR'
    #idx2subsystem[20] = 'ELVTR'

    return subsystem2idx, idx2subsystem

    