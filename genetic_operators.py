'''
File contains the genetic operators used in the genetic algorithm.
Author: João Pedro Campos
November 2023
'''

# Import statements
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from IPython.display import Image
from IPython.core.display import HTML
from params import *
from fault_tree import *

class Individual:
    '''
    Class to represent an individual in the population
    '''
    def __init__(self, genes, fitness):
        self.genes = genes
        self.fitness = fitness
        self.rank = None
        self.crowding_distance = None

    def update_fitness(self):
        '''
        Update the fitness of the individual
        '''
        self.fitness = compute_fitness(self)
    
    def copy(self):
        '''
        Copy the individual
        '''
        return Individual(self.genes.copy(), self.fitness.copy())
    
    def __repr__(self):
        return f'Genes: {self.genes}, Fitness: {self.fitness}, Rank: {self.rank}, Crowding Distance: {self.crowding_distance}'
    
    def __lt__(self, other):
        return self.rank < other.rank
    
    def __le__(self, other):
        return self.rank <= other.rank
    
    def __gt__(self, other):
        return self.rank > other.rank
    
    def __ge__(self, other):
        return self.rank >= other.rank
    
    def __eq__(self, other):
        return self.rank == other.rank
    
    def __ne__(self, other):
        return self.rank != other.rank
    
    def __hash__(self):
        return hash((self.genes, self.fitness, self.rank, self.crowding_distance))
    

def crossover(parent1, parent2):
    '''
    Perform crossover between two parents
    '''
    # Choose a random index
    idx = random.randint(0, len(parent1.genes)-1)

    # Create the children
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Perform crossover
    child1.genes = parent1.genes[:idx] + parent2.genes[idx:]
    child2.genes = parent2.genes[:idx] + parent1.genes[idx:]

    # Update the fitness of the children
    child1.update_fitness()
    child2.update_fitness()

    return child1, child2

def mutation(individual):
    '''
    Perform mutation on an individual
    '''
    # Choose a random index
    idx = random.randint(0, len(individual.genes)-1)

    # Choose a random gene, different from the current one
    gene = random.randint(0, 3)
    while gene == individual.genes[idx]:
        gene = random.randint(0, 3)
    
    # Mutate the gene
    individual.genes[idx] = gene

    # Update the fitness of the individual
    individual.update_fitness()

    return individual

def generate_random_individual(num_sensors = 9):
    '''
    Generate a random individual
    '''
    # Generate a random individual
    individual = Individual([random.randint(0, 3) for i in range(num_sensors)], [0, 0])
    individual.update_fitness()

    return individual

def generate_random_population(population_size = 100, num_sensors = 9):
    '''
    Generate a random population
    '''
    # Generate a random population
    population = []
    for i in range(population_size):
        population.append(generate_random_individual(num_sensors))

    return population

def compute_fitness(individual):
    '''
    Compute the fitness of an individual
    '''
    # Load the failure tree
    G = load_failure_tree_cdst()

    # Update the failure tree with the individual's genes
    new_G = allocate_sensors(G, individual.genes)

    # Compute the total entropy
    total_entropy = compute_total_entropy(new_G)
    
    # Compute the total cost
    total_cost = compute_total_cost(new_G)

    # Compute the fitness
    fitness = [total_entropy, total_cost]


    return fitness