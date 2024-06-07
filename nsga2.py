'''
This file implements the NSGA-II algorithm.
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
from genetic_operators import *

def fast_non_dominated_sort(population):
    '''
    Perform fast non-dominated sort on the population
    '''
    # Initialize the ranks
    ranks = {}
    ranks[1] = []

    # Initialize the domination count
    domination_count = {}
    domination_count[0] = 0

    # Initialize the domination set
    domination_set = {}
    domination_set[0] = []

    # Initialize the front
    front = {}
    front[1] = []

    # Perform domination check
    for p in range(len(population)):
        domination_count[p] = 0
        domination_set[p] = []
        for q in range(len(population)):
            if population[p].fitness[0] < population[q].fitness[0] and population[p].fitness[1] < population[q].fitness[1]:
                if q not in domination_set[p]:
                    domination_set[p].append(q)
            elif population[p].fitness[0] > population[q].fitness[0] and population[p].fitness[1] > population[q].fitness[1]:
                domination_count[p] += 1

        if domination_count[p] == 0:
            population[p].rank = 1
            front[1].append(p)

    # Initialize the front counter
    i = 1

    # Update the fronts
    while front[i] != []:
        temp = []
        for p in front[i]:
            for q in domination_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    population[q].rank = i + 1
                    temp.append(q)
        i += 1
        front[i] = temp

    return population

def crowding_distance_sort(population):
    '''
    Perform crowding distance sort on the population
    '''
    # Initialize the crowding distance
    crowding_distance = {}
    for p in range(len(population)):
        crowding_distance[p] = 0

    # Compute the crowding distance
    for m in range(2):
        sorted_population = sorted(population, key=lambda x: x.fitness[m])
        crowding_distance[0] = float('inf')
        sorted_population[0].crowding_distance = crowding_distance[0]
        crowding_distance[len(population) - 1] = float('inf')
        sorted_population[-1].crowding_distance = crowding_distance[len(population) - 1]
        for i in range(1, len(population) - 1):
            crowding_distance[i] += (sorted_population[i+1].fitness[m] - sorted_population[i-1].fitness[m]) / (sorted_population[-1].fitness[m] - sorted_population[0].fitness[m])
            sorted_population[i].crowding_distance = crowding_distance[i]

    return population

def nsga2_algorithm(population_size = 100, num_generations = 100, num_sensors = 9):
    '''
    Perform the NSGA-II algorithm
    '''
    # Generate a random population
    population = []
    for i in range(population_size):
        population.append(generate_random_individual(num_sensors))

    # Perform NSGA-II
    for i in range(num_generations):
        print('Generation: {}'.format(i))
        # Perform fast non-dominated sort
        population = fast_non_dominated_sort(population)

        # Perform crowding distance sort
        population = crowding_distance_sort(population)

        # Create the mating pool
        mating_pool = []
        for j in range(int(population_size/2)):
            a1 = random.randint(0, population_size-1)
            a2 = random.randint(0, population_size-1)
            if population[a1].rank < population[a2].rank:
                mating_pool.append(population[a1])
            elif population[a1].rank > population[a2].rank:
                mating_pool.append(population[a2])
            else:
                #print(population)
                #print('----------------')
                if population[a1].crowding_distance > population[a2].crowding_distance:
                    mating_pool.append(population[a1])
                else:
                    mating_pool.append(population[a2])

        # Perform crossover
        offspring = []
        for j in range(int(population_size/2)):
            p1 = mating_pool[j]
            p2 = mating_pool[len(mating_pool) - j - 1]
            c1, c2 = crossover(p1, p2)
            offspring.append(c1)
            offspring.append(c2)

        # Perform mutation
        for j in range(len(offspring)):
            offspring[j] = mutation(offspring[j])

        # Create the new population
        population = mating_pool + offspring

    # Perform fast non-dominated sort
    population = fast_non_dominated_sort(population)

    # Perform crowding distance sort
    population = crowding_distance_sort(population)

    return population


def plot_pareto_front(population, filter = True):
    '''
    Plot the pareto front
    '''
    # Create the dataframe
    df = pd.DataFrame(columns=['fitness1', 'fitness2', 'rank'])
    for individual in population:
        df = df.append({'fitness1': individual.fitness[0], 'fitness2': individual.fitness[1], 'rank': individual.rank}, ignore_index=True)

    # Filter the pareto front
    name = ''
    if filter:
        df = df[df['rank'] == 1]
        name = '_filtered'

    # Plot the pareto front
    plt.figure(figsize=(13, 8))
    # Increase font size
    plt.rcParams.update({'font.size': 18})
    plt.scatter(df['fitness1'], df['fitness2'], c=df['rank'], cmap='viridis')
    plt.xlabel('Total entropy')
    plt.ylabel('Total cost')
    plt.title('Pareto front')
    plt.savefig('pareto_front' + name + '.png')


def filter_pareto_front(population):
    '''
    Filter the pareto front
    '''
    # Create the dataframe
    df = pd.DataFrame(columns=['fitness1', 'fitness2', 'rank'])
    for individual in population:
        df = df.append({'fitness1': individual.fitness[0], 'fitness2': individual.fitness[1], 'genes': individual.genes, 'rank': individual.rank}, ignore_index=True)

    # Filter the pareto front
    df = df[df['rank'] == 1]

    return df



if __name__ == '__main__':
    
    for i in range(10):
        # Perform the NSGA-II algorithm
        population = nsga2_algorithm(population_size = 100, num_generations = 200, num_sensors = 9)

        # Plot the pareto front
        plot_pareto_front(population, filter = False)
        plot_pareto_front(population, filter = True)

        # Filter the pareto front
        df = filter_pareto_front(population)
        df.to_csv('pareto_front_'+str(i)+'.csv')

    #count = 0
    #for individual in population:
    #    if individual.rank == 1:
    #        draw_solution(individual.genes, 'solution' + str(count))
    #        count += 1


