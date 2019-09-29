#!/usr/bin/env python3
import monkdata as m
import dtree as d
import numpy as np
import random

root = m.attributes[4]

def calculateEntropy():
    entropy = [d.entropy(m.monk1test), d.entropy(m.monk2test), d.entropy(m.monk3test)]
    print(entropy)
    return entropy

def find_information_gain(data_set, attributes):
    information_gain = [
        d.averageGain(data_set, attributes[0]),  
        d.averageGain(data_set, attributes[1]), 
        d.averageGain(data_set, attributes[2]), 
        d.averageGain(data_set, attributes[3]), 
        d.averageGain(data_set, attributes[4]), 
        d.averageGain(data_set, attributes[5])
    ]    
    return information_gain

def find_root_node(information_gain):
    return m.attributes[np.argmax(information_gain)]

def split(node):
    #splitting
    sub_set_A5_value_1_m1 = d.select(m.monk1, node, 1)
    sub_set_A5_value_not_1_m1 = d.select(m.monk1, node, 2) + d.select(m.monk1, node, 3) + d.select(m.monk1, node, 4) + d.select(m.monk1, node, 5)

    #calculating gain to figure out which attribute to use in each of the next nodes
    information_gain_left = find_information_gain(sub_set_A5_value_1_m1, m.attributes)
    information_gain_right = find_information_gain(sub_set_A5_value_not_1_m1, m.attributes)
    information_gain = max(max(information_gain_left), max(information_gain_right))

    #classifying the most common result in each sub tree
    majority_class_left = d.mostCommon(sub_set_A5_value_1_m1)
    majority_class_right = d.mostCommon(sub_set_A5_value_not_1_m1)

    print('left: ', majority_class_left)
    print('right: ', majority_class_right)
    print('information gain: ', information_gain)

    #split(root_nodes)


def build_and_check_trees():
    tree_m1 = d.buildTree(m.monk1, m.attributes)
    tree_m2 = d.buildTree(m.monk2, m.attributes)
    tree_m3 = d.buildTree(m.monk3, m.attributes)

    print(1 - d.check(tree_m1, m.monk1))
    print(1 - d.check(tree_m2, m.monk2))
    print(1 - d.check(tree_m3, m.monk3))

    print(1 - d.check(tree_m1, m.monk1test))
    print(1 - d.check(tree_m2, m.monk2test))
    print(1 - d.check(tree_m3, m.monk3test))


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)

    return ldata[:breakPoint], ldata[breakPoint:]       

def prune(data, test, fraction):
    pruned = []
    train, validate = partition(data, fraction)
    tree = d.buildTree(train, m.attributes)
    trees = d.allPruned(tree)
    lowest_error = 1 - d.check(tree, validate)
    temp_tree = 0
    best_tree = tree

    for tree in trees:
      current_error = 1 - d.check(tree, validate)
      if lowest_error > current_error:
        lowest_error = current_error
        best_tree = tree


    return 1 - d.check(best_tree, test)


if __name__ == "__main__":
    #entropy = calculateEntropy()
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    information_gain = [
        find_information_gain(m.monk1test, m.attributes), 
        find_information_gain(m.monk2test, m.attributes), 
        find_information_gain(m.monk3test, m.attributes)
    ]

    root_nodes = [
        find_root_node(information_gain[0]), 
        find_root_node(information_gain[1]), 
        find_root_node(information_gain[2])
    ]
    build_and_check_trees()
    mean_erros_m1 = []
    mean_erros_m3 = []

    all_errors_m1_for_fraction = []
    all_errors_m3_for_fraction = []  
    iterations = 1000
    index = 0
    for fraction in fractions:
        print(fraction)
        for iteration in range(iterations):
            all_errors_m1_for_fraction.append(prune(m.monk1, m.monk1test, fraction))
            all_errors_m3_for_fraction.append(prune(m.monk3, m.monk3test, fraction))
        
        mean_erros_m1.append(sum(all_errors_m1_for_fraction)/len(all_errors_m1_for_fraction))
        mean_erros_m3.append(sum(all_errors_m3_for_fraction)/len(all_errors_m3_for_fraction))
        print(mean_erros_m1)
        print(mean_erros_m3)
