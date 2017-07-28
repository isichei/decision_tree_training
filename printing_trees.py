import numpy as np 
from sklearn import tree
import pydotplus
from scipy.stats import entropy as e_fun
import math

def split(splitter_function, data) :
    true_split = list(filter(lambda x: splitter_function(x), data))
    false_split = list(filter(lambda x: not splitter_function(x), data))
    return true_split, false_split

# manual gini
def g_fun(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    if fair_area != 0 :
      return (fair_area - area) / fair_area
    else :
      return 0


def show_tree(classifier, feature_names = None, target_names = None) :
    dot_data = tree.export_graphviz(classifier, out_file=None, 
                             feature_names=feature_names,  
                             class_names=target_names,  
                             filled=True, rounded=True,  
                             special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    return graph.create_png() 


def gini(x) :
    clean_x = [value for value in x if not math.isnan(value)]
    if len(clean_x) > 1 :
      return g_fun(clean_x)
    else : 
      return 0

def entropy(x) :
    clean_x = [value for value in x if not math.isnan(value)]
    if len(clean_x) > 1 :
      return e_fun(clean_x)
    else : 
      return 0

def manual_tree_classifier(data, nodes, outcome_i = None, thres = 0.5, gini_measure = True) :
    
    if gini_measure :
        measure_function = gini
        m_text = '<br/>gini = '
    else :
      measure_function = entropy
      m_text = '<br/>entropy = '
        
    if outcome_i is None :
        outcome_i = data.shape[1]-1

    node_data = []

    n_nodes = len(nodes)*2 + 1

    # Define length of data_splits based on nodes array
    for i in range(n_nodes) :
        node_data.append([])

    all_nodes = list(range(1,n_nodes))
    nd = []
    for i in range(len(nodes)) :
        nd.append(all_nodes[:2])
        all_nodes = all_nodes[2:]

    node_data[0] = data
    for i in range(len(nodes)) :
        node_data[nd[i][0]], node_data[nd[i][1]] = split(nodes[i], node_data[i])

    m = []
    p = []
    s = []
    
    for i in range(n_nodes) :
        m.append(measure_function(list(map(lambda x: x[outcome_i], node_data[i]))))
        prop = np.sum(list(map(lambda x: x[outcome_i], node_data[i])))
        s.append(len(node_data[i]))
        p.append(prop / s[-1] if (prop != 0 and s[-1] != 0) else 0)

    p = np.around(p, 2)
    m = np.around(m, 2)


    dot_string = 'digraph Tree { node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ; edge [fontname=helvetica] ;'

    for n in range(n_nodes) :
        colour = '>, fillcolor="#4bc330"] ;' if p[n] >= 0.5 else '>, fillcolor="#e34464"] ;'
        s1 = str(n) + '[label=<node_' + str(n) + m_text + str(m[n]) + '<br/>samples = ' + str(s[n]) + '<br/> percentage class = ' + str(p[n]) + colour 
        dot_string = dot_string + s1

    for i in range(len(nd)) :
        dot_string = dot_string + str(i) + ' -> ' + str(nd[i][0]) + '[labeldistance=2.5, labelangle=45, headlabel="True"];'
        dot_string = dot_string + str(i) + ' -> ' + str(nd[i][1]) + '[labeldistance=2.5, labelangle=-45, headlabel="False"];'

    dot_string = dot_string + "}"

    graph = pydotplus.graph_from_dot_data(dot_string)
    
    return node_data, graph.create_png()

# # Using those arrays, we can parse the tree structure:
# def print_tree(estimator, feature_names = None) :
#     n_nodes = estimator.tree_.node_count
#     children_left = estimator.tree_.children_left
#     children_right = estimator.tree_.children_right
#     feature = estimator.tree_.feature
#     threshold = estimator.tree_.threshold
    

#     features_out = []
#     for f in feature :
#         if feature_names is None :
#             features_out.append("X[:," + str(f) + "]")
#         else :
#             features_out.append(feature_names[f])

#     # The tree structure can be traversed to compute various properties such
#     # as the depth of each node and whether or not it is a leaf.
#     node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
#     is_leaves = np.zeros(shape=n_nodes, dtype=bool)
#     stack = [(0, -1)]  # seed is the root node id and its parent depth
#     while len(stack) > 0:
#         node_id, parent_depth = stack.pop()
#         node_depth[node_id] = parent_depth + 1

#         # If we have a test node
#         if (children_left[node_id] != children_right[node_id]):
#             stack.append((children_left[node_id], parent_depth + 1))
#             stack.append((children_right[node_id], parent_depth + 1))
#         else:
#             is_leaves[node_id] = True

#     print("The binary tree structure has %s nodes and has "
#           "the following tree structure:"
#           % n_nodes)
#     for i in range(n_nodes):
#         if is_leaves[i]:
#             print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
#         else:
#             print("%snode=%s test node: go to node %s if %s <= %ss else to "
#                   "node %s."
#                   % (node_depth[i] * "\t",
#                      i,
#                      children_left[i],
#                      features_out[i],
#                      threshold[i],
#                      children_right[i],
#                      ))
#     print()

# def print_tree_decision_single(estimator, X_test, sample_id, feature_names = None) :
#     # First let's retrieve the decision path of each sample. The decision_path
#     # method allows to retrieve the node indicator functions. A non zero element of
#     # indicator matrix at the position (i, j) indicates that the sample i goes
#     # through the node j.

#     node_indicator = estimator.decision_path(X_test)

#     # Similarly, we can also have the leaves ids reached by each sample.

#     leave_id = estimator.apply(X_test)

#     # Now, it's possible to get the tests that were used to predict a sample or
#     # a group of samples. First, let's make it for the sample.

#     node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
#                                         node_indicator.indptr[sample_id + 1]]

#     print('Rules used to predict sample %s: ' % sample_id)
#     for node_id in node_index:
#         if leave_id[sample_id] != node_id:
#             continue

#         if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
#             threshold_sign = "<="
#         else:
#             threshold_sign = ">"

#         print("decision id node %s : (X[%s, %s] (= %s) %s %s)"
#               % (node_id,
#                  sample_id,
#                  feature[node_id],
#                  X_test[i, feature[node_id]],
#                  threshold_sign,
#                  threshold[node_id]))

# def print_tree_decisions_group(estimator, X_test, sample_ids) :
#     # For a group of samples, we have the following common node.
#     sample_ids = [0, 1]
#     common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
#                     len(sample_ids))

#     common_node_id = np.arange(n_nodes)[common_nodes]

#     print("\nThe following samples %s share the node %s in the tree"
#           % (sample_ids, common_node_id))
#     print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))