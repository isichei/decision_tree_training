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



def manual_tree_regressor(data, nodes, outcome_i = None) :

	colours = ['"#F8F8FF"','"#CCCCFF"','"#C4C3D0"','"#92A1CF"','"#8C92AC"','"#0000FF"','"#2A52BE"','"#002FA7"','"#003399"','"#00009C"','"#120A8F"','"#00008B"','"#000080"','"#191970"','"#082567"','"#002366"']
	m_text = '<br/>mean = '
	m_text2 = '<br/>var = '

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
	m2 = []
	p = []
	s = []

	for i in range(n_nodes) :
		m.append(np.mean(list(map(lambda x: x[outcome_i], node_data[i]))))
		m2.append(np.var(list(map(lambda x: x[outcome_i], node_data[i]))))
		s.append(len(node_data[i]))

	m = np.around(m, 2)
	m2 = np.around(m2, 2)
	c_i = np.around(m, 0)

	dot_string = 'digraph Tree { node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ; edge [fontname=helvetica] ;'
	colour_text = '>, fillcolor='

	for n in range(n_nodes) :
		s1 = str(n) + '[label=<node_' + str(n) + m_text + str(m[n]) + m_text2 + str(m2[n]) + '<br/>samples = ' + str(s[n]) + colour_text + colours[int(np.around(m[n],0))] + '] ;'
		dot_string = dot_string + s1

	for i in range(len(nd)) :
		dot_string = dot_string + str(i) + ' -> ' + str(nd[i][0]) + '[labeldistance=2.5, labelangle=45, headlabel="True"];'
		dot_string = dot_string + str(i) + ' -> ' + str(nd[i][1]) + '[labeldistance=2.5, labelangle=-45, headlabel="False"];'

	dot_string = dot_string + "}"

	graph = pydotplus.graph_from_dot_data(dot_string)

	return node_data, graph.create_png()
