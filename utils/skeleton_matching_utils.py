import networkx as nx
import sys

sys.path.append('../')
from parameters import parameters 

from collections import namedtuple
from operator import itemgetter
Matching = namedtuple('Matching', 'id nodes score')


def get_person_proposal_from_network_output(outputs, subgraph, indices, nodes_camera, jsons_for_head=None, CLASSIFICATION_THRESHOLD=0.5):
    #
    #
    # Process the output graph as it comes from the GNN
    #
    #
    # Output:
    matchings = []
    G = nx.Graph()                    # Graph where matching nodes are edges
    # Temporary variables
    nodes_for_matching = dict()       # A dictionary with matching-nodes as keys containing pairs of heads
    heads_linked_in_cameras = dict()  # A dictionary with head nodes as keys holding a set of linked heads
    head_original_camera = dict()     # A dictionary with head nodes as keys holding in which camera it is
    # Code
    src, dst = [x.tolist() for x in subgraph.edges()]
    if type(outputs) is not list:
        output_features = outputs.tolist()
    else:
        output_features = outputs

    for link_i in range(len(src)):
        link_src = src[link_i]
        link_dst = dst[link_i]
        if link_dst == link_src: # ignore self edges
            continue
        if link_src in indices:
            X = link_src
            other = link_dst
        else:
            continue
        camera = nodes_camera[other]

        G.add_node(other, camera=camera) # Add the head to the graph (used for connected components)

        heads_linked_in_cameras[other] = [camera] # Each head is in its own camera
        head_original_camera[other] = camera
        try:                                       # Add the head in the dictionary directly or creating the set
            node_for_matching = nodes_for_matching[X]
            node_for_matching.add(other)
            if len(node_for_matching) == 2:       # If we have iterated over the two sides ...
                if output_features[X] > CLASSIFICATION_THRESHOLD:  # ... and the threshold is met ...
                    matchings.append(Matching(X, list(node_for_matching), output_features[X]))  # ... we add the matching
        except KeyError:
            nodes_for_matching[X] = set([other])
    cams_for_human = {}
    human_index = {}
    cur_human_idx = 0

    for m in sorted(matchings, key=itemgetter(Matching._fields.index('score')), reverse=True):
        a = m.nodes[0]
        b = m.nodes[1]
        original_a = head_original_camera[a]
        linked_a = heads_linked_in_cameras[a]
        original_b = head_original_camera[b]  
        linked_b = heads_linked_in_cameras[b]
        if original_a in linked_b or original_b in linked_a:
            continue
        
        if a in human_index.keys():
            if original_b in cams_for_human[human_index[a]]:
                continue
        if b in human_index.keys():
            if original_a in cams_for_human[human_index[b]]:
                continue
    
        if a not in human_index.keys() and b not in human_index.keys():
            human_index[a] = cur_human_idx
            human_index[b] = cur_human_idx
            cams_for_human[cur_human_idx] = []
            cams_for_human[cur_human_idx].append(original_a)
            cams_for_human[cur_human_idx].append(original_b)
            cur_human_idx += 1
        elif a in human_index.keys() and b not in human_index.keys():
            human_index[b] = human_index[a]
            cams_for_human[human_index[a]].append(original_b)
        elif b in human_index.keys() and a not in human_index.keys():
            human_index[a] = human_index[b]
            cams_for_human[human_index[b]].append(original_a)
        else:
            valid_link = True
            for c in cams_for_human[human_index[b]]:
                if c in cams_for_human[human_index[a]]:
                    valid_link = False
                    break
            if valid_link:
                new_index = human_index[a]
                old_index = human_index[b]
                for n in list(human_index.keys()):
                    if human_index[n] == old_index:
                        human_index[n] = new_index
                del cams_for_human[old_index]
            else:
                continue

        G.add_edge(a, b)
        heads_linked_in_cameras[a].append(original_b)
        heads_linked_in_cameras[b].append(original_a)
    

    #
    #
    # Analyse the connected components to provide the final output
    #
    #

    final_output = []
    cc = nx.connected_components(G)
    for component in cc:
        if len(component) < parameters.min_number_of_views:
            continue
        
        person = dict()
        person_valid = []
        for cam in parameters.used_cameras_skeleton_matching:
            person[cam] = None
        for component_item in component:
            person[G.nodes[component_item]["camera"]] = component_item

        final_output.append(person)

    return final_output
