import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from nxviz import CircosPlot, ArcPlot
from matplotlib import cm

###############################################################################
#               BUILD GRAPHS FUNCTIONS
###############################################################################


def build_networks_SEC_announcement(G, start_time, sec_announcement_time, end_time):

    # Create empty graphs for the subsets
    G_before = nx.DiGraph()
    G_after = nx.DiGraph()

    # Define the threshold time
    start_time = start_time
    sec_announcement_time = sec_announcement_time
    end_time = end_time

    # Iterate over nodes in the original graph
    for node, data in G.nodes(data=True):
        if data['time'] >= start_time and data['time'] < sec_announcement_time:
            # Add node to the graph for two hours before threshold time
            G_before.add_node(node, **data)
        if data['time'] >= sec_announcement_time and data['time'] <= end_time :
            # Add node to the graph for two hours after threshold time
            G_after.add_node(node, **data)

    # Iterate over edges in the original graph and add edges to subsets if both source and target nodes are present
    for edge in G.edges():
        source, target = edge
        if source in G_before and target in G_before:
            G_before.add_edge(source, target)
        elif source in G_after and target in G_after:
            G_after.add_edge(source, target)
            
    return G_before, G_after
    
def draw_subset_graph(G, nodes, nb_nodes=1000):

    subset_nodes=list(nodes.keys())[0:nb_nodes]
    # Create a subgraph containing only the subset of nodes
    subgraph = G.subgraph(subset_nodes)

    # Draw the subgraph
    nx.draw(subgraph, node_size=1)
    plt.show()

    
def build_subgraph(G, nb_nodes=500):

    selected_nodes = []
    for n, v in G.nodes(data=True):
        if len(selected_nodes) <= nb_nodes+1:
            selected_nodes.append(n)

    sub_G = G.subgraph(selected_nodes)

    return sub_G

def build_subgraph_top_nodes(G, hash_nodes_transactions):
    # Initialize lists to store input and output edges
    input_edges = []
    output_edges = []
    
    # Iterate over each target node
    for target_node_hash in hash_nodes_transactions:
        # Find all edges incident to the target node
        input_edges.extend([(u, v) for (u, v) in G.in_edges(target_node_hash)])
        output_edges.extend([(u, v) for (u, v) in G.out_edges(target_node_hash)])

    # Create a subgraph containing only the target nodes and their incident edges
    subgraph_nodes = set([node for edge in input_edges + output_edges for node in edge])
    subgraph = G.subgraph(subgraph_nodes)
    
    return subgraph

###############################################################################
#               STATISTICS FUNCTIONS
###############################################################################

def plot_statistics_bitcoin_transaction(nodes_df):

    # Make sure 'time' column is in datetime format
    nodes_df['time'] = pd.to_datetime(nodes_df['time'])

    # Splitting the dataframe based on time intervals
    start_time = '2024-01-10 19:30:00'
    sec_announcement_time = '2024-01-10 21:30:00'
    end_time = '2024-01-10 23:30:00'

    transaction_df_before = nodes_df[(nodes_df['time'] >= start_time) & (nodes_df['time'] < sec_announcement_time)]
    transaction_df_after = nodes_df[(nodes_df['time'] >= sec_announcement_time) & (nodes_df['time'] <= end_time)]

    # Plotting histograms of 'fee_usd' for before and after dataframes
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.hist(transaction_df_before['fee_usd'], bins=200, color='lightblue')
    plt.title('Histogram of fee_usd (Before)')
    plt.xlabel('Fee (USD)')
    plt.ylabel('Number of Transactions')

    plt.subplot(1, 2, 2)
    plt.hist(transaction_df_after['fee_usd'], bins=200, color='coral')
    plt.title('Histogram of fee_usd (After)')
    plt.xlabel('Fee (USD)')
    plt.ylabel('Number of Transactions')

    plt.tight_layout()
    plt.show()

    # Plotting number of transactions over time for before and after dataframes
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    transaction_df_before.set_index('time').resample('1T').size().plot(color='lightblue')
    plt.title('Number of Transactions Over Time (Before the SEC announcement)')
    plt.xlabel('Time')
    plt.ylabel('Number of Transactions')

    plt.subplot(1, 2, 2)
    transaction_df_after.set_index('time').resample('1T').size().plot(color='coral')
    plt.title('Number of Transactions Over Time (After the SEC announcement)')
    plt.xlabel('Time')
    plt.ylabel('Number of Transactions')

    # Plotting boxplots of 'fee_usd' for before and after dataframes
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(transaction_df_before['fee_usd'], boxprops=dict(color="lightblue"))
    plt.title('Boxplot of fee_usd (Before)')
    plt.ylabel('Fee (USD)')

    plt.subplot(1, 2, 2)
    plt.boxplot(transaction_df_after['fee_usd'], boxprops=dict(color="coral"))
    plt.title('Boxplot of fee_usd (After)')
    plt.ylabel('Fee (USD)')

    # Plotting histograms of 'weight' for before and after dataframes
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.hist(transaction_df_before['weight'], bins=200, color='lightblue')
    plt.title('Histogram of weight (Before)')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(transaction_df_after['weight'], bins=200, color='coral')
    plt.title('Histogram of weight (After)')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Plotting boxplots of 'weight' for before and after dataframes
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.boxplot(transaction_df_before['weight'], boxprops=dict(color="lightblue"))
    plt.title('Boxplot of weight (Before)')
    plt.ylabel('Weight')

    plt.subplot(1, 2, 2)
    plt.boxplot(transaction_df_after['weight'], boxprops=dict(color="coral"))
    plt.title('Boxplot of weight (After)')
    plt.ylabel('Weight')

    plt.tight_layout()
    plt.show()

def compute_graph_statistics(G):
    print(G)
    print('Number of nodes', len(G.nodes))
    print('Number of edges', len(G.edges))
    print('Average degree', sum(dict(G.degree).values()) / len(G.nodes))
    print('Density', nx.density(G))
    
def nodes_with_highest_degree(G, n_node=10):

    # Compute the degree of each node
    degree_dict = dict(G.degree())

    # Sort the nodes based on their degree in descending order
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # Select the nodes with the highest degree
    num_nodes_to_select = n_node  # Example: select top 10 nodes
    nodes_with_highest_degree = sorted_nodes[:num_nodes_to_select]

    # Create a dictionary to store the degrees of all nodes
    nodes_degrees = {node: degree_dict[node] for node in nodes_with_highest_degree}

    # Convert dictionary to list of tuples
    top_node_degrees = [(key, value) for key, value in nodes_degrees.items()]

    # Create DataFrame from list of tuples
    df = pd.DataFrame(top_node_degrees, columns=['hash_transaction', 'degrees'])

    return df, nodes_with_highest_degree


def compute_betweenness_centrality(G_before, G_after):
    
    # Compute the degree centrality Before and After the sec decision
    deg_cent_before = nx.degree_centrality(G_before)
    deg_cent_after = nx.degree_centrality(G_after)
    
    # Compute the betweenness centrality Before and After the sec decision
    bet_cent_before = nx.betweenness_centrality(G_before)
    bet_cent_after = nx.betweenness_centrality(G_after)

    fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # Adjust the figsize here as desired    

    # Create a scatter plot of betweenness centrality and degree centrality
    axs[0].scatter(list(bet_cent_before.values()), list(deg_cent_before.values()), color="lightblue", label="Before SEC announcement")
    axs[0].set_title("Plot of betweenness centrality and degree centrality after the SEC announcement")
    axs[0].set_xlabel("betweenness centrality")
    axs[0].set_ylabel("deg_cent_before")
    axs[0].legend()    

    # Create a scatter plot of betweenness centrality and degree centrality
    axs[1].scatter(list(bet_cent_after.values()), list(deg_cent_after.values()), color="coral", label="Before SEC announcement")
    axs[1].set_title("Plot of betweenness centrality and degree centrality before the SEC announcement")
    axs[1].set_xlabel("betweenness centrality")
    axs[1].set_ylabel("deg_cent_before")
    axs[1].legend()    


def compute_degree_distribution(G_before, G_after):
    
    # Compute the degree centrality of the two graphs
    deg_cent_before = nx.degree_centrality(G_before)
    deg_cent_after = nx.degree_centrality(G_after)

    # Compute the degree distribution of the two graphs
    degrees_before = [G_before.degree(n) for n in G_before.nodes()]
    degrees_after = [G_after.degree(n) for n in G_after.nodes()]
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Adjust the figsize here as desired
    
    #### PLOT BEFORE ANNOUNCEMENT ####
    # Plot a histogram of the degree centrality distribution of the graph.
    axs[0][0].hist(list(deg_cent_before.values()), color="lightblue", label="Before SEC announcement", stacked=True, bins=10)
    axs[0][0].set_title("Degree centrality distribution before the SEC announcement")
    axs[0][0].legend()
    
    # Plot a histogram of the degree distribution of the graph
    axs[1][0].hist(degrees_before, color="lightblue", label="Before SEC announcement", stacked=True, bins=10)
    axs[1][0].set_title("Degree distribution of the graph before the SEC announcement")
    axs[1][0].legend()

    # Plot a scatter plot of the centrality distribution and the degree distribution
    axs[2][0].scatter(degrees_before, list(deg_cent_before.values()), color="lightblue", label="Before SEC announcement")
    axs[2][0].set_title("Centrality and degree distribution before the SEC announcement")
    axs[2][0].set_xlabel("Degrees")
    axs[2][0].set_ylabel("Degree Centrality")
    axs[2][0].legend()

    #### PLOT BEFORE ANNOUNCEMENT ####

    # Plot a histogram of the degree centrality distribution of the graph.
    axs[0][1].hist(list(deg_cent_after.values()), color="coral", label="After SEC announcement", stacked=True, bins=10)
    axs[0][1].set_title("Degree centrality distribution after the SEC announcement")
    axs[0][1].legend()

    # Plot a histogram of the degree distribution of the graph
    axs[1][1].hist(degrees_after, color="coral", label="After SEC announcement", stacked=True, bins=10)
    axs[1][1].set_title("Degree distribution of the graph after the SEC announcement")
    axs[1][1].legend()
    
    # Plot a scatter plot of the centrality distribution and the degree distribution
    axs[2][1].scatter(degrees_after, list(deg_cent_after.values()), color="coral", label="After SEC announcement")
    axs[2][1].set_title("Centrality and degree distribution after the SEC announcement")
    axs[2][1].set_xlabel("Degrees")
    axs[2][1].set_ylabel("Degree Centrality")
    axs[2][1].legend()
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the space between subplots
    
###############################################################################
#               PLOTS FUNCTIONS
###############################################################################

def draw_subset_graph(G, nodes, title, nb_nodes=5000):

    subset_nodes=list(nodes.keys())[0:nb_nodes]
    # Create a subgraph containing only the subset of nodes
    subgraph = G.subgraph(subset_nodes)

    # Draw the subgraph
    nx.draw(subgraph, node_size=1)
    plt.title(title+"("+str(nb_nodes)+")")
    plt.show()
    
    
def plot_node_with_highest_degrees(G, nodes_highest_degree, type_sec="(After SEC announcement)"):
    degree_dict = dict(G.degree())
    # Create a subgraph containing only the nodes with the highest degree
    subgraph = G.subgraph(nodes_highest_degree)
    nb_nodes = len(nodes_highest_degree)

    # Specify node size and colors based on their degree
    node_size = [degree_dict[node] * 10 for node in subgraph.nodes()]
    node_color = [degree_dict[node] for node in subgraph.nodes()]

    # Draw the subgraph with customized node size and colors
    nx.draw(subgraph, node_size=node_size, node_color=node_color)
    plt.title("Plot bitcoin network transaction for the first "+ str(nb_nodes)+" nodes with the highest degree "+ type_sec)
    
    # Add color bar
    norm = plt.Normalize(vmin=min(node_color), vmax=max(node_color))
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal')
    cbar.set_label('Degree')
    
    plt.show()
    
    
def plot_graph_for_specific_node(G, hash_node_transaction):
    # Define the hash of the node you want to focus on
    target_node_hash = hash_node_transaction

    # Find all edges incident to the target node
    input_edges = [(u, v) for (u, v) in G.in_edges(target_node_hash)]
    output_edges = [(u, v) for (u, v) in G.out_edges(target_node_hash)]

    # Create a subgraph containing the target node and its incident edges
    subgraph_nodes = set([node for edge in input_edges + output_edges for node in edge])
    subgraph = G.subgraph(subgraph_nodes)

    # Get neighbors of the target node
    neighbor_nodes = list(subgraph.neighbors(target_node_hash))

    # Draw the subgraph
    pos = nx.spring_layout(subgraph)  # You can choose any layout algorithm you prefer

    # Draw input edges with 'slateblue' color
    nx.draw_networkx_edges(subgraph, pos, edgelist=input_edges, edge_color='slateblue', width=2)

    # Draw output edges with 'coral' color
    nx.draw_networkx_edges(subgraph, pos, edgelist=output_edges, edge_color='coral', width=2)
    
    # Draw neighbor nodes in 'yellow'
    nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph_nodes, node_color='gray', node_size=50)
    
    # Draw the target node in 'green'
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[target_node_hash], node_color='green', node_size=200)


    # Add labels if needed
    # nx.draw_networkx_labels(subgraph, pos)

    # Add a text box to display the number of input and output edges
    plt.text(0.1, 0.9, f'Input Edges: {len(input_edges)}', transform=plt.gca().transAxes, color='slateblue')
    plt.text(0.1, 0.85, f'Output Edges: {len(output_edges)}', transform=plt.gca().transAxes, color='coral')

    plt.title("Focus on one bitcoin transaction\n(" + hash_node_transaction + ")")
    plt.axis('off')  # Turn off the axis
    plt.show()
    
def circo_plot_graph(G_before, G_after, node_order='input_total_usd', node_color='input_total_usd'):
    # Create the CircosPlot object for G_before
    c_before = CircosPlot(G_before, node_order=node_order, node_color=node_color)
    c_before.ax.set_title("Circos Plot of bitcoin transaction network before the SEC announcement ordered by " + node_order)

    # Create the CircosPlot object for G_after
    c_after = CircosPlot(G_after, node_order=node_order, node_color=node_color)
    c_after.ax.set_title("Circos Plot of bitcoin transaction network after the SEC announcement" + node_order)

    c_before.draw
    c_after.draw

    plt.show()

def arc_plot_graph(G_before, G_after, node_order='input_total_usd', node_color='input_total_usd'):
    # Create the ArcPlot object for G_before
    a_before = ArcPlot(G_before, node_order=node_order, node_color=node_color)
    a_before.ax.set_title("Arc plot of bitcoin transaction network before the SEC announcement" + node_order)

    # Create the ArcPlot object for G_after
    a_after = ArcPlot(G_after, node_order=node_order, node_color=node_color)
    a_after.ax.set_title("Arc plot of bitcoin transaction network after the SEC announcement" + node_order)

    a_before.draw
    a_after.draw

    plt.show()
    
def transaction_fees_amount(G_before, G_after):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the figsize here as desired    

    # Create a color map
    cmap = cm.get_cmap('coolwarm')  # You can choose any colormap you prefer

    # Extract the 'fee_usd' attribute for all nodes
    node_values_before = [G_before.nodes[n]['fee_usd'] for n in G_before.nodes()]
    node_values_after = [G_after.nodes[n]['fee_usd'] for n in G_after.nodes()]

    # Normalize node values between 0 and 1 for colormap
    norm_before = plt.Normalize(min(node_values_before), max(node_values_before))
    norm_after = plt.Normalize(min(node_values_after), max(node_values_after))

    # Create a color list based on the color map and normalized values
    node_colors_before = [cmap(norm_before(value)) for value in node_values_before]
    node_colors_after = [cmap(norm_after(value)) for value in node_values_after]

    # Draw the graphs side by side
    pos_before = nx.circular_layout(G_before) 
    pos_after = nx.circular_layout(G_after) 
    
    # Draw before network
    nx.draw(G_before, pos_before, ax=axs[0], with_labels=False, node_color=node_colors_before, cmap=cmap, node_size=20)
    axs[0].set_title('Bitcoin Transaction Network after the SEC announcement')

    # Draw after network
    nx.draw(G_after, pos_after, ax=axs[1], with_labels=False, node_color=node_colors_after, cmap=cmap, node_size=20)
    axs[1].set_title('Bitcoin Transaction Network after the SEC announcement')

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_before)
    sm.set_array([])
    plt.colorbar(sm, ax=axs, label='Value (USD)', orientation='horizontal', pad=0.05)

    plt.suptitle('Bitcoin Transaction Network with the highest amount in dollars ($)')
    plt.show()

def plot_graph_before_after(G_before, G_after, hash_nodes_before, hash_nodes_after):
    # Initialize lists to store input and output edges for G_before and G_after
    input_edges_before = []
    output_edges_before = []
    input_edges_after = []
    output_edges_after = []
    
    # Iterate over each target node in hash_nodes_before
    for target_node_hash in hash_nodes_before:
        # Find all edges incident to the target node in G_before
        input_edges_before.extend([(u, v) for (u, v) in G_before.in_edges(target_node_hash)])
        output_edges_before.extend([(u, v) for (u, v) in G_before.out_edges(target_node_hash)])
        
    # Iterate over each target node in hash_nodes_after
    for target_node_hash in hash_nodes_after:
        # Find all edges incident to the target node in G_after
        input_edges_after.extend([(u, v) for (u, v) in G_after.in_edges(target_node_hash)])
        output_edges_after.extend([(u, v) for (u, v) in G_after.out_edges(target_node_hash)])

    # Create subgraphs containing only the target nodes and their incident edges for G_before and G_after
    subgraph_nodes_before = set([node for edge in input_edges_before + output_edges_before for node in edge])
    subgraph_before = G_before.subgraph(subgraph_nodes_before)

    subgraph_nodes_after = set([node for edge in input_edges_after + output_edges_after for node in edge])
    subgraph_after = G_after.subgraph(subgraph_nodes_after)

    # Draw the subgraphs side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Draw G_before
    pos_before = nx.circular_layout(subgraph_before)
    nx.draw(subgraph_before, pos_before, ax=axs[0], node_size=20, with_labels=False, edgelist=input_edges_before, edge_color='slateblue', label='Input Edges (Before)')
    nx.draw(subgraph_before, pos_before, ax=axs[0], node_size=20, with_labels=False, edgelist=output_edges_before, edge_color='coral', label='Output Edges (Before)')

    # Count the number of input and output edges for G_before
    num_input_edges_before = len(input_edges_before)
    num_output_edges_before = len(output_edges_before)
    
    # Add text to show the counts for G_before
    axs[0].text(0.05, 0.95, f'Input Edges: {num_input_edges_before}', transform=axs[0].transAxes, color='slateblue')
    axs[0].text(0.05, 0.90, f'Output Edges: {num_output_edges_before}', transform=axs[0].transAxes, color='coral')

    axs[0].set_title("Bitcoin transactions before the SEC announcement")

    # Draw G_after
    pos_after = nx.circular_layout(subgraph_after)
    nx.draw(subgraph_after, pos_after, ax=axs[1], node_size=20, with_labels=False, edgelist=input_edges_after, edge_color='slateblue', label='Input Edges (After)')
    nx.draw(subgraph_after, pos_after, ax=axs[1], node_size=20, with_labels=False, edgelist=output_edges_after, edge_color='coral', label='Output Edges (After)')

    # Count the number of input and output edges for G_after
    num_input_edges_after = len(input_edges_after)
    num_output_edges_after = len(output_edges_after)
    
    # Add text to show the counts for G_after
    axs[1].text(0.05, 0.95, f'Input Edges: {num_input_edges_after}', transform=axs[1].transAxes, color='slateblue')
    axs[1].text(0.05, 0.90, f'Output Edges: {num_output_edges_after}', transform=axs[1].transAxes, color='coral')

    axs[1].set_title("Bitcoin transactions after the SEC announcement")

    plt.show()
    

def plot_distribution_degrees(degrees_before, degrees_after):

    plt.hist([degrees_before, degrees_after], bins=range(min(min(degrees_before), min(degrees_after)), max(max(degrees_before), max(degrees_after)) + 1, 1), color=['lightblue', 'coral'], label=['Before SEC announcement', 'After SEC announcement'])

    # Adding labels and title
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Frequency of Degrees')
    plt.legend()
    plt.show()
    
def plot_transaction_graph_bis(G_before, G_after):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the figsize here as desired
    
    # Create a color map
    cmap = cm.get_cmap('coolwarm')  # You can choose any colormap you prefer
    
    # Extract the 'input_total_usd' attribute for all nodes
    node_values_before = [G_before.nodes[n]['input_total_usd'] for n in G_before.nodes()]
    node_values_after = [G_after.nodes[n]['input_total_usd'] for n in G_after.nodes()]

    # Normalize node values between 0 and 1 for the colormap
    norm_before = plt.Normalize(min(node_values_before), max(node_values_before))
    norm_after = plt.Normalize(min(node_values_after), max(node_values_after))

    # Create a color list based on the colormap and normalized values
    node_colors_before = [cmap(norm_before(value)) for value in node_values_before]
    node_colors_after = [cmap(norm_after(value)) for value in node_values_after]

    # Draw the graphs side by side
    pos_before = nx.random_layout(G_before)  # Change circular_layout() to spring_layout()
    pos_after = nx.random_layout(G_after)  # Change circular_layout() to spring_layout()
    
    # Draw the network before
    nx.draw(G_before, pos_before, ax=axs[0], with_labels=False, node_color=node_colors_before, cmap=cmap, node_size=[G_before.degree(n) * 100 for n in G_before.nodes()])
    axs[0].set_title('Bitcoin Transaction Network before the SEC announcement')

    # Draw the network after
    nx.draw(G_after, pos_after, ax=axs[1], with_labels=False, node_color=node_colors_after, cmap=cmap, node_size=[G_after.degree(n) * 100 for n in G_after.nodes()])
    axs[1].set_title('Bitcoin Transaction Network after the SEC announcement')

    # Add the color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_before)
    sm.set_array([])
    plt.colorbar(sm, ax=axs, label='Input value (\$)', orientation='horizontal', pad=0.05)

    plt.suptitle('Bitcoin Transaction Network with node size based on degrees and node color based on input total \$-USD')
    plt.show()
    
def plot_transaction_bitcoin_network(G, G_before, G_after):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Ajustez la taille de la figure ici selon vos préférences
    
    # Créer une carte de couleur
    cmap = cm.get_cmap('coolwarm')  # Vous pouvez choisir n'importe quelle carte de couleur que vous préférez
    
    # Extraire l'attribut 'input_total_usd' pour tous les nœuds
    node_values_before = [G_before.nodes[n]['input_total_usd'] for n in G_before.nodes()]
    node_values_after = [G_after.nodes[n]['input_total_usd'] for n in G_after.nodes()]

    # Normaliser les valeurs des nœuds entre 0 et 1 pour la colormap
    norm_before = plt.Normalize(min(node_values_before), max(node_values_before))
    norm_after = plt.Normalize(min(node_values_after), max(node_values_after))

    # Créer une liste de couleur basée sur la carte de couleur et les valeurs normalisées
    node_colors_before = [cmap(norm_before(value)) for value in node_values_before]
    node_colors_after = [cmap(norm_after(value)) for value in node_values_after]

    # Dessiner les graphes côte à côte
    pos_before = nx.circular_layout(G_before) 
    pos_after = nx.circular_layout(G_after) 
    
    edge_width_before = list()
    for u,v in G_before.edges():
        usd_value = G[u][v][0]["value_usd"] 
        edge_width_before.append(usd_value)
                
    edge_width_after = list()
    for u,v in G_after.edges():
        usd_value = G[u][v][0]["value_usd"]
        edge_width_after.append(usd_value)
        
    edge_width_before_norm = [(x - min(edge_width_before)) / (max(edge_width_before) - min(edge_width_before)) for x in edge_width_before]
    edge_width_after_norm = [(x - min(edge_width_after)) / (max(edge_width_after) - min(edge_width_after)) for x in edge_width_after]
    
    edge_width_before_norm = [x * 5 for x in edge_width_before_norm]

    edge_width_after_norm = [x * 5 for x in edge_width_after_norm]
    
    # Dessiner le réseau avant
    nx.draw(G_before, pos_before, ax=axs[0], with_labels=False, node_color=node_colors_before, cmap=cmap, node_size=[G_before.degree(n) * 100 for n in G_before.nodes()], width=edge_width_before_norm)
    axs[0].set_title('Bitcoin Transaction Network before the SEC announcement')

    # Dessiner le réseau après
    nx.draw(G_after, pos_after, ax=axs[1], with_labels=False, node_color=node_colors_after, cmap=cmap, node_size=[G_after.degree(n) * 100 for n in G_after.nodes()], width=edge_width_after_norm)
    axs[1].set_title('Bitcoin Transaction Network after the SEC announcement')

    # Ajouter la barre de couleur
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_before)
    sm.set_array([])
    plt.colorbar(sm, ax=axs, label='Input Value ($)', orientation='horizontal', pad=0.05)

    plt.suptitle('Bitcoin Transaction Network with node size based on degrees and node color based on input total \$-USD and edges size based on the value \$-USD transaction')
    plt.show()
    
def plot_betweenness_centrality_network(G, type_sec="Before the SEC announcement"):
        
    pos = nx.random_layout(G)
    
    betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
    
    # Node color based on degree
    node_color = [2.0e6 * G.degree(v) for v in G]
    
    # Node size based on betweenness centrality
    node_size = [v * 1.0e6 for v in betCent.values()]
    
    edges_width = list()
    
    plt.figure(figsize=(12, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos=pos,
                           node_color=node_color,
                           node_size=node_size)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos=pos)
    
    # Add color bar for node color
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_color) / 2.0e6, vmax=max(node_color) / 2.0e6))
    sm.set_array([])
    cbar = plt.colorbar(sm, label='Node Degree', orientation='horizontal')

    
    plt.title('Betweeness centrality '+ type_sec)
    plt.axis('off')
    plt.show()