import pandas as pd
#获取弱边数据集

# Load the datasets
geometry_path = 'physics_utf8.csv'
node_path = 'physics(id+name+lm+node).csv'

geometry_df = pd.read_csv(geometry_path)
node_df = pd.read_csv(node_path)

# Create a mapping from node_name to node_id
name_to_id_map = node_df.set_index('node_name')['node_id'].to_dict()

# Initialize a list to store the edges
edges = []

# Iterate through the geometry dataframe
for index, row in geometry_df.iterrows():
    if row['RefD'] > 0:  # Check if RefD > 0
        # Get the source and target node names
        source_node_name = row['A']
        target_node_name = row['B']

        # Convert node names to node IDs using the mapping
        source_node_id = name_to_id_map.get(source_node_name, None)
        target_node_id = name_to_id_map.get(target_node_name, None)

        # If both IDs are found, add the edge to the list
        if source_node_id is not None and target_node_id is not None:
            edges.append((source_node_id, target_node_id))

# Convert the list of edges into a DataFrame
edges_df = pd.DataFrame(edges, columns=['source_node_id', 'target_node_id'])

# Output the first few rows to verify the result
print(edges_df.head())

# Save the edges dataframe to a new CSV file
edges_output_path = 'edges_dataset.csv'
edges_df.to_csv(edges_output_path, index=False)

