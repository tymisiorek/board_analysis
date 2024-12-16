import pandas as pd
from collections import defaultdict
import json
import remove_non_sample as rns

absolute_path = "C:\\Users\\tykun\\\OneDrive\\Documents\\SchoolDocs\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"

years = ["1999", "2000", "2005", "2008", "2009", "2013"]

board_member_dict = defaultdict(set)
edges_list = []
nodes_dict = defaultdict(lambda: {'Interlock_Count': 0, 'AffiliationId': None})

# Load base positions from JSON
base_positions_path = f"{absolute_path}{board_dataframes}full_board_interlocking_positions.json"

with open(base_positions_path, 'r') as json_file:
    base_positions_data = json.load(json_file)

# Create a dictionary mapping node IDs to positions
base_positions_dict = {
    node['id']: {'x': node['x'], 'y': node['y']}
    for node in base_positions_data['nodes']
}

# Iterate through each year
for year in years:
    print(f"Processing interlocks for: {year}")
    same_year_interlocked_edges_path = f"{absolute_path}{board_dataframes}{year}_interlocked_edges.csv"
    same_year_interlocked_nodes_path = f"{absolute_path}{board_dataframes}{year}_interlocked_nodes.csv"

    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)
    # Remove non-sample schools
    boards_df = rns.remove_non_samples(boards_df)

    board_statistics_path = f"{absolute_path}{altered_dataframes}university_board_statistics.csv"
    board_statistics_df = pd.read_csv(board_statistics_path)

    # Iterate over each board member
    for index, row in boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']  # Ensure 'AffiliationId' is a column in boards_df
        
        # If this board member has been seen before in a different institution, record an interlock
        for previous_institution in board_member_dict[name]:
            if previous_institution != institution:
                # Record the interlock as an edge
                edges_list.append({
                    'Source': previous_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1  # Each interlock counts as 1 by default
                })
                # Increment the interlock count for the involved institutions
                nodes_dict[previous_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1

        # Add the current institution to the set of institutions this member is associated with
        board_member_dict[name].add(institution)

        # Add or update the AffiliationId in the nodes dictionary
        if institution not in nodes_dict or nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id

    # Filter nodes_dict to include only institutions with interlocks
    filtered_nodes_dict = {key: value for key, value in nodes_dict.items() if value['Interlock_Count'] > 0}

    # Create a DataFrame for nodes (universities) with their interlock counts and regions
    nodes_df = pd.DataFrame([(key, value['Interlock_Count'], value['AffiliationId']) 
                             for key, value in filtered_nodes_dict.items()], 
                            columns=['Id', 'Interlock_Count', 'AffiliationId'])
    nodes_df['Label'] = nodes_df['Id']  # Use the institution name as the label

    # Add 'female_president' status to nodes_df
    def lookup_female_president(row):
        # First attempt: Match by AffiliationId and year
        matching_rows = board_statistics_df[
            (board_statistics_df['AffiliationId'] == row['AffiliationId']) & 
            (board_statistics_df['Year'] == int(year))
        ]
        if not matching_rows.empty:
            return matching_rows['female_president'].iloc[0]
        
        # Fallback: Match by Institution name and year
        matching_rows = board_statistics_df[
            (board_statistics_df['Institution'] == row['Id']) & 
            (board_statistics_df['Year'] == int(year))
        ]
        if not matching_rows.empty:
            return matching_rows['female_president'].iloc[0]

        # Default: No match
        return None

    nodes_df['female_president'] = nodes_df.apply(lookup_female_president, axis=1)
    nodes_df['female_president'] = nodes_df['female_president'].fillna('unknown')

    # Add x, y positions from base_positions_dict
    def get_position(node_id):
        if node_id in base_positions_dict:
            return base_positions_dict[node_id]['x'], base_positions_dict[node_id]['y']
        return None, None  # Default position for missing nodes
    
    nodes_df['x'], nodes_df['y'] = zip(*nodes_df['Id'].apply(get_position))

    # Ensure correct column order and uniqueness
    nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count', 'AffiliationId', 'female_president', 'x', 'y']]

    # Create a DataFrame for edges (interlocks between institutions)
    edges_df = pd.DataFrame(edges_list)

    # Ensure correct column order for edges
    edges_df = edges_df[['Source', 'Target', 'Type', 'Weight']]

    # Save the DataFrames to CSV files
    nodes_df.to_csv(same_year_interlocked_nodes_path, index=False)
    edges_df.to_csv(same_year_interlocked_edges_path, index=False)
