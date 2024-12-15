import pandas as pd
from collections import defaultdict

import remove_non_sample as rns


absolute_path = "C:\\Users\\tykun\\\OneDrive\\Documents\\SchoolDocs\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts =  "scripts\\"
board_dataframes = "board_dataframes\\"

years = ["1999", "2000", "2005", "2008", "2009", "2013"]



board_member_dict = defaultdict(set)
edges_list = []
nodes_dict = defaultdict(lambda: {'Interlock_Count': 0})

# Iterate through each year
for year in years:
    print(f"Interlocks for: {year}")
    same_year_interlocked_edges_path = f"{absolute_path}{board_dataframes}{year}_interlocked_edges.csv"
    same_year_interlocked_nodes_path = f"{absolute_path}{board_dataframes}{year}_interlocked_nodes.csv"

    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)
    #remove non sample schools
    boards_df = rns.remove_non_samples(boards_df)

    # Iterate over each board member
    for index, row in boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
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

        # Ensure the Region is recorded for each institution

    # Create a DataFrame for nodes (universities) with their interlock counts and regions
    nodes_df = pd.DataFrame([(key, value['Interlock_Count']) for key, value in nodes_dict.items()], 
                            columns=['Id', 'Interlock_Count'])
    nodes_df['Label'] = nodes_df['Id']  # Use the institution name as the label

    # Ensure correct column order and uniqueness
    nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count']]

    # Create a DataFrame for edges (interlocks between institutions)
    edges_df = pd.DataFrame(edges_list)

    # Ensure correct column order for edges
    edges_df = edges_df[['Source', 'Target', 'Type', 'Weight']]

    # Save the DataFrames to CSV files
    nodes_df.to_csv(same_year_interlocked_nodes_path, index=False)
    edges_df.to_csv(same_year_interlocked_edges_path, index=False)