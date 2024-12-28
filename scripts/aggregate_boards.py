import pandas as pd
from collections import defaultdict
import json
import remove_non_sample as rns

absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"

years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

# This dictionary maps board members (by name) to a set of institutions they've appeared in
board_member_dict = defaultdict(set)

# This list will hold all edge information (interlocks) across all years
edges_list = []

# A dictionary tracking each institution's cumulative data (including interlock count)
nodes_dict = defaultdict(lambda: {
    'Interlock_Count': 0,
    'AffiliationId': None,
})

# Path to board statistics (needed to look up female_president info)
board_statistics_path = f"{absolute_path}{altered_dataframes}sample_board_statistics.csv"
board_statistics_df = pd.read_csv(board_statistics_path)

# ------------------------------------------------------------------------------
# Process each year and accumulate edges and nodes
# ------------------------------------------------------------------------------
for year in years:
    print(f"Processing interlocks for: {year}")

    # Read the boards for this year
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)

    # Remove non-sample schools
    boards_df = rns.remove_non_samples(boards_df)

    # Iterate over each board member
    for index, row in boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']  # Ensure 'AffiliationId' is a column in boards_df

        # If this board member has been seen before in a different institution, record an interlock
        for previous_institution in board_member_dict[name]:
            if previous_institution != institution:
                # Record the interlock as an edge with the current year
                edges_list.append({
                    'Source': previous_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1,      # Each interlock counts as 1 by default
                    'Year': year      # Label the year
                })
                # Increment the interlock count in the nodes dictionary
                nodes_dict[previous_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1

        # Add the current institution to the set of institutions this member is associated with
        board_member_dict[name].add(institution)

        # Add or update the AffiliationId in the nodes dictionary
        if nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id

# ------------------------------------------------------------------------------
# Build the aggregated nodes DataFrame
# ------------------------------------------------------------------------------
# Filter nodes_dict to include only institutions with interlocks
filtered_nodes_dict = {key: value for key, value in nodes_dict.items() if value['Interlock_Count'] > 0}

# Convert the filtered_nodes_dict into a DataFrame
nodes_df = pd.DataFrame([
    (key, value['Interlock_Count'], value['AffiliationId'])
    for key, value in filtered_nodes_dict.items()
], columns=['Id', 'Interlock_Count', 'AffiliationId'])

# Add a label column for visualization (e.g., label is the same as Id)
nodes_df['Label'] = nodes_df['Id']

# Function to look up whether the institution had a female president in any year
def lookup_female_president(row):
    # Find all matching rows by AffiliationId
    matching_rows = board_statistics_df[
        (board_statistics_df['AffiliationId'] == row['AffiliationId'])
    ]
    if not matching_rows.empty:
        # If multiple years, you might want to aggregate or choose a specific logic
        return matching_rows['female_president'].mode()[0]  # Most common status
    # Fallback: Match by Institution name
    matching_rows = board_statistics_df[
        (board_statistics_df['Institution'] == row['Id'])
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]
    # Default: No match
    return 'unknown'

nodes_df['female_president'] = nodes_df.apply(lookup_female_president, axis=1)
nodes_df['female_president'] = nodes_df['female_president'].fillna('unknown')

# Add "control" and "region" columns to nodes_df
def lookup_column(row, column_name):
    # Find all matching rows by AffiliationId
    matching_rows = board_statistics_df[
        (board_statistics_df['AffiliationId'] == row['AffiliationId'])
    ]
    if not matching_rows.empty:
        # If multiple matches, return the most common value
        return matching_rows[column_name].mode()[0]
    # Fallback: Match by Institution name
    matching_rows = board_statistics_df[
        (board_statistics_df['Institution'] == row['Id'])
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]
    # Default: No match
    return 'unknown'

# Add control and region columns
nodes_df['control'] = nodes_df.apply(lambda row: lookup_column(row, 'control'), axis=1)
nodes_df['region'] = nodes_df.apply(lambda row: lookup_column(row, 'region'), axis=1)

# Ensure correct column order
nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count', 'AffiliationId', 'female_president', 'control', 'region']]

# ------------------------------------------------------------------------------ 
# Build the aggregated edges DataFrame
# ------------------------------------------------------------------------------ 
edges_df = pd.DataFrame(edges_list)

# Ensure the edges_df has the columns in the desired order
edges_df = edges_df[['Source', 'Target', 'Type', 'Weight', 'Year']]

# ------------------------------------------------------------------------------ 
# Save the aggregated nodes and edges DataFrames to CSV files
# ------------------------------------------------------------------------------ 
aggregated_nodes_path = f"{absolute_path}{board_dataframes}aggregated_nodes.csv"
aggregated_edges_path = f"{absolute_path}{board_dataframes}aggregated_edges.csv"

nodes_df.to_csv(aggregated_nodes_path, index=False)
edges_df.to_csv(aggregated_edges_path, index=False)