import pandas as pd
from collections import defaultdict

# Define absolute path and subdirectories
absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"

# List of years to process (ensure they are in chronological order)
years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

def remove_non_samples(df):
    """
    Filter the DataFrame to include only primary samples.
    """
    return df[df['PrimarySample'] == True]

# Initialize lists and dictionaries
all_edges = []          # List to hold all edges across all years
filtered_edges = []     # List to hold edges excluding identical board interlocks

# Path to board statistics (needed to look up female_president info)
board_statistics_path = f"{absolute_path}{altered_dataframes}sample_board_statistics.csv"
board_statistics_df = pd.read_csv(board_statistics_path)

# A helper function to compute groupings of institutions that share >= threshold% membership
def group_institutions_by_membership(institution_to_members, threshold):
    """
    Given a dictionary: institution -> set_of_members,
    return a list of groups (lists) of institutions,
    where each group has boards with >= `threshold` overlap.

    Overlap is computed as:
        overlap = (size of intersection) / (size of smaller board)
    """
    institutions = list(institution_to_members.keys())
    n = len(institutions)
    
    # Build graph of institutions that have >= threshold overlap
    adjacency = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            inst_i, inst_j = institutions[i], institutions[j]
            members_i = institution_to_members[inst_i]
            members_j = institution_to_members[inst_j]
            
            # Avoid zero-member boards if they exist
            size_i = len(members_i)
            size_j = len(members_j)
            if size_i == 0 or size_j == 0:
                continue
            
            intersection_size = len(members_i.intersection(members_j))
            smaller_board_size = min(size_i, size_j)
            overlap_ratio = intersection_size / smaller_board_size
            
            if overlap_ratio >= threshold:
                # Link them in adjacency
                adjacency[inst_i].append(inst_j)
                adjacency[inst_j].append(inst_i)
    
    # Find connected components with DFS
    visited = set()
    groups = []
    
    for inst in institutions:
        if inst not in visited:
            # DFS to get all institutions connected to 'inst'
            stack = [inst]
            group = []
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    group.append(current)
                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            groups.append(sorted(group))
    
    # Filter out any "groups" that contain just one institution
    filtered_groups = [g for g in groups if len(g) > 1]
    return filtered_groups

# Dictionary to store the identical board groups for each year
year_to_identical_groups = dict()

# ------------------------------------------------------------------------------
# Phase 1: Identify Identical Board Groups for Each Year
# ------------------------------------------------------------------------------
for year in years:
    print(f"\nProcessing identical board groups for: {year}")

    # Read the boards for this year
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)

    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"
    double_boards_df = pd.read_csv(double_boards_path)
    
    # Remove non-sample schools
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    # Build a mapping: institution -> set of board member names
    institution_to_members_year = defaultdict(set)
    
    for index, row in boards_df.iterrows():
        institution = row['Institution']
        name = row['Name']
        institution_to_members_year[institution].add(name)
    
    for index, row in double_boards_df.iterrows():
        institution = row['Institution']
        name = row['Name']
        institution_to_members_year[institution].add(name)
    
    # Identify identical board groups with threshold=0.75
    # Adjust the threshold here as per your analysis requirements
    identical_board_groups = group_institutions_by_membership(institution_to_members_year, threshold=1)
    year_to_identical_groups[year] = identical_board_groups

    # Print the identified groups
    if identical_board_groups:
        print(f"Identical board groups for {year}: {len(identical_board_groups)} group(s)")
        for i, group in enumerate(identical_board_groups, 1):
            print(f"  Group {i}: {group}")
    else:
        print(f"No identical board groups found for {year}.")

# ------------------------------------------------------------------------------
# Phase 2: Process Interlocks Excluding Identical Board Groups
# ------------------------------------------------------------------------------
# Initialize a dictionary to track nodes
nodes_dict = defaultdict(lambda: {
    'Interlock_Count': 0,
    'AffiliationId': None,
})

# Initialize a set to track institutions that have already been involved in interlocks
seen_institutions = set()

for year in years:
    print(f"\nProcessing interlocks for: {year}")
    
    # Read the boards for this year
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)

    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"
    double_boards_df = pd.read_csv(double_boards_path)
    
    # Remove non-sample schools
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    # Count the number of unique institutions for this year
    unique_institutions = set(boards_df['Institution'].unique()).union(set(double_boards_df['Institution'].unique()))
    num_institutions = len(unique_institutions)
    
    # Initialize interlock counters for the year
    total_interlocks = 0
    excluded_interlocks_count = 0  # Count of interlocks within identical board groups
    
    # Get identical board groups for the current year
    identical_board_groups = year_to_identical_groups.get(year, [])
    
    # Create a mapping: institution -> group index
    institution_to_group = {}
    for group_index, group in enumerate(identical_board_groups):
        for institution in group:
            institution_to_group[institution] = group_index
    
    # Create a mapping: board member name -> set of institutions they've served on this year
    board_member_dict_year = defaultdict(set)
    
    # Function to determine if two institutions are in the same identical board group
    def is_same_group(inst1, inst2):
        return (inst1 in institution_to_group and 
                inst2 in institution_to_group and 
                institution_to_group[inst1] == institution_to_group[inst2])
    
    # Process normal boards
    for index, row in boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']
        
        # Iterate over institutions this board member has previously served
        for prev_institution in board_member_dict_year[name]:
            if prev_institution != institution:
                # Check if both institutions are in the same identical board group
                if is_same_group(prev_institution, institution):
                    excluded_interlocks_count += 1  # Increment excluded interlock count
                    continue  # Skip recording this interlock
                
                # Create an unordered pair for uniqueness
                pair = tuple(sorted([prev_institution, institution]))
                
                # Record the edge
                edge = {
                    'Source': prev_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1,
                    'Year': year
                }
                all_edges.append(edge)
                filtered_edges.append(edge)
                total_interlocks += 1
                
                # Update nodes_dict
                nodes_dict[prev_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1
                
        # Add the institution to the board member's set
        board_member_dict_year[name].add(institution)
        
        # Update AffiliationId if not already set
        if nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id

    # Process double boards
    for index, row in double_boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']
        
        # Iterate over institutions this board member has previously served
        for prev_institution in board_member_dict_year[name]:
            if prev_institution != institution:
                # Check if both institutions are in the same identical board group
                if is_same_group(prev_institution, institution):
                    excluded_interlocks_count += 1  # Increment excluded interlock count
                    continue  # Skip recording this interlock
                
                # Create an unordered pair for uniqueness
                pair = tuple(sorted([prev_institution, institution]))
                
                # Record the edge
                edge = {
                    'Source': prev_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1,
                    'Year': year
                }
                all_edges.append(edge)
                filtered_edges.append(edge)
                total_interlocks += 1
                
                # Update nodes_dict
                nodes_dict[prev_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1
                
        # Add the institution to the board member's set
        board_member_dict_year[name].add(institution)
        
        # Update AffiliationId if not already set
        if nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id
    
    # ------------------------------------------------------------------------------
    # Tracking and Printing New Institutions Per Year
    # ------------------------------------------------------------------------------
    # Collect institutions involved in interlocks this year from filtered_edges
    institutions_this_year = set()
    for edge in filtered_edges:
        if edge['Year'] == year:
            institutions_this_year.add(edge['Source'])
            institutions_this_year.add(edge['Target'])
    
    # Determine new institutions (those not seen in previous years)
    new_institutions = institutions_this_year - seen_institutions
    
    # Print new institutions if it's not the first year and there are new institutions
    if year != years[0] and new_institutions:
        print(f"New institutions added in {year}: {sorted(new_institutions)}")
    
    # Update seen_institutions with institutions from this year
    seen_institutions.update(institutions_this_year)


# ------------------------------------------------------------------------------
# Building the Nodes DataFrame
# ------------------------------------------------------------------------------
filtered_nodes_dict = {key: value for key, value in nodes_dict.items() if value['Interlock_Count'] > 0}
nodes_df = pd.DataFrame([
    (key, value['Interlock_Count'], value['AffiliationId'])
    for key, value in filtered_nodes_dict.items()
], columns=['Id', 'Interlock_Count', 'AffiliationId'])

nodes_df['Label'] = nodes_df['Id']

def lookup_female_president(row, year):
    """
    Checks if the institution had a female president in the specific year being processed.
    """
    matching_rows = board_statistics_df[
        (board_statistics_df['AffiliationId'] == row['AffiliationId']) &
        (board_statistics_df['Year'] == year) 
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]

    matching_rows = board_statistics_df[
        (board_statistics_df['Institution'] == row['Id']) &
        (board_statistics_df['Year'] == year) 
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]

    return 'unknown'


def lookup_column(row, column_name):
    matching_rows = board_statistics_df[
        (board_statistics_df['AffiliationId'] == row['AffiliationId'])
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]
    # Fallback: match by Institution name
    matching_rows = board_statistics_df[
        (board_statistics_df['Institution'] == row['Id'])
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]
    return 'unknown'

nodes_df['female_president'] = nodes_df.apply(
    lambda row: lookup_female_president(row, year), axis=1
).fillna('unknown')
nodes_df['control'] = nodes_df.apply(lambda row: lookup_column(row, 'control'), axis=1)
nodes_df['region'] = nodes_df.apply(lambda row: lookup_column(row, 'region'), axis=1)

nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count', 'AffiliationId',
                     'female_president', 'control', 'region']]

# ------------------------------------------------------------------------------
# Building the Edges DataFrame
# ------------------------------------------------------------------------------
edges_df = pd.DataFrame(filtered_edges)
edges_df = edges_df[['Source', 'Target', 'Type', 'Weight', 'Year']]

# ------------------------------------------------------------------------------
# Institutions with identical board (state schools)
# ------------------------------------------------------------------------------
print("\nINSTITUTION GROUPS WITH ≥ threshold% BOARD OVERLAP (BY YEAR)")
for year in sorted(year_to_identical_groups.keys()):
    groups = year_to_identical_groups[year]
    if not groups:
        continue
    print(f"\nYear: {year} => {len(groups)} group(s)")
    for i, group in enumerate(groups, 1):
        print(f"  Group {i}: {group}")

# ------------------------------------------------------------------------------
# Saving the DataFrames to CSV
# ------------------------------------------------------------------------------
aggregated_nodes_path = f"{absolute_path}{board_dataframes}aggregated_nodes.csv"
aggregated_edges_path = f"{absolute_path}{board_dataframes}aggregated_edges.csv"

nodes_df.to_csv(aggregated_nodes_path, index=False)
edges_df.to_csv(aggregated_edges_path, index=False)
