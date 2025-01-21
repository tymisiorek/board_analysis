import pandas as pd
from collections import defaultdict

absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"

years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

def remove_non_samples(df):
    df = df[df['PrimarySample'] == True]
    return df

# This list holds all edge information (interlocks) across all years
edges_list = []

# A dictionary tracking each institution's cumulative data (including interlock count)
nodes_dict = defaultdict(lambda: {
    'Interlock_Count': 0,
    'AffiliationId': None,
})

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
            #DFS to get all institutions connected to 'inst'
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
    
    # Filter out any "groups" that contain just one institution unless you want them
    # We'll keep them only if they have at least 2 institutions
    filtered_groups = [g for g in groups if len(g) > 1]
    return filtered_groups

# A dictionary to store the identical board groups for each year
year_to_identical_groups = dict()

# ------------------------------------------------------------------------------
# Process each year and accumulate edges and nodes
# ------------------------------------------------------------------------------
for year in years:
    print(f"Processing interlocks for: {year}")

    # Read the boards for this year
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    boards_df = pd.read_csv(boards_path)

    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"
    double_boards_df = pd.read_csv(double_boards_path)
    
    # Remove non-sample schools
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)

    # Create a fresh dictionary for this year only
    # Maps board members (by name) -> set of institutions they've served on this year
    board_member_dict_year = defaultdict(set)

    # Build also a mapping: institution -> set of board member names (for overlap checks)
    institution_to_members_year = defaultdict(set)

    # -- Process normal boards --
    for index, row in boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']  # Ensure 'AffiliationId' is present

        # For the network logic (unchanged)
        for previous_institution in board_member_dict_year[name]:
            if previous_institution != institution:
                edges_list.append({
                    'Source': previous_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1,
                    'Year': year
                })
                nodes_dict[previous_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1

        board_member_dict_year[name].add(institution)

        # Track institution membership (for overlap detection)
        institution_to_members_year[institution].add(name)

        if nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id

    # -- Process double boards --
    for index, row in double_boards_df.iterrows():
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']

        for previous_institution in board_member_dict_year[name]:
            if previous_institution != institution:
                edges_list.append({
                    'Source': previous_institution,
                    'Target': institution,
                    'Type': 'Undirected',
                    'Weight': 1,
                    'Year': year
                })
                nodes_dict[previous_institution]['Interlock_Count'] += 1
                nodes_dict[institution]['Interlock_Count'] += 1

        board_member_dict_year[name].add(institution)

        # Track institution membership (for overlap detection)
        institution_to_members_year[institution].add(name)

        if nodes_dict[institution]['AffiliationId'] is None:
            nodes_dict[institution]['AffiliationId'] = affiliation_id

    # --------------------------------------------------------------------------
    # Identify institutions that share >= threshold% of the same board
    # --------------------------------------------------------------------------
    identical_board_groups = group_institutions_by_membership(institution_to_members_year, threshold=0.7)
    year_to_identical_groups[year] = identical_board_groups



filtered_nodes_dict = {key: value for key, value in nodes_dict.items() if value['Interlock_Count'] > 0}
nodes_df = pd.DataFrame([
    (key, value['Interlock_Count'], value['AffiliationId'])
    for key, value in filtered_nodes_dict.items()
], columns=['Id', 'Interlock_Count', 'AffiliationId'])

nodes_df['Label'] = nodes_df['Id']

def lookup_female_president(row):
    matching_rows = board_statistics_df[
        (board_statistics_df['AffiliationId'] == row['AffiliationId'])
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]
    # Fallback: match by institution name
    matching_rows = board_statistics_df[
        (board_statistics_df['Institution'] == row['Id'])
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

nodes_df['female_president'] = nodes_df.apply(lookup_female_president, axis=1).fillna('unknown')
nodes_df['control'] = nodes_df.apply(lambda row: lookup_column(row, 'control'), axis=1)
nodes_df['region'] = nodes_df.apply(lambda row: lookup_column(row, 'region'), axis=1)

nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count', 'AffiliationId',
                     'female_president', 'control', 'region']]

edges_df = pd.DataFrame(edges_list)
edges_df = edges_df[['Source', 'Target', 'Type', 'Weight', 'Year']]

#institutions with identical board (state schools)
print("\nINSTITUTION GROUPS WITH ≥ threshold% BOARD OVERLAP (BY YEAR)")
for year in sorted(year_to_identical_groups.keys()):
    groups = year_to_identical_groups[year]
    if not groups:
        continue
    print(f"\nYear: {year} => {len(groups)} group(s)")
    for i, group in enumerate(groups, 1):
        print(f"  Group {i}: {group}")

aggregated_nodes_path = f"{absolute_path}{board_dataframes}aggregated_nodes.csv"
aggregated_edges_path = f"{absolute_path}{board_dataframes}aggregated_edges.csv"

nodes_df.to_csv(aggregated_nodes_path, index=False)
edges_df.to_csv(aggregated_edges_path, index=False)
