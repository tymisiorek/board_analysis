import pandas as pd
from collections import defaultdict
import networkx as nx

absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"
yearly_interlocks = "yearly_interlocks\\"

years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]
all_nodes_df = pd.DataFrame()
all_nodes_list = []

def remove_non_samples(df):
    """
    Filter the DataFrame to include only primary samples (PrimarySample == True).
    """
    return df[df['PrimarySample'] == True]

# Load board statistics (for 'female_president', 'control', 'region', etc.)
board_statistics_path = f"{absolute_path}{altered_dataframes}sample_board_statistics.csv"
board_statistics_df = pd.read_csv(board_statistics_path)

def group_institutions_by_membership(institution_to_members, threshold):
    """
    Given a dictionary: {institution -> set_of_members}, return a list of groups
    (lists) of institutions that have >= `threshold` overlap in membership.

    overlap = (size of intersection) / (size of smaller board)
    """
    institutions = list(institution_to_members.keys())
    n = len(institutions)

    # Build adjacency list for institutions with >= threshold overlap
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
                adjacency[inst_i].append(inst_j)
                adjacency[inst_j].append(inst_i)

    # Find connected components via DFS
    visited = set()
    groups = []
    for inst in institutions:
        if inst not in visited:
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

    # Filter out groups of size 1
    return [g for g in groups if len(g) > 1]

def lookup_female_president(row):
    """
    Helper to find 'female_president' in board_statistics_df using
    AffiliationId or Institution name.
    """
    matching_rows = board_statistics_df[
        board_statistics_df['AffiliationId'] == row['AffiliationId']
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]

    # Fallback: match by institution name
    matching_rows = board_statistics_df[
        board_statistics_df['Institution'] == row['Id']
    ]
    if not matching_rows.empty:
        return matching_rows['female_president'].mode()[0]

    return 'unknown'

def lookup_column(row, column_name):
    """
    Generic helper to look up any column (like 'control', 'region') in
    board_statistics_df using AffiliationId or Institution name.
    """
    matching_rows = board_statistics_df[
        board_statistics_df['AffiliationId'] == row['AffiliationId']
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]

    # Fallback: match by institution name
    matching_rows = board_statistics_df[
        board_statistics_df['Institution'] == row['Id']
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]

    return 'unknown'

# Keep a global edge ID counter so edges across years can be unique
edge_id_counter = 1

for year in years:
    print(f"Processing year: {year}")

    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"

    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)

    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)

    if boards_df.empty and double_boards_df.empty:
        print(f"  No data after filtering for {year}. Skipping.")
        continue

    #identitical board groups
    institution_to_members = defaultdict(set)

    #doublde and single board
    for _, row in boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    for _, row in double_boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])

    #identical board threshold
    threshold = 0.9 
    identical_board_groups = group_institutions_by_membership(institution_to_members, threshold)

    print(f"  Found {len(identical_board_groups)} identical board group(s) with threshold={threshold}.")
    for i, g in enumerate(identical_board_groups, start=1):
        print(f"    Group {i}: {g}")

    #inst to group
    institution_to_group = {}
    for idx, group in enumerate(identical_board_groups):
        for inst in group:
            institution_to_group[inst] = idx

    def is_same_group(inst1, inst2):
        """
        Check if both institutions belong to the same 'identical board group.'
        """
        return (inst1 in institution_to_group 
                and inst2 in institution_to_group 
                and institution_to_group[inst1] == institution_to_group[inst2])

    #build edges
    year_edges = []
    year_nodes_dict = defaultdict(lambda: {'Interlock_Count': 0, 'AffiliationId': None})

    # For storing board membership per person
    board_member_dict = defaultdict(set)

    # Use a context dict to avoid nonlocal
    context = {
        'excluded_interlocks_count': 0,
        'total_interlocks': 0,
        'edge_id_counter': edge_id_counter
    }

    def process_board_row(row, ctx):
        name = row['Name']
        institution = row['Institution']
        affiliation_id = row['AffiliationId']

        for prev_institution in board_member_dict[name]:
            if prev_institution == institution:
                continue

            # Exclude edges if they are in the same "identical board group"
            if is_same_group(prev_institution, institution):
                ctx['excluded_interlocks_count'] += 1
                continue

            # Create an edge
            pair = tuple(sorted([prev_institution, institution]))
            edge_id = f"e{ctx['edge_id_counter']}"
            ctx['edge_id_counter'] += 1

            edge = {
                'Id': edge_id,
                'Source': pair[0],
                'Target': pair[1],
                'Type': 'Undirected',
                'Weight': 1,
                'Year': year
            }
            year_edges.append(edge)
            ctx['total_interlocks'] += 1

            # Update node interlock counts
            year_nodes_dict[prev_institution]['Interlock_Count'] += 1
            year_nodes_dict[institution]['Interlock_Count'] += 1

        board_member_dict[name].add(institution)
        if year_nodes_dict[institution]['AffiliationId'] is None:
            year_nodes_dict[institution]['AffiliationId'] = affiliation_id

    # Process each row in the single and double boards
    for _, row in boards_df.iterrows():
        process_board_row(row, context)

    for _, row in double_boards_df.iterrows():
        process_board_row(row, context)

    # Retrieve updated counters
    excluded_interlocks_count = context['excluded_interlocks_count']
    total_interlocks = context['total_interlocks']
    edge_id_counter = context['edge_id_counter']  


    year_nodes_dict_filtered = {
        inst: data for inst, data in year_nodes_dict.items()
    }

    nodes_df = pd.DataFrame(
        [
            (inst, data['Interlock_Count'], data['AffiliationId'])
            for inst, data in year_nodes_dict_filtered.items()
        ],
        columns=['Id', 'Interlock_Count', 'AffiliationId']
    )
    nodes_df['Label'] = nodes_df['Id']

    # Lookup extra columns from board_statistics_df
    nodes_df['female_president'] = nodes_df.apply(lookup_female_president, axis=1)
    nodes_df['control'] = nodes_df.apply(lambda row: lookup_column(row, 'control'), axis=1)
    nodes_df['region'] = nodes_df.apply(lambda row: lookup_column(row, 'region'), axis=1)

    # Reorder
    nodes_df = nodes_df[
        ['Id', 'Label', 'Interlock_Count', 'AffiliationId',
         'female_president', 'control', 'region']
    ]

    edges_df = pd.DataFrame(year_edges)
    if not edges_df.empty:
        edges_df = edges_df[['Id', 'Source', 'Target', 'Type', 'Weight', 'Year']]


    if total_interlocks + excluded_interlocks_count > 0:
        proportion_excluded = (
            excluded_interlocks_count
            / (total_interlocks + excluded_interlocks_count)
        ) * 100
    else:
        proportion_excluded = 0.0

    print(f"  Institutions: {len(year_nodes_dict)} | "
          f"Total interlocks: {total_interlocks} | "
          f"Excluded interlocks: {excluded_interlocks_count} ({proportion_excluded:.2f}%)")

    G = nx.Graph()

    # Add nodes
    for _, node_row in nodes_df.iterrows():
        G.add_node(
            node_row["Id"],
            AffiliationId=node_row["AffiliationId"],
            Label=node_row["Label"],
            Interlock_Count=node_row["Interlock_Count"],
            female_president=node_row["female_president"],
            control=node_row["control"],
            region=node_row["region"]
        )

    # Add edges
    for _, edge_row in edges_df.iterrows():
        G.add_edge(
            edge_row["Source"],
            edge_row["Target"],
            weight=edge_row.get("Weight", 1)
        )

    # Compute centralities
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    degree_dict = dict(G.degree())
    clustering = nx.clustering(G, weight="weight")

    nodes_df["betweenness"] = nodes_df["Id"].map(betweenness)
    nodes_df["closeness"] = nodes_df["Id"].map(closeness)
    nodes_df["eigenvector"] = nodes_df["Id"].map(eigenvector)
    nodes_df["degree"] = nodes_df["Id"].map(degree_dict)
    nodes_df["clustering"] = nodes_df["Id"].map(clustering)

    year_nodes_csv_path = f"{absolute_path}{yearly_interlocks}{year}_nodes.csv"
    year_edges_csv_path = f"{absolute_path}{yearly_interlocks}{year}_edges.csv"
    year_gexf_path = f"{absolute_path}{graphs}{year}_graph.gexf"

    nodes_df.to_csv(year_nodes_csv_path, index=False)
    edges_df.to_csv(year_edges_csv_path, index=False)
    nx.write_gexf(G, year_gexf_path)

    # Convert 'Year' columns to strings for consistency
    board_statistics_df['Year'] = board_statistics_df['Year'].astype(str)

    # Add the current year to nodes_df for proper merging
    nodes_df['Year'] = year
    all_nodes_list.append(nodes_df)
    all_nodes_df = pd.concat([all_nodes_df, nodes_df], ignore_index=True)


merged_df = board_statistics_df.merge(
    all_nodes_df[["Year", "AffiliationId", "betweenness", "closeness", "eigenvector", "degree", "clustering"]],
    on = ["Year", "AffiliationId"],
    how="left"
)
merged_df.to_csv(f'{absolute_path}{altered_dataframes}sample_board_statistics.csv', index = False)