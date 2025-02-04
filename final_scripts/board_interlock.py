import pandas as pd
from collections import defaultdict
import networkx as nx
import re

absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts = "scripts\\"
board_dataframes = "board_dataframes\\"
yearly_interlocks = "yearly_interlocks\\"

years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]
#combined years nodes (for the statistics, we separate the graphs by year, but for images of the network, used the whole graph because the nodes are in the same position)
all_nodes_df = pd.DataFrame()  
all_nodes_list = []        
all_interlocks_list = []   


def remove_non_samples(df):
    return df[df['PrimarySample'] == True]

board_statistics_path = f"{absolute_path}{altered_dataframes}sample_board_statistics.csv"
board_statistics_df = pd.read_csv(board_statistics_path)


def group_institutions_by_membership(institution_to_members: dict, threshold: float) -> list:
    """
    Given a dictionary {institution -> set_of_members}, return a list of groups (lists)
    of institutions that have an overlap ratio (intersection divided by the smaller board's size)
    >= threshold.

    Used to not inflate the number of interlocks when we observe interlocks between two state schools that 
    operate under the same board. 
    Defining the correct threshold is difficult, should probably be lower, as it's not counting boards that shouldnt be
    """
    
    institutions = list(institution_to_members.keys())
    n = len(institutions)
    adjacency = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            inst_i, inst_j = institutions[i], institutions[j]
            members_i = institution_to_members[inst_i]
            members_j = institution_to_members[inst_j]
            if not members_i or not members_j:
                continue
            intersection_size = len(members_i.intersection(members_j))
            smaller_board_size = min(len(members_i), len(members_j))
            overlap_ratio = intersection_size / smaller_board_size
            if overlap_ratio >= threshold:
                adjacency[inst_i].append(inst_j)
                adjacency[inst_j].append(inst_i)
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
    return [g for g in groups if len(g) > 1]

def lookup_female_president(row, year):
    """
    Look up the 'female_president' boolean from board_statistics_df for a given row and year.
    
    The function first filters board_statistics_df by the specified year, then attempts to
    match either on 'AffiliationId' (using row['AffiliationId']) or on 'Institution' (using row['Id']).
    It returns the boolean value from the 'female_president' column.
    
    Raises:
        ValueError: If no matching record is found.
    """
    filtered_df = board_statistics_df[board_statistics_df['Year'] == int(year)]
    if row["Id"] in filtered_df["Institution"].values:
        matching_row = filtered_df[filtered_df["Institution"] == row["Id"]].iloc[0]
        if matching_row['female_president'] == True:
            return True
        else:
            return False
    elif row["AffiliationId"] in filtered_df["AffiliationId"].values:
        matching_row = filtered_df[filtered_df["AffiliationId"] == row["AffiliationId"]].iloc[0]
        if matching_row['female_president'] == True:
            return True
        else:
            return False
    return "Unknown"

def lookup_column(row, column_name):
    """
    look up any column (e.g., 'control', 'region') in board_statistics_df
    using AffiliationId or Institution name.
    """
    matching_rows = board_statistics_df[
        board_statistics_df['AffiliationId'] == row['AffiliationId']
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]
    matching_rows = board_statistics_df[
        board_statistics_df['Institution'] == row['Id']
    ]
    if not matching_rows.empty:
        return matching_rows[column_name].mode()[0]
    return 'unknown'

# ------------------------------------------------------------------------------
# Name Cleaning and Canonicalization (without fuzzy matching)
# ------------------------------------------------------------------------------
substrings_to_remove = [
    "Rev.", "SJ", "Sister", "Brother", "Father", "OP", "The Very",
    "Sr.", "O.P.", "Very Rev.", "Br.", "Dr.", "Md.", "S.J.", "Very Rev",
    "M.D.", "O.P", "S.J", "J.R", "Jr.", "Jr ", "III"
]

def clean_name(raw_name: str) -> str:
    """
    Clean and canonicalize a board member's name by:
      - Removing specified title substrings (case-insensitive)
      - Removing punctuation and extra whitespace
      - Converting to title case.
    """
    for title in substrings_to_remove:
        title_clean = title.strip()
        raw_name = re.sub(r'\b' + re.escape(title_clean) + r'\b', '', raw_name, flags=re.IGNORECASE)
    raw_name = re.sub(r'[^\w\s]', '', raw_name)
    cleaned_name = " ".join(raw_name.split())
    return cleaned_name.title()

# Global edge counter so that edge IDs remain unique across years (when building the combined network)
edge_id_counter = 1


'''
Compute network by year for network and for centrality measures in the regression
Compute entire network for global network
'''
for year in years:
    print(f"Processing year: {year}")
    board_member_dict = defaultdict(set)
    

    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"
    
    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    #Map board members to their institution
    institution_to_members = defaultdict(set)
    for _, row in boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    for _, row in double_boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    
    #Board sizes for edge weights
    board_sizes = {inst: len(members) for inst, members in institution_to_members.items()}
    
    '''
    #Identify identical board groups for state systems (not going to right now)
    threshold = 0.3
    identical_board_groups = group_institutions_by_membership(institution_to_members, threshold)
    print(f"  Found {len(identical_board_groups)} identical board group(s) with threshold={threshold}.")
    for i, g in enumerate(identical_board_groups, start=1):
        print(f"    Group {i}: {g}")
    
    # Map each institution to its group index.
    institution_to_group = {}
    for idx, group in enumerate(identical_board_groups):
        for inst in group:
            institution_to_group[inst] = idx

    def is_same_group(inst1, inst2):
        """Return True if both institutions belong to the same identical board group."""
        return (inst1 in institution_to_group and inst2 in institution_to_group and 
                institution_to_group[inst1] == institution_to_group[inst2])
    '''
    


    '''
    Create the network for the year that's being iterated over
    '''
    edge_accum = {}  # Key: tuple(sorted([inst1, inst2])); Value: dict with keys: Id, Source, Target, Weight, Year
    year_nodes_dict = defaultdict(lambda: {'Interlock_Count': 0, 'AffiliationId': None})
    
    # Context for tracking counts and generating unique edge IDs.
    context = {
        'excluded_interlocks_count': 0,
        'total_interlocks': 0,
        'edge_id_counter': edge_id_counter
    }
    # To avoid duplicate contributions from the same board member within a year.
    created_interlocks = defaultdict(set)
    
    def process_board_row(row, ctx):
        """
        Process a board row (using raw names) to update interlocks.
        Skip the row if "vacant" is in the board member's name.
        For each board member row, for every previous institution that this person has served on,
        create (or update) an edge whose weight is computed as:
            w = 1 / min(board_sizes[prev_institution], board_sizes[current_institution])
        The weight is added to the edge for this year (edges are stored per year).
        """

        #some names are listed as vacant, cannot count that as an interlock
        if "vacant" in row['Name'].lower():
            return
        
        name = row['Name'] 
        name = clean_name(row['Name'])
        institution = row['Institution']
        affiliation_id = row['AffiliationId']
        
        for prev_institution in board_member_dict[name]:
            if prev_institution == institution:
                continue
            # if is_same_group(prev_institution, institution):
            #     ctx['excluded_interlocks_count'] += 1
            #     continue
            pair = tuple(sorted([prev_institution, institution]))
            # Avoid duplicate contributions from the same board member.
            if pair in created_interlocks[name]:
                continue
            created_interlocks[name].add(pair)
            
            # Calculate weight contribution as 1 divided by the size of the smaller board.
            w = 1 
            
            # Since edges are calculated per year, if this pair already exists, update its weight.
            if pair in edge_accum:
                edge_accum[pair]['Weight'] += w
            else:
                edge_id = f"e{ctx['edge_id_counter']}"
                ctx['edge_id_counter'] += 1
                edge_accum[pair] = {
                    'Id': edge_id,
                    'Source': pair[0],
                    'Target': pair[1],
                    'Type': 'Undirected',
                    'Weight': w,
                    'Year': year
                }
            ctx['total_interlocks'] += 1
            # Sum the contributions for the nodes.
            year_nodes_dict[prev_institution]['Interlock_Count'] += w
            year_nodes_dict[institution]['Interlock_Count'] += w
        
        board_member_dict[name].add(institution)
        if year_nodes_dict[institution]['AffiliationId'] is None:
            year_nodes_dict[institution]['AffiliationId'] = affiliation_id

    for _, row in boards_df.iterrows():
        process_board_row(row, context)
    for _, row in double_boards_df.iterrows():
        process_board_row(row, context)
    
    excluded_interlocks_count = context['excluded_interlocks_count']
    total_interlocks = context['total_interlocks']
    edge_id_counter = context['edge_id_counter']
    
    #add weights to the edges
    for pair, edge in edge_accum.items():
        shared_members = edge['Weight']
        total_unique = board_sizes.get(pair[0], 0) + board_sizes.get(pair[1], 0) - shared_members
        edge['Weight'] = shared_members / total_unique if total_unique > 0 else 0

    #nodes df for the current year
    nodes_df = pd.DataFrame(
        [(inst, data['Interlock_Count'], data['AffiliationId'])
         for inst, data in year_nodes_dict.items()],
        columns=['Id', 'Interlock_Count', 'AffiliationId']
    )
    nodes_df['Label'] = nodes_df['Id']
    
    #lookup extra attributes from board_statistics_df.
    nodes_df['female_president'] = nodes_df.apply(lambda row: lookup_female_president(row, year), axis=1)
    nodes_df['control'] = nodes_df.apply(lambda row: lookup_column(row, 'control'), axis=1)
    nodes_df['region'] = nodes_df.apply(lambda row: lookup_column(row, 'region'), axis=1)
    
    nodes_df = nodes_df[['Id', 'Label', 'Interlock_Count', 'AffiliationId',
                         'female_president', 'control', 'region']]
    
    # Convert the accumulated edges dictionary to a DataFrame.
    edges_df = pd.DataFrame(list(edge_accum.values()))
    edges_df = edges_df[['Id', 'Source', 'Target', 'Type', 'Weight', 'Year']]

    print(f"Institutions: {len(year_nodes_dict)} | Total interlocks: {total_interlocks}")
    
    '''
    #for the thresholding (not used right now)
    if total_interlocks + excluded_interlocks_count > 0:
        proportion_excluded = (excluded_interlocks_count / (total_interlocks + excluded_interlocks_count)) * 100
    else:
        proportion_excluded = 0.0
    print(f"  Institutions: {len(year_nodes_dict)} | Total interlocks: {total_interlocks} | Excluded interlocks: {excluded_interlocks_count} ({proportion_excluded:.2f}%)")
    '''
    
    #make network and get centrality measures for the current year
    G = nx.Graph()
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
    for _, edge_row in edges_df.iterrows():
        G.add_edge(edge_row["Source"], edge_row["Target"], weight=edge_row.get("Weight", 1))
    
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
    
    #optionally save these to csv and gephi, not really using them currently outside of using them for the centrality measures though
    # year_nodes_csv_path = f"{absolute_path}{yearly_interlocks}{year}_nodes.csv"
    # year_edges_csv_path = f"{absolute_path}{yearly_interlocks}{year}_edges.csv"
    # year_gexf_path = f"{absolute_path}{graphs}{year}_graph.gexf"
    
    # nodes_df.to_csv(year_nodes_csv_path, index=False)
    # edges_df.to_csv(year_edges_csv_path, index=False)
    # nx.write_gexf(G, year_gexf_path)
    
    # Add a 'Year' column for merging later.
    nodes_df["Year"] = year
    all_nodes_list.append(nodes_df)
    all_nodes_df = pd.concat([all_nodes_df, nodes_df], ignore_index=True)
    
    if not edges_df.empty:
        all_interlocks_list.append(edges_df)

#Merge the centrality measures from the yearly networks into the university_board_statistics df
board_statistics_df['Year'] = board_statistics_df['Year'].astype(str)
merged_df = board_statistics_df.merge(
    all_nodes_df[["Year", "AffiliationId", "betweenness", "closeness", "eigenvector", "degree", "clustering"]],
    on=["Year", "AffiliationId"],
    how="left",
    suffixes=("", "_new")
)
for col in ["betweenness", "closeness", "eigenvector", "degree", "clustering", "Board_Size"]:
    if f"{col}_new" in merged_df.columns:
        merged_df[col] = merged_df[f"{col}_new"]
        merged_df.drop(columns=[f"{col}_new"], inplace=True)
centrality_columns = ["betweenness", "closeness", "eigenvector", "degree", "clustering"]

#write to csv and fill in empty values (when the inst wasn't in the network, the centrality measures are nan)
merged_df[centrality_columns] = merged_df[centrality_columns].fillna(0)
merged_df.to_csv(f'{absolute_path}{altered_dataframes}sample_board_statistics.csv', index=False)

#create a global network for gephi
combined_nodes_df = (
    all_nodes_df
    .groupby("Id", as_index=False)
    .agg({
        "Interlock_Count": "sum",
        "AffiliationId": "first",
        "Label": "first",
        "female_president": "first",
        "control": "first",
        "region": "first"
    })
)
all_nodes_csv_path = f"{absolute_path}{yearly_interlocks}all_nodes.csv"
combined_nodes_df.to_csv(all_nodes_csv_path, index=False)

#For edges: combine all yearly edge DataFrames (each edge retains its 'Year' attribute).
all_interlocks_df = pd.concat(all_interlocks_list, ignore_index=True)
all_edges_csv_path = f"{absolute_path}{yearly_interlocks}all_edges.csv"
all_interlocks_df.to_csv(all_edges_csv_path, index=False)

print(f"Saved combined nodes to: {all_nodes_csv_path}")
print(f"Saved combined edges to: {all_edges_csv_path}")
