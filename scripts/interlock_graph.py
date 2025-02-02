import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re

# =============================================================================
# File paths (adjust these as needed)
# =============================================================================
absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
board_dataframes = "board_dataframes\\"

# =============================================================================
# Years to process
# =============================================================================
years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

# =============================================================================
# Store total interlocks per year (for plotting later)
# =============================================================================
yearly_interlock_counts = []

# =============================================================================
# Helper Functions
# =============================================================================
def remove_non_samples(df):
    """Filter the DataFrame to include only rows with PrimarySample == True."""
    return df[df['PrimarySample'] == True]

def clean_name(raw_name):
    """
    Clean and canonicalize a board member's name by:
      - Removing specified title substrings (case-insensitive)
      - Removing punctuation and extra whitespace
      - Converting to title case.
    """
    substrings_to_remove = [
        "Rev.", "SJ", "Sister", "Brother", "Father", "OP", "The Very",
        "Sr.", "O.P.", "Very Rev.", "Br.", "Dr.", "Md.", "S.J.", "Very Rev",
        "M.D.", "O.P", "S.J", "J.R", "Jr.", "Jr ", "III"
    ]
    for title in substrings_to_remove:
        raw_name = re.sub(r'\b' + re.escape(title) + r'\b', '', raw_name, flags=re.IGNORECASE)
    # Remove punctuation and extra whitespace
    raw_name = re.sub(r'[^\w\s]', '', raw_name)
    return " ".join(raw_name.split()).title()

def group_institutions_by_membership(institution_to_members, threshold):
    """
    Given a dictionary: {institution -> set_of_members}, return a list of groups
    (lists) of institutions that have an overlap (intersection/size of smaller board)
    greater than or equal to threshold.
    """
    institutions = list(institution_to_members.keys())
    n = len(institutions)
    
    # Build an adjacency list for institutions meeting the threshold.
    adjacency = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            inst_i, inst_j = institutions[i], institutions[j]
            members_i = institution_to_members[inst_i]
            members_j = institution_to_members[inst_j]
            if not members_i or not members_j:
                continue
            overlap_ratio = len(members_i.intersection(members_j)) / min(len(members_i), len(members_j))
            if overlap_ratio >= threshold:
                adjacency[inst_i].append(inst_j)
                adjacency[inst_j].append(inst_i)
    
    # Find connected components via DFS.
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
                    stack.extend(adjacency[current])
            groups.append(sorted(group))
    
    # Only return groups with more than one institution.
    return [g for g in groups if len(g) > 1]

# =============================================================================
# Process Each Year
# =============================================================================
edge_id_counter = 1  # (Not used for further processing, only for demonstration.)
for year in years:
    print(f"Processing year: {year}")
    
    # Reinitialize board membership dictionary for this year so that interlocks are counted only within the same year.
    board_member_dict = defaultdict(set)
    
    # Load data for the year.
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"
    
    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)
    
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    if boards_df.empty and double_boards_df.empty:
        print(f"  No data after filtering for {year}. Skipping.")
        continue
    
    # Build mapping: institution -> set of board member names (using raw names)
    institution_to_members = defaultdict(set)
    for _, row in boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    for _, row in double_boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    
    # Identify "identical" board groups (using a threshold of, e.g., 0.4)
    threshold = 0.5
    identical_board_groups = group_institutions_by_membership(institution_to_members, threshold)
    print(f"  Found {len(identical_board_groups)} identical board group(s) with threshold={threshold}.")
    for i, g in enumerate(identical_board_groups, start=1):
        print(f"    Group {i}: {g}")
    
    # Map each institution (if any) to its group index.
    institution_to_group = {}
    for idx, group in enumerate(identical_board_groups):
        for inst in group:
            institution_to_group[inst] = idx

    def is_same_group(inst1, inst2):
        """Return True if both institutions are in the same identical board group."""
        return (inst1 in institution_to_group and inst2 in institution_to_group and 
                institution_to_group[inst1] == institution_to_group[inst2])
    
    # Count interlocks.
    total_interlocks = 0
    excluded_interlocks_count = 0
    
    # For each board member (using the cleaned name), record the set of institutions they belong to.
    for _, row in pd.concat([boards_df, double_boards_df]).iterrows():
        # Use the cleaned name for consistency.
        person_name = clean_name(row['Name'])
        institution = row['Institution']
        
        # For each institution this person already serves on, count an interlock.
        for prev_inst in board_member_dict[person_name]:
            if prev_inst == institution:
                continue
            if is_same_group(prev_inst, institution):
                excluded_interlocks_count += 1
            else:
                total_interlocks += 1
        board_member_dict[person_name].add(institution)
    
    if total_interlocks + excluded_interlocks_count > 0:
        proportion_excluded = (excluded_interlocks_count / (total_interlocks + excluded_interlocks_count)) * 100
    else:
        proportion_excluded = 0.0
    print(f"  Total interlocks: {total_interlocks}, Excluded: {excluded_interlocks_count} ({proportion_excluded:.2f}%)")
    
    yearly_interlock_counts.append((int(year), total_interlocks))

# Create a DataFrame and visualize the yearly interlock counts.
df_interlocks = pd.DataFrame(yearly_interlock_counts, columns=['Year', 'TotalInterlocks'])
df_interlocks.sort_values('Year', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df_interlocks['Year'], df_interlocks['TotalInterlocks'], marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#1f77b4')
plt.xlabel('Year', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Number of Interlocks', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Interlocks in University Boards Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xticks(df_interlocks['Year'], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
for x, y in zip(df_interlocks['Year'], df_interlocks['TotalInterlocks']):
    plt.text(x, y, f'{y}', fontsize=12, ha='center', va='bottom', fontweight='bold')
plt.ylim(0, df_interlocks['TotalInterlocks'].max() * 1.1)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx

# -- File paths (adjust these as needed) --
absolute_path = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
board_dataframes = "board_dataframes\\"

# -- Years you want to process --
years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

# -- Store total interlocks per year --
yearly_interlock_counts = []
# -- Store average path lengths per year --
yearly_avg_path_lengths = []

# 1) Helper function to remove non-sample rows
def remove_non_samples(df):
    """ Filter the DataFrame to include only rows with PrimarySample == True. """
    return df[df['PrimarySample'] == True]

# 2) Helper function to group institutions by membership overlap
def group_institutions_by_membership(institution_to_members, threshold):
    """
    Given a dictionary: {institution -> set_of_members}, return a list of groups
    (lists) of institutions that have >= `threshold` overlap in membership.
    
    overlap_ratio = (size of intersection) / (size of smaller board)
    """
    institutions = list(institution_to_members.keys())
    n = len(institutions)

    # Build adjacency for institutions with >= threshold overlap
    adjacency = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            inst_i = institutions[i]
            inst_j = institutions[j]
            members_i = institution_to_members[inst_i]
            members_j = institution_to_members[inst_j]
            if len(members_i) == 0 or len(members_j) == 0:
                continue
            overlap_ratio = len(members_i.intersection(members_j)) / min(len(members_i), len(members_j))
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
                    stack.extend(adjacency[current])
            groups.append(sorted(group))
    # Return only groups of size > 1
    return [g for g in groups if len(g) > 1]

# 3) Process each year
edge_id_counter = 1  # (Not used for our counting, just kept for consistency.)

for year in years:
    print(f"Processing year: {year}")

    # -- Read data --
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"

    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)

    # -- Filter out non-samples --
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)

    if boards_df.empty and double_boards_df.empty:
        print(f"  No data after filtering for {year}. Skipping.")
        continue

    # -- Build mapping: institution -> set of members --
    institution_to_members = defaultdict(set)
    for _, row in boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    for _, row in double_boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])

    # -- Find "identical" boards based on membership overlap --
    # (Here threshold is 0.4; adjust as desired)
    threshold = 0.4
    identical_board_groups = group_institutions_by_membership(institution_to_members, threshold)
    print(f"  Found {len(identical_board_groups)} identical board group(s) with threshold={threshold}.")

    # -- Map each institution to its group index --
    institution_to_group = {}
    for idx, group in enumerate(identical_board_groups):
        for inst in group:
            institution_to_group[inst] = idx

    def is_same_group(inst1, inst2):
        """ Return True if both institutions are in the same identical board group. """
        return (inst1 in institution_to_group and inst2 in institution_to_group and 
                institution_to_group[inst1] == institution_to_group[inst2])

    # -- Count interlocks --
    total_interlocks = 0
    excluded_interlocks_count = 0

    # Per-person set of institutions they belong to.
    board_member_dict = defaultdict(set)

    # For each row, check if the board member connects two institutions.
    for _, row in pd.concat([boards_df, double_boards_df]).iterrows():
        person_name = row['Name']
        institution = row['Institution']
        for prev_inst in board_member_dict[person_name]:
            if prev_inst == institution:
                continue
            if is_same_group(prev_inst, institution):
                excluded_interlocks_count += 1
            else:
                total_interlocks += 1
        board_member_dict[person_name].add(institution)

    # -- Print summary for this year --
    if total_interlocks + excluded_interlocks_count > 0:
        proportion_excluded = (excluded_interlocks_count / (total_interlocks + excluded_interlocks_count)) * 100
    else:
        proportion_excluded = 0.0
    print(f"  Total interlocks: {total_interlocks}, Excluded: {excluded_interlocks_count} ({proportion_excluded:.2f}%)")
    yearly_interlock_counts.append((int(year), total_interlocks))

    # ---------------------------
    # Build the Graph (using the interlock counts above)
    # ---------------------------
    # Create an undirected graph from the interlock counts.
    # (For this simple script, we only count interlocks per year as above.)
    G = nx.Graph()
    for person, institutions in board_member_dict.items():
        insts = list(institutions)
        # For each unique pair:
        for i in range(len(insts)):
            for j in range(i+1, len(insts)):
                # Only add edge if not in the same identical board group.
                if not is_same_group(insts[i], insts[j]):
                    G.add_edge(insts[i], insts[j])
    # Compute average path length.
    # If G is not connected, we compute the average shortest path length on its largest connected component.
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph)
        else:
            avg_path_length = float('nan')
    yearly_avg_path_lengths.append((int(year), avg_path_length))
    print(f"  Average path length (largest CC if disconnected): {avg_path_length:.2f}")

# 4) Create a DataFrame and visualize interlocks over time.
df_interlocks = pd.DataFrame(yearly_interlock_counts, columns=['Year', 'TotalInterlocks'])
df_interlocks.sort_values('Year', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df_interlocks['Year'], df_interlocks['TotalInterlocks'], marker='o', linestyle='-', linewidth=2.5, markersize=8, color='#1f77b4')
plt.xlabel('Year', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Number of Interlocks', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Interlocks in University Boards Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xticks(df_interlocks['Year'], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
for x, y in zip(df_interlocks['Year'], df_interlocks['TotalInterlocks']):
    plt.text(x, y, f'{y}', fontsize=12, ha='center', va='bottom', fontweight='bold')
plt.ylim(0, 115)
plt.tight_layout()
plt.show()

# 5) Create a DataFrame and visualize average path length over time.
df_avg_path = pd.DataFrame(yearly_avg_path_lengths, columns=['Year', 'AvgPathLength'])
df_avg_path.sort_values('Year', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df_avg_path['Year'], df_avg_path['AvgPathLength'], marker='s', linestyle='-', linewidth=2.5, markersize=8, color='orange')
plt.xlabel('Year', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Average Path Length', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Average Path Length in University Board Networks Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xticks(df_avg_path['Year'], fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
for x, y in zip(df_avg_path['Year'], df_avg_path['AvgPathLength']):
    plt.text(x, y, f'{y:.2f}', fontsize=12, ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()
