import pandas as pd
from collections import defaultdict
import re

# Set up paths.
absolute_path = r"C:\Users\tykun\OneDrive\Documents\SchoolDocs\VSCodeProjects\connectedData\board_analysis\\"
board_dataframes = "board_dataframes\\"
yearly_interlocks = "yearly_interlocks\\"

years = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]

def remove_non_samples(df):
    return df[df['PrimarySample'] == True]

# -------------------------------------------------------------------------------
# Name cleaning function (without fuzzy matching)
# -------------------------------------------------------------------------------
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
        # Remove the title only when it appears as a whole word.
        raw_name = re.sub(r'\b' + re.escape(title_clean) + r'\b', '', raw_name, flags=re.IGNORECASE)
    # Remove punctuation.
    raw_name = re.sub(r'[^\w\s]', '', raw_name)
    # Remove extra whitespace and convert to title case.
    cleaned_name = " ".join(raw_name.split())
    return cleaned_name.title()

# -------------------------------------------------------------------------------
# Process each year.
# -------------------------------------------------------------------------------
# List to accumulate interlock rows across years.
pairs_table_rows = []

for year in years:
    print(f"Processing year for interlock table: {year}")
    
    # Load the board data for the given year.
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"

    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    # We will need two things:
    # (1) A mapping from each institution to its set of cleaned board member names (for board sizes).
    # (2) A mapping from each board member (cleaned) to the set of institutions on which they serve.
    institution_to_members = defaultdict(set)
    board_member_dict = defaultdict(set)
    # To avoid duplicate contributions (if a board member appears twice, count only once per pair)
    created_interlocks = defaultdict(set)
    # For accumulating interlock contributions per institution pair.
    edge_accum = {}  # Key: tuple(sorted([inst1, inst2])); Value: count (sum of contributions)

    # Combine the two dataframes.
    combined_df = pd.concat([boards_df, double_boards_df], ignore_index=True)

    # Process each board row.
    for _, row in combined_df.iterrows():
        raw_name = row['Name']
        # Skip if the board member is marked as "vacant".
        if "vacant" in raw_name.lower():
            continue
        # Clean the board member's name.
        name = clean_name(raw_name)
        institution = row['Institution']

        # Add the cleaned name to the institution's membership set.
        institution_to_members[institution].add(name)

        # For each institution this board member has previously been on, count an interlock.
        for prev_institution in board_member_dict[name]:
            if prev_institution == institution:
                continue  # Skip if same institution.
            # Use sorted tuple so that order does not matter.
            pair = tuple(sorted([prev_institution, institution]))
            # Avoid double counting this board member’s contribution on this pair.
            if pair in created_interlocks[name]:
                continue
            created_interlocks[name].add(pair)
            # Add contribution (here, weight contribution is 1 per board member).
            if pair in edge_accum:
                edge_accum[pair] += 1
            else:
                edge_accum[pair] = 1

        # Record that this board member is now on the current institution.
        board_member_dict[name].add(institution)

    # Compute board sizes (unique cleaned board members per institution).
    board_sizes = {inst: len(members) for inst, members in institution_to_members.items()}

    # Now, for each interlock (institution pair), compute a normalized weight.
    # The normalization divides the raw contribution (i.e. the number of board members shared)
    # by the total unique number of board members across the two boards.
    for pair, raw_shared in edge_accum.items():
        source, target = pair
        size_source = board_sizes.get(source, 0)
        size_target = board_sizes.get(target, 0)
        total_unique = size_source + size_target - raw_shared
        normalized_weight = raw_shared / total_unique if total_unique > 0 else 0

        pairs_table_rows.append({
            'Source': source,
            'Target': target,
            'board_size_source': size_source,
            'board_size_target': size_target,
            'shared_members': normalized_weight,  # normalized weight per second code logic
            'Year': year
        })
    
    # Count nodes that had an interlock recorded.
    # These are institutions that appear in at least one interlock (edge).
    nodes_with_interlocks = set()
    for inst1, inst2 in edge_accum.keys():
        nodes_with_interlocks.add(inst1)
        nodes_with_interlocks.add(inst2)
    print(f"Year {year}: {len(nodes_with_interlocks)} nodes with interlocks recorded.")
    
    # (Optionally, you could still print the total number of institutions processed:)
    # print(f"Year {year}: {len(institution_to_members)} institutions processed.")

# Convert the accumulated rows into a DataFrame.
pairs_df = pd.DataFrame(pairs_table_rows)

# Write the interlock table to CSV.
output_path = f"{absolute_path}{yearly_interlocks}interlock_table.csv"
pairs_df.to_csv(output_path, index=False)
print(f"\nInterlock table saved to: {output_path}")
print(pairs_df.head())
