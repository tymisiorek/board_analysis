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

def remove_non_samples(df):
    return df[df['PrimarySample'] == True]


# List to accumulate rows for the interlock table across all years.
pairs_table_rows = []

for year in years:
    print(f"Processing year for pair table: {year}")

    # Load the board data for the given year.
    boards_path = f"{absolute_path}{board_dataframes}{year}_boards.csv"
    double_boards_path = f"{absolute_path}{board_dataframes}{year}_double_board.csv"

    boards_df = pd.read_csv(boards_path)
    double_boards_df = pd.read_csv(double_boards_path)
    boards_df = remove_non_samples(boards_df)
    double_boards_df = remove_non_samples(double_boards_df)
    
    # Build the institution -> set(members) mapping.
    institution_to_members = defaultdict(set)
    for _, row in boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    for _, row in double_boards_df.iterrows():
        institution_to_members[row['Institution']].add(row['Name'])
    
    # Compute board sizes for each institution.
    board_sizes = {inst: len(members) for inst, members in institution_to_members.items()}
    
    # Create pairs of institutions and calculate the number of shared members.
    institutions = list(institution_to_members.keys())
    n = len(institutions)
    for i in range(n):
        for j in range(i + 1, n):
            source = institutions[i]
            target = institutions[j]
            shared_members = len(institution_to_members[source].intersection(institution_to_members[target]))
            if shared_members > 0:
                pairs_table_rows.append({
                    'Source': source,
                    'Target': target,
                    'board_size_source': board_sizes[source],
                    'board_size_target': board_sizes[target],
                    'shared_members': shared_members,
                    'Year': year
                })

# Convert the accumulated rows into a DataFrame.
pairs_df = pd.DataFrame(pairs_table_rows)
output_path = f"{absolute_path}{yearly_interlocks}interlock_table.csv"
pairs_df.to_csv(output_path, index=False)
print(f"Pair table saved to: {output_path}")
print(pairs_df.head())
