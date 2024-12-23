import os
import pandas as pd
import numpy as np

POSITION_BANK = [
    "President", "Chancellor", "Provost", "Director", "Dean", "Controller", "Trustee", 
    "Member", "Regent", "Chairman", "Overseer", "Assistant", "Librarian", "Secretary", 
    "Chaplain", "Minister", "Treasurer", "Senior Counsel", "General Counsel", "Legal Counsel", 
    "University Counsel", "College Counsel", "Special Counsel", "Corporation Counsel", 
    "Corporate Counsel", "Officer", "Chief", "Professor", "Commissioner", "Fellow", 
    "Chairperson", "Manager", "Clergy", "Coordinator", "Auditor", "Governor", 
    "Representative", "Stockbroker", "Advisor", "Commandant", "Rector", "Attorney", 
    "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy", 
    "Inspector General", "Instructor", "Registrar", "Ombuds", "Administrator", "Liaison", 
    "Administrative Associate", "Webmaster", "Specialist", "University Planner", "Architect"
]

ABSOLUTE_PATH = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
ALTERED_DATAFRAMES = "altered_dataframes\\"
GPT_DATAFRAMES = "gpt_dataframes\\"
GRAPHS = "graphs\\"
SCRIPTS = "scripts\\"
BOARD_DATAFRAMES = "board_dataframes\\"
SPLIT_DATAFRAMES = "split_dataframes\\"
TEMPORARY = "temporary_data\\"
YEARS = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]



def calculate_occurrences(string: str, key: str) -> int:
    """
    Count the number of occurrences of a given key within the string, 
    splitting by whitespace. Handles punctuation differently if key is a comma or semicolon.
    
    :param string: The input string to search within.
    :param key: The substring or character to count (e.g., 'and', ',', ';').
    :return: The integer count of occurrences.
    """
    words = string.split()
    occurrences_count = 0

    if key in {",", ";"}:
        # Count the times key appears as part of any word token
        for word in words:
            if key in word:
                occurrences_count += 1
    else:
        # Direct equality comparison if key is a normal word (like 'and')
        for word in words:
            if word == key:
                occurrences_count += 1
    return occurrences_count


def count_appearances(position: str) -> int:
    """
    Count how many position-related words from POSITION_BANK appear in the given string.

    :param position: A string containing potential positions.
    :return: The total count of matching position words found.
    """
    total = 0
    for word in POSITION_BANK:
        total += position.count(word)
    return total


def check_appears_twice(position: str) -> bool:
    """
    Check if the given string contains two or more position-related words 
    from POSITION_BANK.

    :param position: The string to check for multiple positions.
    :return: True if two or more positions appear, False otherwise.
    """
    return count_appearances(position) >= 2


def extract_position(position: str) -> str:
    """
    Extract the first occurrence of a known position word from POSITION_BANK
    that appears in the given position string.

    :param position: A string with potential position titles.
    :return: The first matched position word found, or None if none found.
    """
    words = position.split()
    for w in words:
        for key in POSITION_BANK:
            if key in w:
                return w
    return None


def split_multiple_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split rows that contain multiple positions joined by 'and'. 
    If both sides of the 'and' contain valid positions from POSITION_BANK, 
    split the row into two rows. Otherwise, leave the row as is.

    :param df: The input DataFrame with columns 'Name' and 'Position'.
    :return: A new DataFrame with split rows for multiple positions.
    """
    new_rows = []
    for index, row in df.iterrows():
        position = row["Position"].title() if not pd.isna(row["Position"]) else ""
        name = row["Name"]
        found_and = 'And' in position
        multiple_positions = check_appears_twice(position)
        num_of_and = calculate_occurrences(position, "And")

        if found_and and multiple_positions:
            last_appearing_and = position.rfind('And')
            first_position = position[:last_appearing_and].strip()
            second_position = position[last_appearing_and + 3:].strip()

            # Check if the substring after the last 'and' actually contains a valid position.
            positions_on_both_sides = sum(word in second_position for word in POSITION_BANK) >= 1
            if not positions_on_both_sides:
                # Single 'and' but the right side doesn't contain a known position.
                if num_of_and == 1:
                    new_rows.append(row)
                    continue
                # If there's more than one 'and', try the previous occurrence.
                elif num_of_and >= 2:
                    last_appearing_and = first_position.rfind('And')
                    second_position = position[last_appearing_and + 3:].strip()
                    first_position = position[:last_appearing_and].strip()

                positions_on_both_sides = sum(word in second_position for word in POSITION_BANK) >= 1
                if not positions_on_both_sides:
                    # If still no valid position found after adjusting, 
                    # just keep as a single row.
                    last_appearing_and = first_position.rfind('And')
                    second_position = position[last_appearing_and + 3:].strip()
                    first_position = position[:last_appearing_and].strip()

            # Final check and row-split
            positions_on_both_sides = sum(word in second_position for word in POSITION_BANK) >= 1
            if positions_on_both_sides:
                row_copy = row.copy()
                row["Position"] = first_position
                row_copy["Position"] = second_position.strip()
                new_rows.append(row)
                new_rows.append(row_copy)
            else:
                new_rows.append(row)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows)


def split_triple_positions(df: pd.DataFrame, delimiter: str) -> pd.DataFrame:
    """
    Split rows that contain 3 or more titles separated by the specified delimiter 
    (e.g., comma or semicolon). If both sides of the delimiter each contain 
    at least one valid position from POSITION_BANK, split the row. Otherwise, 
    keep the row as is.

    :param df: Input DataFrame with columns 'Name' and 'Position'.
    :param delimiter: Delimiter to split on (e.g. ',', ';').
    :return: A new DataFrame with split rows for triple or multiple positions.
    """
    new_rows = []
    for index, row in df.iterrows():
        position = row["Position"].title() if not pd.isna(row["Position"]) else ""
        appears_twice = check_appears_twice(position)
        num_of_delimiter = calculate_occurrences(position, delimiter)

        if num_of_delimiter > 0 and appears_twice:
            last_appearing_delimiter = position.rfind(delimiter)
            first_position = position[:last_appearing_delimiter].strip()
            second_position = position[last_appearing_delimiter + 1:].strip()

            positions_on_both_sides = sum(word in second_position for word in POSITION_BANK) >= 1
            if not positions_on_both_sides:
                # If second part doesn't contain a known position, attempt fallback logic
                if num_of_delimiter == 1:
                    new_rows.append(row)
                    continue

                last_appearing_delimiter = first_position.rfind(delimiter)
                second_position = position[last_appearing_delimiter + 1:].strip()
                first_position = position[:last_appearing_delimiter].strip()
                positions_on_both_sides = sum(word in second_position for word in POSITION_BANK) >= 1

                # If still no match and exactly 2 delimiters, treat as single or dual-split
                if not positions_on_both_sides and num_of_delimiter == 2:
                    new_rows.append(row)
                    continue
                elif positions_on_both_sides and num_of_delimiter == 2:
                    new_row_copy = row.copy()
                    row["Position"] = first_position
                    new_row_copy["Position"] = second_position.strip()
                    new_rows.append(row)
                    new_rows.append(new_row_copy)
                    continue

            # If positions were found on both sides or too many delimiters, finalize the split
            new_row_copy = row.copy()
            row["Position"] = first_position
            new_row_copy["Position"] = second_position.strip()
            new_rows.append(row)
            new_rows.append(new_row_copy)
        else:
            new_rows.append(row)

    return pd.DataFrame(new_rows)


def merge_incorrectly_split(df: pd.DataFrame, delimiter: str) -> pd.DataFrame:
    """
    Merge rows that were incorrectly split. For example, if the next row 
    belongs to the same person and we detect a phrase like "Office Of" 
    or something indicating it should be combined, we re-merge them.

    :param df: Input DataFrame with columns 'Name' and 'Position'.
    :param delimiter: Delimiter used (e.g., ',', ';').
    :return: A new DataFrame with corrected (merged) rows.
    """
    df = df.reset_index(drop=True)
    idx = 1
    while idx < len(df):
        current_position = str(df.at[idx, "Position"]) if not pd.isna(df.at[idx, "Position"]) else ""
        current_position = current_position.lower().title()
        current_name = str(df.at[idx, "Name"]).lower().title()

        num_delimiters = calculate_occurrences(current_position, delimiter)
        # Merge if "Office Of" found without delimiter
        if num_delimiters == 0 and "Office Of" in current_position:
            if idx - 1 < len(df):
                prev_name = str(df.at[idx - 1, "Name"]).lower().title()
                if current_name == prev_name:
                    df.at[idx, "Position"] = df.at[idx + 1, "Position"] + ", " + df.at[idx, "Position"]
                    df = df.drop(idx - 1).reset_index(drop=True)

        # Merge if there's a single recognized position that ends with "'s" 
        # (like "President's Office"), presumably belonging with the previous row
        num_positions = count_appearances(current_position)
        if num_positions == 1:
            spec_position = extract_position(current_position)
            if spec_position and "'s" in spec_position.lower() and "office" in current_position.lower():
                if idx - 1 < len(df):
                    prev_name = str(df.at[idx - 1, "Name"]).lower().title()
                    if current_name == prev_name:
                        df.at[idx, "Position"] = df.at[idx - 1, "Position"] + ", " + df.at[idx, "Position"]
                        df = df.drop(idx - 1).reset_index(drop=True)

        idx += 1
    return pd.DataFrame(df)


def all_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with rows whose 'Position' contains at least 
    two recognized positions from POSITION_BANK.

    :param df: Input DataFrame with columns 'Position'.
    :return: A subset DataFrame with rows that have multiple positions.
    """
    rows_with_multiples = []
    for index, row in df.iterrows():
        if check_appears_twice(row["Position"]):
            rows_with_multiples.append(row)
    return pd.DataFrame(rows_with_multiples)


def apply_splitting_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the repeated splitting and merging functions to handle 
    complex multiple-position rows. The order of operations is significant.

    :param df: Original DataFrame to process.
    :return: Processed DataFrame after multiple splits and merges.
    """
    # Split triple positions by semicolons repeatedly
    df = split_triple_positions(df, ";")
    df = split_triple_positions(df, ";")
    # Split multiple positions with 'and' repeatedly
    df = split_multiple_positions(df)
    df = split_multiple_positions(df)
    # Split triple positions by commas repeatedly
    df = split_triple_positions(df, ",")
    df = split_triple_positions(df, ",")
    # Merge incorrectly split rows
    df = merge_incorrectly_split(df, ",")
    # If you want, you can also run merge_incorrectly_split(df, ";") as in the commented code
    # df = merge_incorrectly_split(df, ";")
    return df

def main():
    """
    Main execution function to read the original CSV, process the DataFrame, 
    fix NaNs, and save the cleaned result.
    """
    for year in YEARS:
        print(f"Processing: {year}")
        path_read = f"{ABSOLUTE_PATH}{GPT_DATAFRAMES}{year}_gptDataframe.csv"
        split_df_path = f"{ABSOLUTE_PATH}{SPLIT_DATAFRAMES}{year}_split_positions.csv"
        full_dataframe = pd.read_csv(path_read)

        # Apply the splitting pipeline
        split_dataframe_all = apply_splitting_pipeline(full_dataframe)

        # Replace NaN in 'Position' column with string 'nan'
        for index, row in split_dataframe_all.iterrows():
            if pd.isna(row["Position"]):
                split_dataframe_all.at[index, "Position"] = "nan"

        # Save final DataFrame to CSV
        split_dataframe_all.to_csv(split_df_path, index=False)


if __name__ == "__main__":
    main()
