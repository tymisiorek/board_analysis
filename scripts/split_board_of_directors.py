import pandas as pd
import os
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import Counter
import copy
from nameparser import HumanName

#global vars
POSITION_BANK = ["President", "Chancellor", "Provost", "Director", "Dean", "Controller", "Trustee", "Member", "Regent", "Chairman", "Overseer", "Assistant", "Librarian", "Secretary", "Chaplain", "Minister", "Treasurer", "Senior Counsel", "General Counsel", "Legal Counsel", "University Counsel", "College Counsel", "Special Counsel", "Corporation Counsel", "Officer", "Chief", "Professor", "Commissioner", "Fellow", "Chairperson", "Manager", "Clergy", "Coordinator", "Auditor", "Governor", "Representative", "Stockbroker", "Advisor", "Commandant", "Rector", "Attorney", "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy", "Inspector General"]
NON_BOARD_WORDS =["President", "Chancellor", "Provost", "Dean", "Controller", "Overseer", "Assistant", "Librarian", "Secretary", "Chaplain", "Minister", "Treasurer", "Senior Counsel", "General Counsel", "Legal Counsel", "University Counsel", "College Counsel", "Special Counsel", "Corporation Counsel", "Officer", "Chief", "Professor", "Commissioner", "Manager", "Clergy", "Coordinator", "Auditor", "Representative", "Stockbroker", "Advisor", "Commandant", "Rector", "Attorney", "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy", "Inspector General", "Vice", "Chancellor,", "President,", "Executive", "Affairs", "Senior", "Associate", "Administration", "University", "College"]
BOARD_WORDS = ["Trustee", "Regent", "Member", "Fellow", "Overseer", "Governor", "Curator", "Visitor", "Manager"]
OTHER_BOARD_WORD = ["President", "Chairman", "Chairperson" ,"Treasurer", "Rector", "Member", "Secretary", "Ex Officio"]

absolute_path = "C:\\Users\\tykun\\\OneDrive\\Documents\\SchoolDocs\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts =  "scripts\\"
board_dataframes = "board_dataframes\\"
split_dataframes = "split_dataframes\\"
temporary = "temporary_data\\"

altered_dataframe_path = f"{absolute_path}{altered_dataframes}"
gpt_dataframe_path = f"{absolute_path}{gpt_dataframes}" 
graph_path = f"{absolute_path}{graphs}"
script_path = f"{absolute_path}{scripts}"
boards_path = f"{absolute_path}{board_dataframes}"
split_path = f"{absolute_path}{split_dataframes}"
temporary_path = f"{absolute_path}{temporary}"

#Valid Years
years = ["1999", "2000", "2005", "2008", "2009", "2013"]


#helper functions
def higher_ascii(char1, char2):    
    if ord(char1.upper()) >= ord(char2.upper()):
        return True
    else:
        return False

def parse_name(raw_name):
    raw = raw_name.replace("Rev", "")
    raw = raw.replace("Very ", "")
    parsed_name = HumanName(raw)
    parsed_name.suffix = ""
    pre_parse = str(parsed_name)
    split_name = pre_parse.split(" ")
    last_name = parsed_name.last
    if last_name == "" or " " in last_name or last_name == '.':
        return split_name[-1]
    else:
        return str(last_name)
    

def extract_institutions(df):
    """
    Extracts the names of all unique institutions from the given DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame containing institution data.
    Returns:
        list: A list of unique institution names.
    """
    institution_list = []  # Initialize an empty list to store unique institution names
    for index, row in df.iterrows():  # Iterate over each row in the DataFrame
        if row["Institution"] not in institution_list:  # Check if the institution is not already in the list
            institution_list.append(row["Institution"])  # Add the institution to the list
    print(institution_list)  # Print the list of unique institutions
    return institution_list  # Return the list of unique institutions


def determine_board_position(df):
    """
    Determines the most common board position names for each institution and identifies if there are multiple boards.
    Args:
        df (pandas.DataFrame): The DataFrame containing institution and position data.
    Returns:
        tuple: A tuple containing two dictionaries:
            - most_frequent (dict): A dictionary where keys are institution names and values are the most common board position names.
            - two_boards (dict): A dictionary where keys are institution names and values indicate the second most common board position name if there are multiple boards, otherwise None.
    """
    most_frequent = {}  # Initialize an empty dictionary to store the most common board position for each institution
    two_boards = {}  # Initialize an empty dictionary to store the second most common board position for each institution
    grouped_df = df.groupby("Institution")  # Group the DataFrame by institution
    for key, value in grouped_df:  # Iterate over each group
        word_count = Counter()  # Initialize a Counter to count the occurrences of board position words
        for position in value["Position"]:  # Iterate over each position in the group
            if not pd.isna(position):  # Check if the position is not NaN
                individual_words = position.split()  # Split the position into individual words
            else:
                individual_words = ""  # If the position is NaN, set individual_words to an empty string
            filtered_words = [word for word in individual_words if word in BOARD_WORDS]  # Filter the words to include only those in BOARD_WORDS
            word_count.update(filtered_words)  # Update the word count with the filtered words
        if word_count:  # Check if there are any counted words
            common_words = word_count.most_common(2)  # Get the two most common words
            most_frequent[key] = common_words[0][0]  # Set the most common word as the most frequent board position for the institution
            # If there are two board words that come up, and the second board word comes up frequently, it is likely a school with two boards
            if len(common_words) >= 2:
                large_enough_board = common_words[1][1] >= (common_words[0][1] / 4) or (common_words[0][1] >= 8)  # Check if the second most common word is frequent enough
            else:
                large_enough_board = False  # If there is only one common word, set large_enough_board to False
            if len(common_words) >= 2 and large_enough_board and common_words[1][0] != "Director":  # Check if there are two common words, the second word is frequent enough, and it is not "Director"
                if common_words[1][0].strip().lower() != common_words[0][0].strip().lower():  # Check if the second common word is not the same as the first common word
                    two_boards[key] = common_words[1][0]  # Set the second common word as the second board position for the institution
            else:
                two_boards[key] = None  # If the conditions are not met, set the second board position to None
        else:
            most_frequent[key] = None
            two_boards[key] = None
    return most_frequent, two_boards  # Return the dictionaries of most frequent and second board positions


def determine_director_schools(df):
    """
    Determines the institutions where 'Director' is a common board position and identifies if there are multiple boards.
    Args:
        df (pandas.DataFrame): The DataFrame containing institution and position data.
    Returns:
        tuple: A tuple containing two dictionaries:
            - most_frequent (dict): A dictionary where keys are institution names and values are the most common 'Director' position names.
            - two_boards (dict): A dictionary where keys are institution names and values indicate the second most common 'Director' position name if there are multiple boards, otherwise None.
    """
    most_frequent = {}
    two_boards = {}
    grouped_df = df.groupby("Institution")
    for key, value in grouped_df:
        word_count = Counter()
        for position in value["Position"]:
            individual_words = position.split()
            board_words_dir = ['Director']
            filtered_words = [word for word in individual_words if word in board_words_dir]
            word_count.update(filtered_words)
        if word_count:
            common_words = word_count.most_common(2)
            most_frequent[key] = common_words[0][0]
            #if there are two board words that come up, and the second board word comes up frequently, likely is a school with two boards
            if len(common_words) >= 2:
                large_enough_board = common_words[1][1] >= (common_words[0][1] / 4) or (common_words[0][1] >= 8)
            else:
                large_enough_board = False
            if len(common_words) >= 2 and large_enough_board:
                if common_words[1][0].strip().lower() != common_words[0][0].strip().lower():
                    two_boards[key] = common_words[1][0]
            else:
                two_boards[key] = None
        else:
            most_frequent[key] = None
            two_boards[key] = None
    return most_frequent, two_boards

def find_word_grouping(df, board_name):
    """
    Finds the start and end indices of each board in the given dataframe, based on the provided board names.
    
    Args:
        df (pd.DataFrame): The dataframe containing board information, indexed by "Institution" and "Position".
        board_name (dict): A dictionary mapping institution names to specific board positions.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A concatenated dataframe of the relevant board sections (or an empty dataframe if none found).
            - dict: A dictionary of the first board occurrence indices per institution.
            - dict: A dictionary of the last board occurrence indices per institution.
            - dict: A dictionary of the first occurrence of each institution in the dataframe.
    """
    grouped_df = df.groupby("Institution")
    first_board_occurrence, last_board_occurrence, first_institution_occurrence = {}, {}, {}
    result_dataframe = []

    for key, value in grouped_df:
        board_position = board_name.get(key, None)
        first_institution_occurrence[key] = value.index[0]

        if board_position is not None:
            all_members = value["Position"].tolist()
            try:
                # Find the first and last index of the board position in the group
                first_index = next(i for i, pos in enumerate(all_members) if board_position == pos.title())
                last_index = len(all_members) - next(i for i, pos in enumerate(reversed(all_members)) if board_position in pos.title()) - 1
                
                first_board_occurrence[key] = value.index[first_index]
                last_board_occurrence[key] = value.index[last_index]
                result_dataframe.append(value.iloc[first_index:last_index + 1])
            except StopIteration:
                # If position is not found, default to the full range of the group
                first_board_occurrence[key] = value.index[0]
                last_board_occurrence[key] = value.index[-1]
        else:
            first_board_occurrence[key] = value.index[0]
            last_board_occurrence[key] = value.index[-1]

    # Check if result_dataframe is empty before concatenating
    if result_dataframe:
        return pd.concat(result_dataframe), first_board_occurrence, last_board_occurrence, first_institution_occurrence
    else:
        print("Warning: No relevant board sections found in the dataframe.")
        return pd.DataFrame(), first_board_occurrence, last_board_occurrence, first_institution_occurrence


# Method to find board positions within the dataframe using substring matching
def find_word_grouping_substring(df, board_name):
    """
    Finds the start and end indices of each board in the given dataframe using substring matching for board names.
    
    Args:
        df (pd.DataFrame): The dataframe containing board information, indexed by "Institution" and "Position".
        board_name (dict): A dictionary mapping institution names to specific board positions.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A concatenated dataframe of the relevant board sections.
            - dict: A dictionary of the first board occurrence indices per institution.
            - dict: A dictionary of the last board occurrence indices per institution.
            - dict: A dictionary of the first occurrence of each institution in the dataframe.
    """
    grouped_df = df.groupby("Institution")
    first_board_occurrence, last_board_occurrence, first_institution_occurrence = {}, {}, {}
    result_dataframe = []

    for key, value in grouped_df:
        board_position = board_name.get(key, None)
        first_institution_occurrence[key] = value.index[0]

        if board_position is not None:
            all_members = value["Position"].tolist()
            try:
                # Find the first and last index of the board position using substring matching
                first_index = next(i for i, pos in enumerate(all_members) if board_position in pos.title())
                last_index = len(all_members) - next(i for i, pos in enumerate(reversed(all_members)) if board_position in pos.title()) - 1
                
                first_board_occurrence[key] = value.index[first_index]
                last_board_occurrence[key] = value.index[last_index]
                result_dataframe.append(value.iloc[first_index:last_index + 1])
            except StopIteration:
                # If position is not found, default to the full range of the group
                first_board_occurrence[key] = value.index[0]
                last_board_occurrence[key] = value.index[-1]
        else:
            first_board_occurrence[key] = value.index[0]
            last_board_occurrence[key] = value.index[-1]

    return pd.concat(result_dataframe), first_board_occurrence, last_board_occurrence, first_institution_occurrence


# Method to verify alphabetical ordering of names when expanding boards upwards
def verify_ordering(full_df, current_board_start, count):
    """
    Verifies that names are ordered alphabetically in ascending order when expanding a board upwards.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        current_board_start (int): The starting index of the current board in the dataframe.
        count (int): Number of rows to check upwards.

    Returns:
        bool: True if the names are in correct alphabetical order, False otherwise.
    """
    original_board_position = full_df.iloc[current_board_start]
    original_last_name = parse_name(original_board_position["Name"])

    while count > 0:
        current_position = full_df.iloc[current_board_start - count]
        current_last_name = parse_name(current_position["Name"])
        correct_ordering = higher_ascii(original_last_name[0], current_last_name[0])
        if not correct_ordering:
            return False
        count -= 1

    return True


# Method to expand a single board upwards within the dataframe
def expand_single_board_upward(full_df, grouped_boards, board_indices_start):
    """
    Expands a single board section upward by checking rows above the current board start index.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        grouped_boards (dict): Dictionary of grouped board dataframes by institution.
        board_indices_start (dict): Dictionary of starting indices for each board.

    Returns:
        dict: Updated dictionary of grouped boards with expanded data.
    """
    sorted_keys = sorted(board_indices_start.keys(), key=lambda k: board_indices_start[k])

    for key in sorted_keys:
        expanded_flag = False
        count = 4  # Number of rows to expand upwards
        current_board_start = board_indices_start[key]

        while count > 0:
            previous_df_row = full_df.iloc[current_board_start - count]
            previous_row_position = previous_df_row["Position"].title()
            original_df_row = full_df.iloc[current_board_start]
            original_row_position = original_df_row["Position"].title()
            same_institution = original_df_row["Institution"] == key
            board_position = any(p in previous_row_position for p in OTHER_BOARD_WORD)

            if same_institution and (board_position or expanded_flag):
                if not expanded_flag:
                    alphabetical_order = verify_ordering(full_df, current_board_start, count)

                if (alphabetical_order and "Dean" not in previous_row_position) or expanded_flag:
                    grouped_boards[key] = pd.concat([pd.DataFrame([previous_df_row]), grouped_boards.get(key, pd.DataFrame())])
                    expanded_flag = True

            count -= 1

    return grouped_boards


def expand_double_board_upward(full_df, grouped_boards, board_indices_start, double_boards):
    """
    Expands double board sections upwards by checking rows above the current board start index.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        grouped_boards (dict): Dictionary of grouped board dataframes by institution.
        board_indices_start (dict): Dictionary of starting indices for each board.
        double_boards (dict): Dictionary indicating which institutions have double boards.

    Returns:
        dict: Updated dictionary of grouped boards with expanded data.
    """
    sorted_keys = sorted(board_indices_start.keys(), key=lambda k: board_indices_start[k])

    for key in sorted_keys:
        expanded_flag = False
        count = 4  # Arbitrary value for the number of rows to expand upwards
        current_board_start = board_indices_start[key]

        while count > 0:
            previous_df_row = full_df.iloc[current_board_start - count]
            previous_row_position = previous_df_row["Position"].title()
            original_df_row = full_df.iloc[current_board_start]
            same_institution = original_df_row["Institution"] == key
            board_position = any(p in previous_row_position for p in OTHER_BOARD_WORD)

            # Check if the key exists in double_boards before accessing it
            if key in double_boards and same_institution and (board_position or expanded_flag) and double_boards[key] is not None:
                if not expanded_flag:
                    # Verify alphabetical ordering of last names before expansion
                    alphabetical_order = verify_ordering(full_df, current_board_start, count)

                if (alphabetical_order and "Dean" not in previous_row_position) or expanded_flag:
                    grouped_boards[key] = pd.concat([pd.DataFrame([previous_df_row]), grouped_boards.get(key, pd.DataFrame())])
                    expanded_flag = True

            count -= 1

    return grouped_boards


# Method to expand a single board section downwards within the dataframe
def expand_single_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index):
    """
    Expands a single board section downward by checking rows below the current board end index.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        grouped_boards (dict): Dictionary of grouped board dataframes by institution.
        board_indices_end (dict): Dictionary of ending indices for each board.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.

    Returns:
        dict: Updated dictionary of grouped boards with expanded data.
    """
    sorted_keys = sorted(board_indices_end.keys(), key=lambda k: board_indices_end[k])

    for key in sorted_keys:
        if sorted_keys.index(key) < len(sorted_keys) - 1:
            current_board_end = board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index[next_key]

            # Expand downwards only if within range
            if (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, next_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])

    return grouped_boards


def expand_double_board_downward(full_df, grouped_boards, double_board_indices_start, double_board_indices_end, 
                                 first_institution_index, first_board_indices_start, first_board_indices_end, double_boards):
    """
    Expands double board sections downward by checking rows below the current board end index.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        grouped_boards (dict): Dictionary of grouped board dataframes by institution.
        double_board_indices_start (dict): Dictionary of starting indices for double boards.
        double_board_indices_end (dict): Dictionary of ending indices for double boards.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.
        first_board_indices_start (dict): Dictionary of starting indices for single boards.
        first_board_indices_end (dict): Dictionary of ending indices for single boards.
        double_boards (dict): Dictionary indicating which institutions have double boards.

    Returns:
        dict: Updated dictionary of grouped boards with expanded data.
    """
    sorted_keys = sorted(double_board_indices_end.keys(), key=lambda k: double_board_indices_end[k])

    for key in sorted_keys:
        # Check if the key exists in double_boards before accessing it
        if key in double_boards and sorted_keys.index(key) < len(sorted_keys) - 1 and double_boards[key] is not None:
            current_board_end = double_board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index.get(next_key, float('inf'))
            current_board_start = double_board_indices_start[key]
            first_board_end = first_board_indices_end.get(key, -1)
            first_board_start = first_board_indices_start.get(key, -1)

            # Check conditions to expand downward
            if (current_board_start > first_board_end) and (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, next_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])
            elif current_board_start < first_board_end and (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, first_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])

    return grouped_boards



# Method to expand director board sections downwards within the dataframe
def expand_director_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index):
    """
    Expands director board sections downward by checking rows below the current board end index.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        grouped_boards (dict): Dictionary of grouped board dataframes by institution.
        board_indices_end (dict): Dictionary of ending indices for director boards.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.

    Returns:
        dict: Updated dictionary of grouped boards with expanded data.
    """
    sorted_keys = sorted(board_indices_end.keys(), key=lambda k: board_indices_end[k])

    for key in sorted_keys:
        if sorted_keys.index(key) < len(sorted_keys) - 1:
            current_board_end = board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index[next_key]

            # Limit the expansion range
            if current_board_end + 3 < next_board_start:
                next_board_start = current_board_end + 3

            for index in range(current_board_end + 1, next_board_start):
                grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])
                index += 1

    return grouped_boards

def expand_board(full_df, board_df, board_indices_start, board_indices_end, first_institution_index):
    """
    Expands a single board section both upwards and downwards within the dataframe and removes duplicate rows.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        board_df (pd.DataFrame): The dataframe containing specific board sections to be expanded.
        board_indices_start (dict): Dictionary of starting indices for each board.
        board_indices_end (dict): Dictionary of ending indices for each board.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.

    Returns:
        pd.DataFrame: A cleaned and expanded dataframe containing board data, with duplicates removed.
    """
    initial_boards = list(set(board_df["Institution"].values))
    grouped_boards = {name: group for name, group in board_df.groupby("Institution")}
    grouped_boards = expand_single_board_upward(full_df, grouped_boards, board_indices_start)
    grouped_boards = expand_single_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index)
    combined_boards = pd.concat(grouped_boards.values())
    final_boards = list(set(combined_boards["Institution"].values))
    indices_to_drop = []

    for index, row in combined_boards.iterrows():
        inst = row["Institution"]
        if inst in final_boards and inst not in initial_boards:
            indices_to_drop.append(index)

    cleaned__df = combined_boards.drop(index=indices_to_drop).reset_index(drop=True)
    return cleaned__df


def expand_double_board(full_df, board_df, board_indices_start, board_indices_end, first_institution_index, double_boards, first_board_indices_start, first_board_indices_end):
    """
    Expands double board sections both upwards and downwards within the dataframe.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        board_df (pd.DataFrame): The dataframe containing specific double board sections to be expanded.
        board_indices_start (dict): Dictionary of starting indices for double boards.
        board_indices_end (dict): Dictionary of ending indices for double boards.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.
        double_boards (dict): Dictionary indicating which institutions have double boards.
        first_board_indices_start (dict): Dictionary of starting indices for single boards.
        first_board_indices_end (dict): Dictionary of ending indices for single boards.

    Returns:
        pd.DataFrame: An expanded dataframe containing double board data.
    """
    grouped_boards = {name: group for name, group in board_df.groupby("Institution")}
    grouped_boards = expand_double_board_upward(full_df, grouped_boards, board_indices_start, double_boards)
    grouped_boards = expand_double_board_downward(full_df, grouped_boards, board_indices_start, board_indices_end, first_institution_index, first_board_indices_start, first_board_indices_end, double_boards)
    combined_boards = pd.concat(grouped_boards.values())
    return combined_boards


def expand_directors(full_df, board_df, board_indices_start, board_indices_end, first_institution_index):
    """
    Expands director board sections both upwards and downwards within the dataframe.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.
        board_df (pd.DataFrame): The dataframe containing specific director board sections to be expanded.
        board_indices_start (dict): Dictionary of starting indices for director boards.
        board_indices_end (dict): Dictionary of ending indices for director boards.
        first_institution_index (dict): Dictionary indicating the first occurrence index of each institution.

    Returns:
        pd.DataFrame: An expanded dataframe containing director board data.
    """
    grouped_boards = {name: group for name, group in board_df.groupby("Institution")}
    grouped_boards = expand_single_board_upward(full_df, grouped_boards, board_indices_start)
    grouped_boards = expand_director_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index)
    combined_boards = pd.concat(grouped_boards.values())
    return combined_boards


def assemble_board_dict(board_df):
    """
    Assembles a dictionary where each key is an institution and the value is the corresponding dataframe subset.

    Args:
        board_df (pd.DataFrame): The dataframe containing board data, with "Institution" as a column.

    Returns:
        dict: A dictionary mapping each institution to its subset of the board dataframe.
    """
    board_dict = {}
    for institution in board_df['Institution'].unique():
        rows = board_df[board_df["Institution"] == institution]
        board_dict[institution] = rows
    return board_dict


# Cleaning + Deletion

def clean_false_members(expanded_boards, university_boards, original_boards):
    """
    Removes false members from the expanded board dataframe based on specific position criteria.

    Args:
        expanded_boards (pd.DataFrame): The dataframe containing expanded board data.
        university_boards (pd.DataFrame): The original university boards dataframe.
        original_boards (pd.DataFrame): The original board data before expansion.

    Returns:
        pd.DataFrame: A cleaned dataframe with rows containing certain positions (e.g., "Dean" or "Director") removed.
    """
    indices_to_drop = []
    for index, row in expanded_boards.iterrows():
        pos = row["Position"]
        if "Dean" in pos or "Director" in pos:
            indices_to_drop.append(index)
    cleaned__df = expanded_boards.drop(index=indices_to_drop).reset_index(drop=True)
    return cleaned__df


def delete_overlap(primary_boards, secondary_boards):
    """
    Deletes rows in the secondary board dataframe that overlap with the primary board dataframe.

    Args:
        primary_boards (pd.DataFrame): The primary board dataframe to check for overlaps.
        secondary_boards (pd.DataFrame): The secondary board dataframe to remove overlapping rows from.

    Returns:
        pd.DataFrame: The modified secondary board dataframe with overlapping rows removed.
    """
    for index, row in secondary_boards.iterrows():
        if any(row.equals(primary_row) for _, primary_row in primary_boards.iterrows()):
            secondary_boards.drop(index, inplace=True)
    return secondary_boards


def validate_double_boards(board_dict, double_boards):
    """
    Validates if rows in the double boards dataframe exist within the original board dictionary.

    Args:
        board_dict (dict): A dictionary where keys are institutions and values are their respective board dataframes.
        double_boards (pd.DataFrame): The dataframe containing double board data to validate.

    Returns:
        list: A list of invalid institutions found in the double boards dataframe.
    """
    invalid_list = []
    for index, row in double_boards.iterrows():
        institution = row["Institution"]
        if institution in board_dict:
            original_board = board_dict[institution]
            is_row_in_df = original_board.apply(lambda x: x.equals(row), axis=1).any()
            if is_row_in_df and institution not in invalid_list:
                print(f"Invalid board: {institution}")
                invalid_list.append(institution)
    return invalid_list


def verify_ordering_entire_board(full_df):
    """
    Verifies the alphabetical ordering of the last names in the entire board dataframe and marks rows for removal if ordering issues are found.

    Args:
        full_df (pd.DataFrame): The complete dataframe containing all board data.

    Returns:
        pd.DataFrame: A modified dataframe with rows that do not follow alphabetical ordering removed.
    """
    indices_to_remove = set()
    for key, group in full_df.groupby('Institution'):
        if len(group) <= 4:
            indices_to_remove.update(group.index)
            continue
        count = 0
        previous_last_name = "000"
        for i, row in group.iterrows():
            name = row["Name"]
            current_last_name = parse_name(name)
            correct_ordering = higher_ascii(current_last_name[0], previous_last_name[0])
            if not correct_ordering and count <= 2:
                print(key, name, "  ", previous_last_name)
                indices_to_remove.add(i - 1)
            count += 1
            previous_last_name = current_last_name
    print("here ", indices_to_remove)
    modified_df = full_df.drop(index=list(indices_to_remove))
    modified_df.reset_index(drop=True, inplace=True)
    return modified_df


def clean_false_members_directors(df):
    """
    Removes false members from the dataframe based on specific position criteria related to directors.

    Args:
        df (pd.DataFrame): The dataframe containing board data.

    Returns:
        pd.DataFrame: A cleaned dataframe with rows containing positions like "Dean" or "Director" removed.
    """
    indices_to_drop = []
    for index, row in df.iterrows():
        pos = row["Position"]
        if "Dean" in pos or "Director," in pos:
            indices_to_drop.append(index)
    cleaned__df = df.drop(index=indices_to_drop).reset_index(drop=False)
    return cleaned__df

#redefine some global
CHAIRPERSONS = ["Chairman", "Chairperson", "President", "Chair", "Chancellor"]
MEMBERS = ["Trustee", "Regent", "Member", "Fellow", "Overseer", "Governor", "Curator", "Visitor", "Manager", "Director"]
OTHER_BOARD_WORD = ["Treasurer", "Faculty Representative", "Rector", "Secretary", "Counsel", "Clerk", "Vacant", "Executive Committee Member", "Special", "Student", "Chief Executive Officer", "Affiliation", "Justice", "Registrar", "Staff Representative", "Librarian",
                    "Alumni Representative", "Faculty Visitor", "Chief Investment Officer"]

def mark_members(board_df, university_boards):
    board_df["FixedPosition"] = ""
    grouped_boards = board_df.groupby("Institution")
    for key, value in grouped_boards:
        for index, row in value.iterrows():
            position = row["Position"].title()
            board_name = university_boards[key]
            pres_appears = any(pos in position for pos in CHAIRPERSONS)
            if board_name is None:
                board_name = "zZbkjlhz01" 

            if board_name in row["Position"]:
                board_df.at[index, "FixedPosition"] = board_name
            elif  pres_appears and "Vice" not in position:
                board_df.at[index, "FixedPosition"] = "Board President"
            elif pres_appears and "Vice" in position:
                board_df.at[index, "FixedPosition"] = "Board Vice President"
            elif any(pos in position for pos in OTHER_BOARD_WORD):
                board_df.at[index, "FixedPosition"] = "Other Board Member"
            else:
                board_df.at[index, "FixedPosition"] = board_name

            if "Ex Officio" in row["Position"]:
                board_df.at[index, "FixedPosition"] += ", Ex Officio"
    return board_df


#remove (dash, comma, period from system inst dict for better matching)
def clean_system_inst_dict(full_df, state_systems):
    """
    Cleans the system institution dictionary by removing dashes, commas, and periods from the keys.

    Args:
        system_inst_dict (dict): The original system institution dictionary.

    Returns:
        dict: The cleaned system institution dictionary.
    """
    id_dict = {}
    for index, row in full_df.iterrows():
        id_dict[row["Institution"]] = row["AffiliationId"]

    #mark state system boards 
    system_id_dict = {}
    system_inst_dict = {}
    for index, row in state_systems.iterrows():
        if not pd.isna(row["StateSystem"]):
            system_id_dict[row["AffiliationId"]] = row["StateSystem"] 
            system_inst_dict[row["Institution"]] = row["StateSystem"]

    #remove (dash, comma, period from system inst dict for better matching)
    system_inst_dict_cleaned = {
        key.replace("-", " ").replace(",", "").replace(".", ""): value
        for key, value in system_inst_dict.items()
    }
    return system_id_dict, system_inst_dict, system_inst_dict_cleaned


def find_unmarked_boards(university_boards, full_first_board):
    """
    Identifies schools with a board where the board position does not appear as an exact string.

    Args:
        university_boards (dict): Dictionary of university boards.
        full_first_board (pd.DataFrame): DataFrame containing the first board information.

    Returns:
        dict: Dictionary of unmarked boards.
    """
    unmarked_boards = {}
    for key, value in university_boards.items():
        if value is not None and key not in full_first_board["Institution"].values:
            unmarked_boards[key] = value
            print(key + ": ", value)
    return unmarked_boards

def create_system_dicts(state_systems):
    """
    Creates dictionaries for mapping state system information.
    
    Args:
        state_systems (pd.DataFrame): The dataframe containing state system data.
    
    Returns:
        tuple: Contains cleaned system ID dictionary and system institution dictionary.
    """
    system_id_dict = {}
    system_inst_dict = {}

    for index, row in state_systems.iterrows():
        if not pd.isna(row["StateSystem"]):
            system_id_dict[row["AffiliationId"]] = row["StateSystem"]
            system_inst_dict[row["Institution"]] = row["StateSystem"]

    return system_id_dict, system_inst_dict



def process_year_data(year, split_path, state_systems):
    """
    Processes the board data for a given year, including cleaning and marking.
    
    Args:
        year (str): The year to process.
        split_path (str): Path to the split CSV files.
        state_systems (pd.DataFrame): The dataframe with state system data.
    
    Returns:
        tuple: Processed dataframes for single and double boards.
    """
    # Load the dataframe for the current year
    df_path = f"{split_path}{year}_split_positions.csv"
    full_df = pd.read_csv(df_path)

    # Create system dictionaries
    system_id_dict, system_inst_dict_cleaned = create_system_dicts(state_systems)

    # Extract position titles for each institution
    university_boards, double_boards = determine_board_position(full_df)

    # Create original first board dataframe
    original_single_boards, single_board_indices_start, single_board_indices_end, single_first_institution_index = find_word_grouping(full_df, university_boards)
    full_first_board = expand_board(full_df, original_single_boards, single_board_indices_start, single_board_indices_end, single_first_institution_index)
    full_first_board = clean_false_members(full_first_board, university_boards, original_single_boards)

    # Identify unmarked boards
    unmarked_boards = find_unmarked_boards(university_boards, full_first_board)

    # Process second boards
    original_double_boards, double_board_indices_start, double_board_indices_end, double_first_institution_index = find_word_grouping(full_df, double_boards)
    full_second_board = expand_double_board(full_df, original_double_boards, double_board_indices_start, double_board_indices_end, double_first_institution_index, double_boards, single_board_indices_start, single_board_indices_end)
    full_second_board = clean_false_members(full_second_board, double_boards, original_double_boards)

    # Process substring-based board positions
    substring_boards, substring_board_indices_start, substring_board_indices_end, substring_first_institution_index = find_word_grouping_substring(full_df, unmarked_boards)
    full_substring_board = expand_board(full_df, substring_boards, substring_board_indices_start, substring_board_indices_end, substring_first_institution_index)
    full_substring_board = clean_false_members(full_substring_board, unmarked_boards, substring_boards)

    # Validate substring board ordering
    validated_substring_board = verify_ordering_entire_board(full_substring_board)
    validated_substring_df, validated_substring_indices_start, validated_substring_indices_end, validated_substring_first_inst_index = find_word_grouping_substring(validated_substring_board, unmarked_boards)
    validated_substring_board = expand_board(validated_substring_board, validated_substring_df, validated_substring_indices_start, validated_substring_indices_end, validated_substring_first_inst_index)
    validated_substring_board = clean_false_members(validated_substring_board, unmarked_boards, validated_substring_df)

    # Process director boards
    director_common, double_directors = determine_director_schools(full_df)
    director_inst_boards = {
        key: value for key, value in director_common.items()
        if key in university_boards and university_boards[key] is None and value is not None
    }
    director_df = full_df[full_df['Institution'].isin(director_inst_boards.keys())]

    director_boards, director_board_indices_start, director_board_indices_end, director_first_institution_index = find_word_grouping(director_df, director_inst_boards)
    full_director_board = expand_directors(full_df, director_boards, director_board_indices_start, director_board_indices_end, director_first_institution_index)
    full_director_board = clean_false_members_directors(full_director_board)

    # Validate director board ordering
    validated_director_board = verify_ordering_entire_board(full_director_board)
    validated_director_df, removed_director_indices_start, removed_director_indices_end, removed_director_institution_index = find_word_grouping(validated_director_board, director_inst_boards)
    final_director_board = expand_directors(validated_director_board, validated_director_df, removed_director_indices_start, removed_director_indices_end, removed_director_institution_index)
    final_director_board = clean_false_members_directors(final_director_board)

    # Group and filter director boards
    grouped_dict = {key: value for key, value in final_director_board.groupby('Institution')}
    keys_to_remove = [institution for institution, group in grouped_dict.items() if group['Position'].str.contains('Director,', case=False).any()]
    for key in keys_to_remove:
        del grouped_dict[key]
    final_director_board = pd.concat(grouped_dict.values()).reset_index(drop=True)

    # Mark institutions with director boards
    for x in final_director_board["Institution"].values:
        university_boards[x] = "Director"

    # Combine single, substring, and director boards
    combined_single_boards = pd.concat([full_first_board, validated_substring_board, final_director_board], ignore_index=True)
    combined_single_boards = combined_single_boards.groupby("Institution").filter(lambda x: len(x) >= 4).reset_index(drop=True)
    combined_single_boards.sort_values(by="Institution", inplace=True)

    # Validate and clean double boards
    board_dict = assemble_board_dict(combined_single_boards)
    invalid_double_boards = validate_double_boards(board_dict, full_second_board)
    full_second_board = full_second_board[~full_second_board["Institution"].isin(invalid_double_boards)].reset_index(drop=True)

    # Mark state systems and add additional attributes
    combined_single_boards["StateSystem"] = ""
    for index, row in combined_single_boards.iterrows():
        if row["AffiliationId"] in system_id_dict:
            combined_single_boards.at[index, "StateSystem"] = system_id_dict[row["AffiliationId"]]

    # Prepare lists for final checks
    id_name_dict = {row["Institution"].replace("-", " ").replace(",", "").replace(".", ""): row["AffiliationId"] for _, row in full_df.iterrows()}
    all_board_ids = list(set(np.concatenate((combined_single_boards["AffiliationId"].values, full_second_board["AffiliationId"].values))))
    all_board_names = list(set(np.concatenate((combined_single_boards["Institution"].values, full_second_board["Institution"].values))))
    all_board_names_cleaned = [name.replace("-", " ").replace(",", "").replace(".", "") for name in all_board_names]

    missing_institutions = [inst for inst, id in id_name_dict.items() if id not in all_board_ids and inst not in all_board_names_cleaned and id not in system_id_dict and inst not in system_inst_dict_cleaned]

    return combined_single_boards, full_second_board, missing_institutions, university_boards


state_systems_path = f"{temporary_path}state_systems_validated.csv"
state_systems = pd.read_csv(state_systems_path)

for year in years:
    print(year)
    combined_single_boards, full_second_board, missing_institutions, university_boards = process_year_data(year, split_path, state_systems)
    # Save outputs
    marked_boards_single_df = mark_members(combined_single_boards, university_boards).drop_duplicates(keep=False)
    marked_boards_single_df.to_csv(f"{boards_path}{year}_single_board.csv", index=False)
    marked_boards_double_df = mark_members(full_second_board, university_boards).drop_duplicates(keep=False)
    marked_boards_double_df.to_csv(f"{boards_path}{year}_double_board.csv", index=False)
