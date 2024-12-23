import pandas as pd
import numpy as np
from collections import Counter
from nameparser import HumanName

class Config:
    """
    Configuration holder to group global constants and paths.
    """
    # Global position-related lists
    POSITION_BANK = [
        "President", "Chancellor", "Provost", "Director", "Dean", "Controller", "Trustee", "Member", "Regent",
        "Chairman", "Overseer", "Assistant", "Librarian", "Secretary", "Chaplain", "Minister", "Treasurer",
        "Senior Counsel", "General Counsel", "Legal Counsel", "University Counsel", "College Counsel", "Special Counsel",
        "Corporation Counsel", "Officer", "Chief", "Professor", "Commissioner", "Fellow", "Chairperson", "Manager",
        "Clergy", "Coordinator", "Auditor", "Governor", "Representative", "Stockbroker", "Advisor", "Commandant",
        "Rector", "Attorney", "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy",
        "Inspector General"
    ]

    NON_BOARD_WORDS = [
        "President", "Chancellor", "Provost", "Dean", "Controller", "Overseer", "Assistant", "Librarian", "Secretary",
        "Chaplain", "Minister", "Treasurer", "Senior Counsel", "General Counsel", "Legal Counsel", "University Counsel",
        "College Counsel", "Special Counsel", "Corporation Counsel", "Officer", "Chief", "Professor", "Commissioner",
        "Manager", "Clergy", "Coordinator", "Auditor", "Representative", "Stockbroker", "Advisor", "Commandant",
        "Rector", "Attorney", "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy",
        "Inspector General", "Vice", "Chancellor,", "President,", "Executive", "Affairs", "Senior", "Associate",
        "Administration", "University", "College"
    ]

    BOARD_WORDS = [
        "Trustee", "Regent", "Member", "Fellow", "Overseer", "Governor", "Curator", "Visitor", "Manager"
    ]

    OTHER_BOARD_WORD = [
        "President", "Chairman", "Chairperson", "Treasurer", "Rector", "Member", "Secretary", "Ex Officio"
    ]

    # Paths
    ABSOLUTE_PATH = "C:\\Users\\tykun\\OneDrive\\Documents\\SchoolDocs\\VSCodeProjects\\connectedData\\board_analysis\\"
    ALTERED_DATAFRAMES = "altered_dataframes\\"
    GPT_DATAFRAMES = "gpt_dataframes\\"
    GRAPHS = "graphs\\"
    SCRIPTS = "scripts\\"
    BOARD_DATAFRAMES = "board_dataframes\\"
    SPLIT_DATAFRAMES = "split_dataframes\\"
    TEMPORARY = "temporary_data\\"

    ALTERED_DATAFRAME_PATH = f"{ABSOLUTE_PATH}{ALTERED_DATAFRAMES}"
    GPT_DATAFRAME_PATH = f"{ABSOLUTE_PATH}{GPT_DATAFRAMES}"
    GRAPH_PATH = f"{ABSOLUTE_PATH}{GRAPHS}"
    SCRIPT_PATH = f"{ABSOLUTE_PATH}{SCRIPTS}"
    BOARDS_PATH = f"{ABSOLUTE_PATH}{BOARD_DATAFRAMES}"
    SPLIT_PATH = f"{ABSOLUTE_PATH}{SPLIT_DATAFRAMES}"
    TEMPORARY_PATH = f"{ABSOLUTE_PATH}{TEMPORARY}"

    # Valid Years
    YEARS = ["1999", "2000", "2005", "2007", "2008", "2009", "2011", "2013", "2018"]


# Additional global definitions reused by mark_members and others
CHAIRPERSONS = ["Chairman", "Chairperson", "President", "Chair", "Chancellor"]
MEMBERS = ["Trustee", "Regent", "Member", "Fellow", "Overseer", "Governor", "Curator", "Visitor", "Manager", "Director"]
OTHER_BOARD_TAGS = [
    "Treasurer", "Faculty Representative", "Rector", "Secretary", "Counsel", "Clerk", "Vacant",
    "Executive Committee Member", "Special", "Student", "Chief Executive Officer", "Affiliation",
    "Justice", "Registrar", "Staff Representative", "Librarian", "Alumni Representative", "Faculty Visitor",
    "Chief Investment Officer"
]


def higher_ascii(char1, char2):
    """Checks if char1 >= char2 in ASCII (case-insensitive)."""
    return ord(char1.upper()) >= ord(char2.upper())


def parse_name(raw_name):
    """Parses a raw name string using HumanName and returns the last name."""
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
    """
    institution_list = []
    for index, row in df.iterrows():
        if row["Institution"] not in institution_list:
            institution_list.append(row["Institution"])
    print(institution_list)
    return institution_list


def determine_board_position(df):
    """
    Determines the most common board position names for each institution and identifies multiple boards.
    """
    most_frequent = {}
    two_boards = {}
    grouped_df = df.groupby("Institution")

    for key, value in grouped_df:
        word_count = Counter()
        for position in value["Position"]:
            if not pd.isna(position):
                individual_words = position.split()
            else:
                individual_words = ""
            filtered_words = [word for word in individual_words if word in Config.BOARD_WORDS]
            word_count.update(filtered_words)

        if word_count:
            common_words = word_count.most_common(2)
            most_frequent[key] = common_words[0][0]
            if len(common_words) >= 2:
                large_enough_board = common_words[1][1] >= (common_words[0][1] / 4) or (common_words[0][1] >= 8)
            else:
                large_enough_board = False

            if len(common_words) >= 2 and large_enough_board and common_words[1][0] != "Director":
                if common_words[1][0].strip().lower() != common_words[0][0].strip().lower():
                    two_boards[key] = common_words[1][0]
            else:
                two_boards[key] = None
        else:
            most_frequent[key] = None
            two_boards[key] = None

    return most_frequent, two_boards


def determine_director_schools(df):
    """
    Determines institutions where 'Director' is common and identifies if there are multiple boards.
    """
    most_frequent = {}
    two_boards = {}
    grouped_df = df.groupby("Institution")

    for key, value in grouped_df:
        word_count = Counter()
        for position in value["Position"]:
            if isinstance(position, str):
                individual_words = position.split()
            else:
                individual_words = []
            board_words_dir = ['Director']
            filtered_words = [word for word in individual_words if word in board_words_dir]
            word_count.update(filtered_words)

        if word_count:
            common_words = word_count.most_common(2)
            most_frequent[key] = common_words[0][0]
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
    Finds the start and end indices of each board in the given dataframe, based on the board_name dictionary.
    """
    grouped_df = df.groupby("Institution")
    first_board_occurrence, last_board_occurrence, first_institution_occurrence = {}, {}, {}
    result_dataframe = []

    for key, value in grouped_df:
        board_position = board_name.get(key, None)
        first_institution_occurrence[key] = value.index[0]

        if board_position is not None:
            all_members = [str(pos) if pd.notna(pos) else "" for pos in value["Position"]]
            try:
                first_index = next(i for i, pos in enumerate(all_members) if board_position == pos.title())
                last_index = len(all_members) - next(i for i, pos in enumerate(reversed(all_members)) if board_position in pos.title()) - 1
                first_board_occurrence[key] = value.index[first_index]
                last_board_occurrence[key] = value.index[last_index]
                result_dataframe.append(value.iloc[first_index:last_index + 1])
            except StopIteration:
                first_board_occurrence[key] = value.index[0]
                last_board_occurrence[key] = value.index[-1]
        else:
            first_board_occurrence[key] = value.index[0]
            last_board_occurrence[key] = value.index[-1]

    if result_dataframe:
        return pd.concat(result_dataframe), first_board_occurrence, last_board_occurrence, first_institution_occurrence
    else:
        print("Warning: No relevant board sections found in the dataframe.")
        return pd.DataFrame(), first_board_occurrence, last_board_occurrence, first_institution_occurrence


def find_word_grouping_substring(df, board_name):
    """
    Finds the start and end indices of each board in the dataframe using substring matching for board names.
    """
    grouped_df = df.groupby("Institution")
    first_board_occurrence, last_board_occurrence, first_institution_occurrence = {}, {}, {}
    result_dataframe = []

    for key, value in grouped_df:
        board_position = board_name.get(key, None)
        first_institution_occurrence[key] = value.index[0]

        if board_position is not None:
            all_members = [str(pos) if pd.notna(pos) else "" for pos in value["Position"]]
            try:
                first_index = next(i for i, pos in enumerate(all_members) if board_position in pos.title())
                last_index = len(all_members) - next(i for i, pos in enumerate(reversed(all_members)) if board_position in pos.title()) - 1
                first_board_occurrence[key] = value.index[first_index]
                last_board_occurrence[key] = value.index[last_index]
                result_dataframe.append(value.iloc[first_index:last_index + 1])
            except StopIteration:
                first_board_occurrence[key] = value.index[0]
                last_board_occurrence[key] = value.index[-1]
        else:
            first_board_occurrence[key] = value.index[0]
            last_board_occurrence[key] = value.index[-1]

    return pd.concat(result_dataframe), first_board_occurrence, last_board_occurrence, first_institution_occurrence


def verify_ordering(full_df, current_board_start, count):
    """
    Verifies if names are in ascending alphabetical order when expanding a board upwards.
    """
    original_board_position = full_df.iloc[current_board_start]
    original_last_name = parse_name(original_board_position["Name"])

    while count > 0:
        current_position = full_df.iloc[current_board_start - count]
        current_last_name = parse_name(current_position["Name"])
        if not higher_ascii(original_last_name[0], current_last_name[0]):
            return False
        count -= 1
    return True


def expand_single_board_upward(full_df, grouped_boards, board_indices_start):
    """
    Expands a single board section upward by checking rows above the current board start index.
    """
    sorted_keys = sorted(board_indices_start.keys(), key=lambda k: board_indices_start[k])

    for key in sorted_keys:
        expanded_flag = False
        count = 4
        current_board_start = board_indices_start[key]

        while count > 0:
            previous_df_row = full_df.iloc[current_board_start - count]
            previous_row_position = previous_df_row["Position"].title()
            original_df_row = full_df.iloc[current_board_start]
            same_institution = original_df_row["Institution"] == key
            board_position = any(p in previous_row_position for p in Config.OTHER_BOARD_WORD)

            if same_institution and (board_position or expanded_flag):
                if not expanded_flag:
                    alphabetical_order = verify_ordering(full_df, current_board_start, count)
                else:
                    alphabetical_order = True

                if (alphabetical_order and "Dean" not in previous_row_position) or expanded_flag:
                    grouped_boards[key] = pd.concat([pd.DataFrame([previous_df_row]), grouped_boards.get(key, pd.DataFrame())])
                    expanded_flag = True

            count -= 1

    return grouped_boards


def expand_double_board_upward(full_df, grouped_boards, board_indices_start, double_boards):
    """
    Expands double board sections upwards by checking rows above the current board start index.
    """
    sorted_keys = sorted(board_indices_start.keys(), key=lambda k: board_indices_start[k])

    for key in sorted_keys:
        expanded_flag = False
        count = 4
        current_board_start = board_indices_start[key]

        while count > 0:
            previous_df_row = full_df.iloc[current_board_start - count]
            previous_row_position = previous_df_row["Position"].title()
            original_df_row = full_df.iloc[current_board_start]
            same_institution = original_df_row["Institution"] == key
            board_position = any(p in previous_row_position for p in Config.OTHER_BOARD_WORD)

            if key in double_boards and same_institution and (board_position or expanded_flag) and double_boards[key] is not None:
                if not expanded_flag:
                    alphabetical_order = verify_ordering(full_df, current_board_start, count)
                else:
                    alphabetical_order = True

                if (alphabetical_order and "Dean" not in previous_row_position) or expanded_flag:
                    grouped_boards[key] = pd.concat([pd.DataFrame([previous_df_row]), grouped_boards.get(key, pd.DataFrame())])
                    expanded_flag = True

            count -= 1

    return grouped_boards


def expand_single_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index):
    """
    Expands a single board section downward by checking rows below the current board end index.
    """
    sorted_keys = sorted(board_indices_end.keys(), key=lambda k: board_indices_end[k])

    for key in sorted_keys:
        if sorted_keys.index(key) < len(sorted_keys) - 1:
            current_board_end = board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index[next_key]

            if (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, next_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])

    return grouped_boards


def expand_double_board_downward(full_df, grouped_boards, double_board_indices_start, double_board_indices_end,
                                 first_institution_index, first_board_indices_start, first_board_indices_end, double_boards):
    """
    Expands double board sections downward by checking rows below the current board end index.
    """
    sorted_keys = sorted(double_board_indices_end.keys(), key=lambda k: double_board_indices_end[k])

    for key in sorted_keys:
        if key in double_boards and sorted_keys.index(key) < len(sorted_keys) - 1 and double_boards[key] is not None:
            current_board_end = double_board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index.get(next_key, float('inf'))
            current_board_start = double_board_indices_start[key]
            first_board_end = first_board_indices_end.get(key, -1)
            first_board_start = first_board_indices_start.get(key, -1)

            if (current_board_start > first_board_end) and (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, next_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])
            elif current_board_start < first_board_end and (current_board_end + 3 >= next_board_start and current_board_end + 1 != next_board_start):
                for index in range(current_board_end + 1, first_board_start):
                    grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])

    return grouped_boards


def expand_director_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index):
    """
    Expands director board sections downward by checking rows below the current board end index.
    """
    sorted_keys = sorted(board_indices_end.keys(), key=lambda k: board_indices_end[k])

    for key in sorted_keys:
        if sorted_keys.index(key) < len(sorted_keys) - 1:
            current_board_end = board_indices_end[key]
            next_key = sorted_keys[sorted_keys.index(key) + 1]
            next_board_start = first_institution_index[next_key]

            if current_board_end + 3 < next_board_start:
                next_board_start = current_board_end + 3

            for index in range(current_board_end + 1, next_board_start):
                grouped_boards[key] = pd.concat([grouped_boards.get(key, pd.DataFrame()), pd.DataFrame([full_df.iloc[index]])])
                index += 1

    return grouped_boards


def expand_board(full_df, board_df, board_indices_start, board_indices_end, first_institution_index):
    """
    Expands a single board section both upwards and downwards within the dataframe, removing duplicates.
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


def expand_double_board(full_df, board_df, board_indices_start, board_indices_end, first_institution_index, double_boards,
                        first_board_indices_start, first_board_indices_end):
    """
    Expands double board sections both upwards and downwards within the dataframe.
    """
    grouped_boards = {name: group for name, group in board_df.groupby("Institution")}
    grouped_boards = expand_double_board_upward(full_df, grouped_boards, board_indices_start, double_boards)
    grouped_boards = expand_double_board_downward(full_df, grouped_boards, board_indices_start, board_indices_end,
                                                  first_institution_index, first_board_indices_start, first_board_indices_end, double_boards)
    combined_boards = pd.concat(grouped_boards.values())
    return combined_boards


def expand_directors(full_df, board_df, board_indices_start, board_indices_end, first_institution_index):
    """
    Expands director board sections both upwards and downwards within the dataframe.
    """
    grouped_boards = {name: group for name, group in board_df.groupby("Institution")}
    grouped_boards = expand_single_board_upward(full_df, grouped_boards, board_indices_start)
    grouped_boards = expand_director_board_downward(full_df, grouped_boards, board_indices_end, first_institution_index)
    combined_boards = pd.concat(grouped_boards.values())
    return combined_boards


def assemble_board_dict(board_df):
    """
    Assembles a dictionary where each key is an institution and the value is the corresponding board subset.
    """
    board_dict = {}
    for institution in board_df['Institution'].unique():
        rows = board_df[board_df["Institution"] == institution]
        board_dict[institution] = rows
    return board_dict


def clean_false_members(expanded_boards, university_boards, original_boards):
    """
    Removes false members (Dean or Director) from the expanded board dataframe.
    """
    indices_to_drop = []
    for index, row in expanded_boards.iterrows():
        pos = row["Position"]
        pos = str(pos) if pd.notna(pos) else ""
        if "Dean" in pos or "Director" in pos:
            indices_to_drop.append(index)
    cleaned__df = expanded_boards.drop(index=indices_to_drop).reset_index(drop=True)
    return cleaned__df


def delete_overlap(primary_boards, secondary_boards):
    """
    Deletes rows in the secondary board dataframe that overlap with the primary board dataframe.
    """
    for index, row in secondary_boards.iterrows():
        if any(row.equals(primary_row) for _, primary_row in primary_boards.iterrows()):
            secondary_boards.drop(index, inplace=True)
    return secondary_boards


def validate_double_boards(board_dict, double_boards):
    """
    Validates if rows in double boards exist within the original board dictionary.
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
    Verifies alphabetical ordering of last names in the entire board dataframe,
    marking rows for removal if ordering issues are found.
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
            if not higher_ascii(current_last_name[0], previous_last_name[0]) and count <= 2:
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
    """
    indices_to_drop = []
    for index, row in df.iterrows():
        pos = row["Position"]
        if "Dean" in pos or "Director," in pos:
            indices_to_drop.append(index)
    cleaned__df = df.drop(index=indices_to_drop).reset_index(drop=False)
    return cleaned__df


def mark_members(board_df, university_boards):
    """
    Assigns a 'FixedPosition' label to each row in board_df based on the board name or other keywords.
    """
    board_df["FixedPosition"] = ""
    grouped_boards = board_df.groupby("Institution")

    for key, value in grouped_boards:
        for index, row in value.iterrows():
            position = row["Position"]
            position = position.title() if isinstance(position, str) else ""
            board_name = university_boards[key]
            pres_appears = any(pos in position for pos in CHAIRPERSONS)
            if board_name is None:
                board_name = "zZbkjlhz01"

            if board_name in position:
                board_df.at[index, "FixedPosition"] = board_name
            elif pres_appears and "Vice" not in position:
                board_df.at[index, "FixedPosition"] = "Board President"
            elif pres_appears and "Vice" in position:
                board_df.at[index, "FixedPosition"] = "Board Vice President"
            elif any(pos in position for pos in OTHER_BOARD_TAGS):
                board_df.at[index, "FixedPosition"] = "Other Board Member"
            else:
                board_df.at[index, "FixedPosition"] = board_name

            if "Ex Officio" in position:
                board_df.at[index, "FixedPosition"] += ", Ex Officio"

    return board_df


def clean_system_inst_dict(full_df, state_systems):
    """
    Cleans the system institution dictionary by removing dashes, commas, and periods from the keys.
    """
    id_dict = {}
    for index, row in full_df.iterrows():
        id_dict[row["Institution"]] = row["AffiliationId"]

    system_id_dict = {}
    system_inst_dict = {}
    for index, row in state_systems.iterrows():
        if not pd.isna(row["StateSystem"]):
            system_id_dict[row["AffiliationId"]] = row["StateSystem"]
            system_inst_dict[row["Institution"]] = row["StateSystem"]

    system_inst_dict_cleaned = {
        key.replace("-", " ").replace(",", "").replace(".", ""): value
        for key, value in system_inst_dict.items()
    }

    return system_id_dict, system_inst_dict, system_inst_dict_cleaned


def find_unmarked_boards(university_boards, full_first_board):
    """
    Identifies schools with a board where the board position doesn't appear as an exact string.
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
    """
    df_path = f"{split_path}{year}_split_positions.csv"
    full_df = pd.read_csv(df_path)

    system_id_dict, system_inst_dict_cleaned = create_system_dicts(state_systems)

    university_boards, double_boards = determine_board_position(full_df)

    original_single_boards, single_board_indices_start, single_board_indices_end, single_first_institution_index = find_word_grouping(full_df, university_boards)
    full_first_board = expand_board(full_df, original_single_boards, single_board_indices_start, single_board_indices_end, single_first_institution_index)
    full_first_board = clean_false_members(full_first_board, university_boards, original_single_boards)

    unmarked_boards = find_unmarked_boards(university_boards, full_first_board)

    original_double_boards, double_board_indices_start, double_board_indices_end, double_first_institution_index = find_word_grouping(full_df, double_boards)
    full_second_board = expand_double_board(
        full_df,
        original_double_boards,
        double_board_indices_start,
        double_board_indices_end,
        double_first_institution_index,
        double_boards,
        single_board_indices_start,
        single_board_indices_end
    )
    full_second_board = clean_false_members(full_second_board, double_boards, original_double_boards)

    substring_boards, substring_board_indices_start, substring_board_indices_end, substring_first_institution_index = find_word_grouping_substring(full_df, unmarked_boards)
    full_substring_board = expand_board(full_df, substring_boards, substring_board_indices_start, substring_board_indices_end, substring_first_institution_index)
    full_substring_board = clean_false_members(full_substring_board, unmarked_boards, substring_boards)

    validated_substring_board = verify_ordering_entire_board(full_substring_board)
    validated_substring_df, validated_substring_indices_start, validated_substring_indices_end, validated_substring_first_inst_index = find_word_grouping_substring(
        validated_substring_board, unmarked_boards
    )
    validated_substring_board = expand_board(
        full_substring_board, validated_substring_df, validated_substring_indices_start,
        validated_substring_indices_end, validated_substring_first_inst_index
    )
    validated_substring_board = clean_false_members(validated_substring_board, unmarked_boards, validated_substring_df)

    director_common, double_directors = determine_director_schools(full_df)
    director_inst_boards = {
        key: value for key, value in director_common.items()
        if key in university_boards and university_boards[key] is None and value is not None
    }
    director_df = full_df[full_df['Institution'].isin(director_inst_boards.keys())]

    director_boards, director_board_indices_start, director_board_indices_end, director_first_institution_index = find_word_grouping(director_df, director_inst_boards)
    full_director_board = expand_directors(full_df, director_boards, director_board_indices_start, director_board_indices_end, director_first_institution_index)
    full_director_board = clean_false_members_directors(full_director_board)

    validated_director_board = verify_ordering_entire_board(full_director_board)
    validated_director_df, removed_director_indices_start, removed_director_indices_end, removed_director_institution_index = find_word_grouping(
        validated_director_board, director_inst_boards
    )
    final_director_board = expand_directors(
        validated_director_board,
        validated_director_df,
        removed_director_indices_start,
        removed_director_indices_end,
        removed_director_institution_index
    )
    final_director_board = clean_false_members_directors(final_director_board)

    grouped_dict = {key: value for key, value in final_director_board.groupby('Institution')}
    keys_to_remove = [institution for institution, group in grouped_dict.items()
                      if group['Position'].str.contains('Director,', case=False).any()]
    for key in keys_to_remove:
        del grouped_dict[key]
    final_director_board = pd.concat(grouped_dict.values()).reset_index(drop=True)

    # Mark institutions with director boards
    for x in final_director_board["Institution"].values:
        university_boards[x] = "Director"

    combined_single_boards = pd.concat([full_first_board, validated_substring_board, final_director_board], ignore_index=True)
    combined_single_boards = combined_single_boards.groupby("Institution").filter(lambda x: len(x) >= 4).reset_index(drop=True)
    combined_single_boards.sort_values(by="Institution", inplace=True)

    board_dict = assemble_board_dict(combined_single_boards)
    invalid_double_boards = validate_double_boards(board_dict, full_second_board)
    full_second_board = full_second_board[~full_second_board["Institution"].isin(invalid_double_boards)].reset_index(drop=True)

    combined_single_boards["StateSystem"] = ""
    for index, row in combined_single_boards.iterrows():
        if row["AffiliationId"] in system_id_dict:
            combined_single_boards.at[index, "StateSystem"] = system_id_dict[row["AffiliationId"]]

    id_name_dict = {row["Institution"].replace("-", " ").replace(",", "").replace(".", ""): row["AffiliationId"] for _, row in full_df.iterrows()}
    all_board_ids = list(set(np.concatenate((combined_single_boards["AffiliationId"].values, full_second_board["AffiliationId"].values))))
    all_board_names = list(set(np.concatenate((combined_single_boards["Institution"].values, full_second_board["Institution"].values))))
    all_board_names_cleaned = [name.replace("-", " ").replace(",", "").replace(".", "") for name in all_board_names]

    missing_institutions = [
        inst for inst, id in id_name_dict.items()
        if id not in all_board_ids and inst not in all_board_names_cleaned and id not in system_id_dict and inst not in system_inst_dict_cleaned
    ]

    return combined_single_boards, full_second_board, missing_institutions, university_boards


def main():
    state_systems_path = f"{Config.TEMPORARY_PATH}state_systems_validated.csv"
    state_systems = pd.read_csv(state_systems_path)

    for year in Config.YEARS:
        print(year)
        combined_single_boards, full_second_board, missing_institutions, university_boards = process_year_data(year, Config.SPLIT_PATH, state_systems)
        marked_boards_single_df = mark_members(combined_single_boards, university_boards).drop_duplicates(keep=False)
        marked_boards_single_df.to_csv(f"{Config.BOARDS_PATH}{year}_boards.csv", index=False)

        marked_boards_double_df = mark_members(full_second_board, university_boards).drop_duplicates(keep=False)
        marked_boards_double_df.to_csv(f"{Config.BOARDS_PATH}{year}_double_board.csv", index=False)


if __name__ == "__main__":
    main()
