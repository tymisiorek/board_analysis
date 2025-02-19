import pandas as pd
import os
import json
import numpy as np


POSITION_BANK = ["President", "Chancellor", "Provost", "Director", "Dean", "Controller", "Trustee", "Member", "Regent", "Chairman", "Overseer", "Assistant", "Librarian", "Secretary", "Chaplain", "Minister", "Treasurer", "Senior Counsel", "General Counsel", "Legal Counsel", "University Counsel", "College Counsel", "Special Counsel", "Corporation Counsel", "Officer", "Chief", "Professor", "Commissioner", "Fellow", "Chairperson", "Manager", "Clergy", "Coordinator", "Auditor", "Governor", "Representative", "Stockbroker", "Advisor", "Commandant", "Rector", "Attorney", "Curator", "Clerk", "Department Head", "Pastor", "Head", "Comptroller", "Deputy", "Inspector General"]
#for classifying position as dean, administration
DEAN_WORDS = ["summer", "student", "faculty", "academic service", "academics", "academic program", "admissions", "admission", "enrollment", "student life", "housing", "academic support", "advising", "enrollment management", 
                       "student relations", "academic computing", "academic records", "student service", "student affairs", "student development", "registrar", "financial aid", "media service", "library service", "university librar",
                       "internation affair", "special program", "continuing education", "external relation", "development", "services"]

#for subinstituion
ADMINISTRATION_WORDS = ["academic service", "academics", "academic program", "admissions", "admission", "enrollment service", "student life", "housing", "academic support", "advising", "enrollment management", 
                       "student relations", "academic computing", "academic records", "student service", "student affairs", "student development", "registrar", "financial aid", "media service", "library service", "university librar"]


absolute_path = "C:\\Users\\tykun\\\OneDrive\\Documents\\SchoolDocs\VSCodeProjects\\connectedData\\board_analysis\\"
altered_dataframes = "altered_dataframes\\"
gpt_dataframes = "gpt_dataframes\\"
graphs = "graphs\\"
scripts =  "scripts\\"
board_dataframes = "board_dataframes\\"
split_dataframes = "split_dataframes\\"


years = ["1999", "2000", "2005", "2007", "2008", "2009", "2010", "2011", "2013", "2018"]


#Extract the names of all the institutions for validation
def extract_institutions(df):
    institution_list = []
    for index, row in df.iterrows():
        if row["Institution"] not in institution_list:
            institution_list.append(row["Institution"])
    return institution_list


def extract_first_member(df):
    first_member_df = []
    previous_institution = None
    for index, row in df.iterrows():
        current_institution = row["Institution"]
        if previous_institution is None or current_institution != previous_institution:
            first_member_df.append(row)
        previous_institution = current_institution
    return pd.DataFrame(first_member_df)

#extract first person from each institution
def extract_first_member_exclude_string(df, string):
    string = string.lower()
    first_member_df = []
    previous_institution = None
    for index, row in df.iterrows():
        current_institution = row["Institution"]
        # Check if the value at index 2 has changed, including handling NaN for the first row
        if previous_institution is None or current_institution != previous_institution:
            if string not in row["Position"].lower():
                first_member_df.append(row)
        previous_institution = current_institution
    return pd.DataFrame(first_member_df)

#Replace all the entries of a column with a standardized value (for president)
def replace_values(df, string):
    df["Position"] = string
    return df 

#Find the institutions that were left out
def find_missing_institutions(institutions, df):
    lowercase_df = df["Institution"].str.lower()
    missing_institutions = []
    for institution in institutions:
        if institution.lower() not in lowercase_df.values:
            missing_institutions.append(institution)
    return missing_institutions


def rename_previous_value(df):
    df.rename(columns={'Previous_Value': 'Fixed Position'}, inplace = True)
    df["Fixed Position"] = ""
    return pd.DataFrame(df)


#validate university list
def count_universities(full_df):
    institutions = full_df["Institution"].unique()
    return list(institutions)

#Presidents
def extract_presidents(df):
    member_list = []
    for index, row in df.iterrows():
        if row["Position"] == "President":
            member_list.append(row['Name'])
        elif row["Position"] == "Chancellor":
            member_list.append(row['Name'])
        elif "Commissioner Of Higher Education" in row["Position"].title():
            member_list.append(row['Name'])
    return member_list

#check if the first person is president and the second is the chancellor (need to decide how to handle these two)
def extract_president_and_chancellor(president_df, df):
    previous_institution = None
    for index, row in df.iterrows():
        current_institution = row["Institution"]
        if previous_institution is None or current_institution != previous_institution:
            if index + 1 < len(df):
                next_row = df.iloc[index + 1]
                if "president" in row["Position"].lower() and "chancellor" == next_row["Position"].lower():
                    president_df = pd.concat([president_df, next_row.to_frame().T], ignore_index=True)
        previous_institution = current_institution
    return president_df

#remove any non presidents/chancellors
def clean_president_list(df):
    cleaned_df = []
    for index, row in df.iterrows():
        if 'president' in row["Position"].lower() or 'chancellor' in row["Position"].lower() or 'director' in row["Position"].lower() or 'superintendent' in row["Position"].lower():
            #Don't want vice president of the university, but exceptions
            if 'vice president' not in row["Position"].lower():
                cleaned_df.append(row)
            elif row["Position"].lower().count('president') >= 2:
                cleaned_df.append(row)
            elif 'vice president' in row["Position"].lower() and 'chancellor' in row["Position"].lower():
                cleaned_df.append(row)
        elif "Commissioner Of Higher Education" in row["Position"].title():
            cleaned_df.append(row)
    cleaned_df = pd.DataFrame(cleaned_df)
    cleaned_df = cleaned_df.map(lambda x: x.title() if isinstance(x, str) else x)

    # cleaned_df = cleaned_df.drop_duplicates(subset = ["Name"], keep = "first")
    cleaned_df = cleaned_df.drop_duplicates(subset = ["Institution"], keep = "first")
    return cleaned_df


def create_presidents_df(full_df):
    # all_institutions = extract_institutions(full_df)
    presidents_initial = extract_first_member(full_df)
    presidents_cleaned = clean_president_list(presidents_initial)
    presidents_cleaned = extract_president_and_chancellor(presidents_cleaned, full_df)
    return pd.DataFrame(presidents_cleaned)

def determine_missing(university_list, presidents_df):
    president_institutions = list(presidents_df["Institution"])
    university_list_normalized = [uni.strip().lower() for uni in university_list]
    president_institutions_normalized = [president.strip().lower() for president in president_institutions]
    missing_list = [uni for uni in university_list_normalized if uni not in president_institutions_normalized]
    return missing_list

for year in years:
    print(f"Processing: {year}")
    path_read = f"{absolute_path}{split_dataframes}{year}_split_positions.csv"
    president_path = f"{absolute_path}{altered_dataframes}{year}_presidents.csv"

    full_dataframe = pd.read_csv(path_read)

    presidents_df = create_presidents_df(full_dataframe)
    university_list = count_universities(full_dataframe)
    missing_presidents = determine_missing(university_list, presidents_df)

    for inst in missing_presidents:
        print(inst.title())

    presidents_df["FixedPosition"] = "President"
    presidents_df.to_csv(president_path, index = False)

