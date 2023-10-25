
import pandas as pd

SCHEDULE = "Update Elo/CFP_Sch_22-23.csv"

pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', None)


def past_schedule(sch):

    # create columns for whether home or away team won and MOV
    sch['MOV'] = sch['Home_Pts'] - sch['Away_Pts']
    sch['Home_Win'] = 1
    sch['Away_Win'] = 0

    # remove the columns that are not needed
    sch.drop(['Game', 'Home_Pts', 'Location', 'Away_Pts', 'Notes'], axis=1, inplace=True)

    # swap the necessary home and away teams and home and away wins
    idx = (sch['Switch_Teams'] == 1)
    sch.loc[idx] = sch.loc[idx].rename(columns={'Home': 'Away', 'Away': 'Home'})
    sch.loc[idx] = sch.loc[idx].rename(columns={'Home_Win': 'Away_Win', 'Away_Win': 'Home_Win'})

    # remove the rankings in front of teams
    sch['Home'] = sch['Home'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)
    sch['Away'] = sch['Away'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)

    # drop switch teams column
    sch.drop(['Switch_Teams'], axis=1, inplace=True)

    return sch


def new_schedule(sch):

    # remove the columns that are not needed
    sch.drop(['Game', 'Home_Pts', 'Location', 'Away_Pts', 'Notes'], axis=1, inplace=True)

    # swap the necessary home and away teams
    idx = (sch['Switch_Teams'] == 1)
    sch.loc[idx] = sch.loc[idx].rename(columns={'Home': 'Away', 'Away': 'Home'})

    # remove the rankings in front of teams
    sch['Home'] = sch['Home'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)
    sch['Away'] = sch['Away'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)

    # drop switch teams column
    sch.drop(['Switch_Teams'], axis=1, inplace=True)

    return sch


def main():

    # read in the file and rename the columns
    sch = pd.read_csv(SCHEDULE)
    sch.columns = ['Game', 'Week', 'Home', 'Home_Pts', 'Location', 'Away', 'Away_Pts', 'Notes']

    # create new columns for neutral and whether to switch the home and away team
    sch['Neutral'] = sch['Location'].apply(lambda x: 1 if x == 'N' else 0)
    sch['Switch_Teams'] = sch['Location'].apply(lambda x: 1 if x == '@' else 0)

    # create the schedule and save it to an Excel file with the same name as the csv
    new_sch = past_schedule(sch)
    new_sch.to_excel(SCHEDULE.split(".")[0] + ".xlsx", index=False)


main()
