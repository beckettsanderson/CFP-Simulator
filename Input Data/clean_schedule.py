
import pandas as pd
import datetime

# input variables
SCHEDULE = "Update Elo/CFB_Sch_23-24.csv"
TYPE = 'Current'

pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None  # default='warn'


def past_schedule(sch):

    # create columns for whether home or away team won and MOV
    sch['MOV'] = sch['Home_Pts'] - sch['Away_Pts']
    sch['Home_Win'] = 1
    sch['Away_Win'] = 0

    # remove the columns that are not needed
    sch.drop(['Game', 'Date', 'Time', 'Day', 'Home_Pts', 'Location', 'Away_Pts'], axis=1, inplace=True)

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
    sch.drop(['Game', 'Date', 'Time', 'Day', 'Home_Pts', 'Location', 'Away_Pts', 'Notes'], axis=1, inplace=True)

    # swap the necessary home and away teams
    idx = (sch['Switch_Teams'] == 1)
    sch.loc[idx] = sch.loc[idx].rename(columns={'Home': 'Away', 'Away': 'Home'})

    # remove the rankings in front of teams
    sch['Home'] = sch['Home'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)
    sch['Away'] = sch['Away'].apply(lambda x: ' '.join(x.split()[1:]) if x.split()[0].startswith("(") else x)

    # drop switch teams column
    sch.drop(['Switch_Teams'], axis=1, inplace=True)

    return sch


def current_schedule(sch):

    # create the datetime object for the most recent Monday at 4 AM
    today = datetime.datetime.now().replace(hour=4, minute=00, second=00, microsecond=00)
    last_monday = today - datetime.timedelta(days=today.weekday())

    # clean the schedule to have datetime
    sch.replace('TBD', '1:00 PM', inplace=True)
    # create datetime column
    sch['Datetime'] = sch.apply(lambda x: x['Date'] + ' ' + x['Time'], axis=1)
    date_format = "%b %d %Y %I:%M %p"
    sch['Datetime'] = sch['Datetime'].apply(lambda x: datetime.datetime.strptime(x, date_format))

    # split the schedule on dates before and after the last monday datetime
    past_sch = sch[sch['Datetime'] < last_monday]
    new_sch = sch[sch['Datetime'] > last_monday]

    # clean the two filtered dataframes after dropping additions
    past_sch.drop(['Datetime'], axis=1, inplace=True)
    new_sch.drop(['Datetime'], axis=1, inplace=True)

    # create the new schedules after the split
    completed_sch = past_schedule(past_sch)
    upcoming_sch = new_schedule(new_sch)

    return completed_sch, upcoming_sch


def main():

    # read in the file and rename the columns
    sch = pd.read_csv(SCHEDULE)
    sch.columns = ['Game', 'Week', 'Date', 'Time', 'Day', 'Home', 'Home_Pts', 'Location', 'Away', 'Away_Pts', 'Notes']

    # create new columns for neutral and whether to switch the home and away team
    sch['Neutral'] = sch['Location'].apply(lambda x: 1 if x == 'N' else 0)
    sch['Switch_Teams'] = sch['Location'].apply(lambda x: 1 if x == '@' else 0)

    # create the schedule and save it to an Excel file with the same name as the csv
    if TYPE.upper() == "PAST":
        past_schedule(sch).to_excel(SCHEDULE.split(".")[0] + ".xlsx", index=False)
    elif TYPE.upper() == "NEW":
        new_schedule(sch).to_excel(SCHEDULE.split(".")[0] + ".xlsx", index=False)
    elif TYPE.upper() == "CURRENT":
        completed_sch, upcoming_sch = current_schedule(sch)
        completed_sch.to_excel(SCHEDULE.split(".")[0] + " (Completed).xlsx", index=False)
        upcoming_sch.to_excel(SCHEDULE.split(".")[0] + " (Upcoming).xlsx", index=False)
    else:
        print("Invalid type, please try again.")


main()
