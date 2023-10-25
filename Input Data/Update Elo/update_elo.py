
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', None)
pd.set_option('mode.chained_assignment', None)

# input data files
ELO = "Elo By Year.xlsx"
SCHEDULE = "CFB_Sch_22-23.xlsx"
CONFERENCES = "../Conferences.xlsx"
YEAR = 2022

""" VARIABLE INPUTS """
# matchup elo adjustments
HOME = 30  # adjustment for home advantage
K = 125
MOL_FAC = 1.5  # margin of loss factor
MOL_BAR = 10  # margin of loss barrier


def fill_elo(x, team_ind_elo, last_elo):
    """ Fill in the elo for the given team or fill with 750 if not in the data """
    try:
        return team_ind_elo.loc[x][last_elo.columns[-1]]
    except KeyError:
        return 750


def calc_win_prob(away_elo, home_elo, neutral):
    """ Calculate the win probability of team 1 given the elo of two teams """
    # checks if the game is at a neutral site
    if neutral == 1:
        away_win_prob = 1 / (1 + (10 ** ((home_elo - away_elo) / 300)))
    # applies home field advantage to home team if not at a neutral site
    else:
        away_win_prob = 1 / (1 + (10 ** ((home_elo - away_elo + HOME) / 300)))

    return away_win_prob


def update_elo(starting_elo, team_win_perc, team_win, mov):
    """ Update the elo for all the teams that played that week based on the results of their games """
    if mov > MOL_BAR and team_win == 0:
        updated_elo = starting_elo + (K * (team_win - team_win_perc) * MOL_FAC)
    else:
        updated_elo = starting_elo + (K * (team_win - team_win_perc))

    return updated_elo


def one_week_sim(last_elo, week_sch):
    """
    Simulate one week of the college football season
    """
    # add the elo for both the home and away team to be able to make the calculation
    team_ind_elo = last_elo.set_index('Team')
    week_sch['Away_Elo'] = week_sch['Away'].apply(lambda x: fill_elo(x, team_ind_elo, last_elo))
    week_sch['Home_Elo'] = week_sch['Home'].apply(lambda x: fill_elo(x, team_ind_elo, last_elo))

    # create a new column with the win percentage using the function
    week_sch['Away_Win%'] = week_sch.apply(lambda x: calc_win_prob(x['Away_Elo'], x['Home_Elo'], x['Neutral']), axis=1)
    week_sch['Home_Win%'] = week_sch['Away_Win%'].apply(lambda x: 1 - x)

    # update the elo for the teams after the week has ended
    week_sch['Away_New_Elo'] = week_sch.apply(lambda x: update_elo(x['Away_Elo'], x['Away_Win%'], x['Away_Win'],
                                                                   x['MOV']), axis=1)
    week_sch['Home_New_Elo'] = week_sch.apply(lambda x: update_elo(x['Home_Elo'], x['Home_Win%'], x['Home_Win'],
                                                                   x['MOV']), axis=1)

    # create a week results dataframe to add to last_elo
    week = week_sch["Week"].iloc[0]
    away_results = week_sch[['Away', 'Away_New_Elo']]
    away_results.columns = ['Team', f'Week_{week}_Elo']
    home_results = week_sch[['Home', 'Home_New_Elo']]
    home_results.columns = ['Team', f'Week_{week}_Elo']
    week_results = pd.concat([away_results, home_results])

    # update last_elo with the new elo for teams
    last_elo = pd.merge(last_elo, week_results[['Team', f'Week_{week}_Elo']], on='Team', how='left')

    # fill in elo for teams that didn't play that week in last_elo
    if week == 1:
        prev_week_elo = 'Starting_Elo'
    elif week == 'Conf_Champ':
        prev_week_elo = last_elo.columns[-2]
    else:
        prev_week_elo = f'Week_{week - 1}_Elo'

    last_elo[f'Week_{week}_Elo'] = last_elo.apply(
        lambda x: x[f'Week_{week}_Elo'] if ~np.isnan(x[f'Week_{week}_Elo']) else x[prev_week_elo], axis=1)

    return last_elo


def eos_elo(elo, fbs):
    """ Update the end of season elo based on if the team is an FBS and move the elo closer to 1500 if so and closer
    to 750 if not """
    if fbs:
        adjusted_elo = (elo * 0.5) + (1500 * 0.5)
    else:
        adjusted_elo = (elo * 0.5) + (750 * 0.5)

    return adjusted_elo


def season_sim(elo_df, sch_df, conf_df):
    """
    Run the calculations for the season
    """
    # pull the most recent elo data
    last_elo_yr = list(elo_df.columns)[-1]
    last_elo = elo_df[['Team', last_elo_yr]]
    last_elo.columns = ['Team', 'Starting_Elo']

    # create a list of the difference weeks
    weeks = sch_df['Week'].unique()

    # loop through each of the weeks in the schedule
    for week in weeks:

        # get the schedule of games for that week
        week_sch = sch_df[sch_df['Week'] == week]

        # run the simulation for one week of the regular season
        last_elo = one_week_sim(last_elo, week_sch)

    # adjust the final elo to shift back towards 750 or 1500 depending on the school
    fbs_teams = list(conf_df['School'])
    last_elo['FBS'] = last_elo['Team'].apply(lambda x: True if x in fbs_teams else False)
    last_elo['Final_Elo'] = last_elo.apply(lambda x: round(eos_elo(x[list(last_elo.columns)[-2]], x['FBS']), 2), axis=1)

    return last_elo


def main():

    # load in the data
    elo_df = pd.read_excel(ELO)
    sch_df = pd.read_excel(SCHEDULE)
    conf_df = pd.read_excel(CONFERENCES)

    # run the season calculations to get the end of season data
    last_elo = season_sim(elo_df, sch_df, conf_df)

    # get the year from the input data and add the year to the elo data
    elo_df[str(YEAR)] = last_elo['Final_Elo']

    # save the elo data into the same excel path with the new column added on
    elo_df.to_excel(ELO, index=False)


main()
