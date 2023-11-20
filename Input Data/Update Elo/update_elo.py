
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('mode.chained_assignment', None)

# input data files
ELO = "Elo By Year.xlsx"
SCHEDULE = "CFB_Sch_23-24 (Completed).xlsx"
CONFERENCES = "../Conferences (Old).xlsx"
YEAR = '2023 - Temp'
CURRENT_SEASON = True

""" VARIABLE INPUTS """
# matchup elo adjustments
HOME = 30  # adjustment for home advantage
K = 125
MOL_FAC = 1.5  # margin of loss factor
MOL_BAR = 10  # margin of loss barrier

# end of season elo adjustments
ZERO_L = 62.5  # reward for having zero losses
ONE_L = 62.5  # reward for having one loss
TWO_L = -31.25  # penalty for having two losses
THREE_L = -93.75  # penalty for having three losses
NON_P5 = -125  # penalty for not being in a Power 5 conference
TITLE_GAME = 41  # bonus for playing in a title game

# conference championship variable list for determining games
CONF_CHIP_LIST = ['Conference USA Championship', 'Pac-12 Championship', 'SEC Championship', 'Big 12 Championship',
                  'American Athletic Conference Championship', 'Sun Belt Championship', 'Big Ten Championship',
                  'MAC Championship', 'ACC Championship', 'MWC Championship']


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


def add_wins(season_wins, week_results):
    """ Updates the season wins if the team won that week """
    if week_results is not np.NAN and week_results == 1:
        season_wins += 1
    return season_wins


def add_losses(season_losses, week_results):
    """ Updates the season losses if the team lost that week """
    if week_results is not np.NAN and week_results == 0:
        season_losses += 1
    return season_losses


def update_elo(starting_elo, team_win_perc, team_win, mov):
    """ Update the elo for all the teams that played that week based on the results of their games """
    if mov > MOL_BAR and team_win == 0:
        updated_elo = starting_elo + (K * (team_win - team_win_perc) * MOL_FAC)
    else:
        updated_elo = starting_elo + (K * (team_win - team_win_perc))

    return updated_elo


def one_week_sim(season_results, last_elo, week_sch):
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

    # get the week for the schedule
    week = week_sch["Week"].iloc[0]

    # store the conference championship games
    week_sch[f'Conf_Chip_{week}'] = week_sch['Notes'].apply(lambda x: 1 if str(x).split("(")[0].strip() in CONF_CHIP_LIST else np.NAN)

    # create a week results dataframe to add to last_elo
    away_results = week_sch[['Away', 'Away_Win', 'Away_New_Elo', f'Conf_Chip_{week}']]
    away_results.columns = ['Team', f'Week_{week}_Win', f'Week_{week}_Elo', f'Conf_Chip_{week}']
    home_results = week_sch[['Home', 'Home_Win', 'Home_New_Elo', f'Conf_Chip_{week}']]
    home_results.columns = ['Team', f'Week_{week}_Win', f'Week_{week}_Elo', f'Conf_Chip_{week}']
    week_results = pd.concat([away_results, home_results])

    # save the results to the season results dataframe
    season_results = pd.merge(season_results, week_results, on='Team', how='left')

    # update the season win totals
    season_results['Season_Wins'] = season_results.apply(lambda x: add_wins(x['Season_Wins'], x[f'Week_{week}_Win']),
                                                         axis=1)
    season_results['Season_Losses'] = season_results.apply(lambda x: add_losses(x['Season_Losses'],
                                                                                x[f'Week_{week}_Win']), axis=1)

    # update last_elo with the new elo for teams
    last_elo = pd.merge(last_elo, week_results[['Team', f'Week_{week}_Elo']], on='Team', how='left')

    # fill in elo for teams that didn't play that week in last_elo and season results
    if week == 1:
        prev_week_elo = 'Starting_Elo'
    else:
        prev_week_elo = f'Week_{week - 1}_Elo'

    last_elo[f'Week_{week}_Elo'] = last_elo.apply(
        lambda x: x[f'Week_{week}_Elo'] if ~np.isnan(x[f'Week_{week}_Elo']) else x[prev_week_elo], axis=1)
    season_results[f'Week_{week}_Elo'] = season_results.apply(
        lambda x: x[f'Week_{week}_Elo'] if ~np.isnan(x[f'Week_{week}_Elo']) else x[prev_week_elo], axis=1)

    return season_results, last_elo


def eos_adjustments(eos_elo, season_losses, p5, conf_champ):
    """
    Adjust the end of season elo's for the teams and create the final rankings
    """
    # give adjustments for having zero, one, two, or three losses
    if season_losses == 0:
        eos_elo += ZERO_L
    elif season_losses == 1:
        eos_elo += ONE_L
    elif season_losses == 2:
        eos_elo += TWO_L
    elif season_losses >= 3:
        eos_elo += THREE_L

    # penalize for teams not being Power 5
    if p5 == 0:
        eos_elo += NON_P5

    # adjust for if the team played in a conference title game
    if conf_champ == 1:
        eos_elo += TITLE_GAME

    return eos_elo


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
    # create baseline dataframe for results
    season_results = pd.DataFrame().assign(Team=conf_df['School'],
                                           Conference=conf_df['Acronym'],
                                           P5=conf_df['P5'],
                                           Season_Wins=0,
                                           Season_Losses=0)

    # pull the most recent elo data
    last_elo_yr = list(elo_df.columns)[-1]
    last_elo = elo_df[['Team', last_elo_yr]]
    last_elo.columns = ['Team', 'Starting_Elo']

    # add starting elo to the season results df
    season_results['Starting_Elo'] = None
    for e_idx, e_row in last_elo.iterrows():
        for idx, row in season_results.iterrows():
            if e_row['Team'] == row['Team']:
                season_results.loc[idx, 'Starting_Elo'] = e_row['Starting_Elo']

    # create a list of the difference weeks and initialize the conference championship column
    weeks = sch_df['Week'].unique()
    season_results['Conf_Chip'] = None

    # loop through each of the weeks in the schedule
    for week in weeks:

        # get the schedule of games for that week
        week_sch = sch_df[sch_df['Week'] == week]

        # run the simulation for one week of the regular season
        season_results, last_elo = one_week_sim(season_results, last_elo, week_sch)

        # make a column for if the team played in a conference championship
        season_results['Conf_Chip'] = season_results.apply(
            lambda x: 1 if x[f'Conf_Chip_{week}'] == 1 else x['Conf_Chip'], axis=1)

    # checks if it's the current season to not double count end of season adjustments
    if CURRENT_SEASON:
        # creates a final elo column
        last_elo['Final_Elo'] = last_elo[list(last_elo.columns)[-1]].apply(lambda x: round(x, 2))
    else:
        # make the end of season adjustments
        season_results['EOS_Elo'] = season_results.apply(lambda x: eos_adjustments(x[f'Week_{weeks[-1]}_Elo'],
                                                                                   x['Season_Losses'],
                                                                                   x['P5'],
                                                                                   x['Conf_Chip']), axis=1)
        last_elo = pd.merge(last_elo, season_results[['Team', f'EOS_Elo']], on='Team', how='left')
        last_elo['EOS_Elo'] = last_elo.apply(
            lambda x: x['EOS_Elo'] if ~np.isnan(x['EOS_Elo']) else x[f'Week_{weeks[-1]}_Elo'], axis=1)

        # adjust the final elo to shift back towards 750 or 1500 depending on the school
        fbs_teams = list(conf_df['School'])
        last_elo['FBS'] = last_elo['Team'].apply(lambda x: True if x in fbs_teams else False)
        last_elo['Final_Elo'] = last_elo.apply(lambda x: round(eos_elo(x['EOS_Elo'], x['FBS']), 2), axis=1)

    team_record = season_results[['Team', 'Season_Wins', 'Season_Losses']]

    return last_elo, team_record


def main():

    # load in the data
    elo_df = pd.read_excel(ELO)
    sch_df = pd.read_excel(SCHEDULE)
    conf_df = pd.read_excel(CONFERENCES)

    # remove current year if in the elo dataframe
    if str(YEAR) in list(elo_df.columns):
        elo_df.drop([str(YEAR)], axis=1, inplace=True)

    # run the season calculations to get the end of season data
    last_elo, team_record = season_sim(elo_df, sch_df, conf_df)

    # get the year from the input data and add the year to the elo data
    elo_df[str(YEAR)] = last_elo['Final_Elo']

    # save the elo data into the same excel path with the new column added on
    elo_df.to_excel(ELO, index=False)

    # save the team records to a file if this is a current year
    if CURRENT_SEASON:
        team_record.to_csv(f'../Team Records {YEAR.split(" ")[0]}.csv', index=False)


main()
