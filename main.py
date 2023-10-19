"""
CFB Simulation Model
"""
from warnings import simplefilter
import pandas as pd
import numpy as np
import progressbar
from datetime import datetime
import os

simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', None)
pd.set_option('mode.chained_assignment', None)

# input data files
CONFERENCES = "./Input Data/Conferences.xlsx"
ELO = "./Input Data/Elo By Year.xlsx"
SCHEDULE = "./Input Data/Mock Schedule.xlsx"
FAV_MOV = "./Input Data/MOV Favorite Win.xlsx"
UPSET_MOV = "./Input Data/MOV Favorite Upset.xlsx"

# set up the global variables for num simulations, qualifiers, and playoff teams based on baseline or git input
try:
    N = int(os.environ['INPUT_N'])
    AQ = int(os.environ['INPUT_AQ'])
    PLAYOFF = int(os.environ['INPUT_PLAYOFF'])
except KeyError:
    N = 100  # number of simulations to run
    AQ = 6  # number of automatic qualifiers
    PLAYOFF = 12  # number of playoff teams


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


def calc_win_prob(away_elo, home_elo, neutral):
    """ Calculate the win probability of team 1 given the elo of two teams """
    # checks if the game is at a neutral site
    if neutral == 1:
        away_win_prob = 1 / (1 + (10 ** ((home_elo - away_elo) / 300)))
    # applies home field advantage to home team if not at a neutral site
    else:
        away_win_prob = 1 / (1 + (10 ** ((home_elo - away_elo + HOME) / 300)))

    return round(away_win_prob, 3)


def check_fav_win(away_win_perc, away_win):
    """ Function to determine whether the favorite of a matchup won the game """
    return (away_win_perc >= 0.5 and away_win == 1) or (away_win_perc < 0.5 and away_win == 0)


def get_mov(fav_win, diff_win_perc, fav_mov, upset_mov):
    """ Select a random margin of victory based on the corresponding game info """
    # adjust win percentage from decimal to whole number and get the values to access the mov tables
    diff_win_perc = diff_win_perc * 100
    min_diff = int(diff_win_perc - (diff_win_perc % 5))
    max_diff = min_diff + 5

    # catch error for if team has 100% chance to win
    if diff_win_perc == 100:
        min_diff = 95
        max_diff = 100

    # use whether the favorite won to select a single random row from the corresponding margin of victory dataset
    if fav_win:
        table = f'Favorite Wins {min_diff}-{max_diff}'
        mov = fav_mov[table].sample()['Margin of Victory'].iloc[0]
    else:
        table = f'Favorite Loses {min_diff}-{max_diff}'
        mov = upset_mov[table].sample()['Margin of Victory'].iloc[0]

    return mov


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


def one_week_sim(season_results, last_elo, week_sch, fav_mov_df, upset_mov_df):
    """
    Simulate one week of the college football season
    """
    # add the elo for both the home and away team to be able to make the calculation
    team_ind_elo = last_elo.set_index('Team')
    week_sch['Away_Elo'] = week_sch['Away'].apply(lambda x: team_ind_elo.loc[x][last_elo.columns[-1]])
    week_sch['Home_Elo'] = week_sch['Home'].apply(lambda x: team_ind_elo.loc[x][last_elo.columns[-1]])

    # fill teams with no elo with the base 750
    week_sch['Away_Elo'].fillna(750, inplace=True)
    week_sch['Home_Elo'].fillna(750, inplace=True)

    # create a new column with the win percentage using the function
    week_sch['Away_Win%'] = week_sch.apply(lambda x: calc_win_prob(x['Away_Elo'], x['Home_Elo'], x['Neutral']), axis=1)
    week_sch['Home_Win%'] = week_sch['Away_Win%'].apply(lambda x: 1 - x)
    week_sch['Diff_Win%'] = abs(week_sch['Away_Win%'] - week_sch['Home_Win%'])

    # create a random number from 0-1 in a column and determine if away team won and whether the favorite won
    week_sch['Win_Val'] = np.random.rand(week_sch.shape[0])
    week_sch['Away_Win'] = week_sch.apply(lambda x: 1 if x['Win_Val'] < x['Away_Win%'] else 0, axis=1)
    week_sch['Home_Win'] = week_sch['Away_Win'].apply(lambda x: 1 if x == 0 else 0)
    week_sch['Fav_Win'] = week_sch.apply(lambda x: 1 if check_fav_win(x['Away_Win%'], x['Away_Win']) else 0, axis=1)
    week_sch['MOV'] = week_sch.apply(lambda x: get_mov(x['Fav_Win'], x['Diff_Win%'], fav_mov_df, upset_mov_df), axis=1)

    # update the elo for the teams after the week has ended
    week_sch['Away_New_Elo'] = week_sch.apply(lambda x: update_elo(x['Away_Elo'], x['Away_Win%'], x['Away_Win'],
                                                                   x['MOV']), axis=1)
    week_sch['Home_New_Elo'] = week_sch.apply(lambda x: update_elo(x['Home_Elo'], x['Home_Win%'], x['Home_Win'],
                                                                   x['MOV']), axis=1)

    # create a week results dataframe to add to season results
    week = week_sch["Week"].iloc[0]
    away_results = week_sch[['Away', 'Away_Win', 'MOV', 'Away_New_Elo']]
    away_results.columns = ['Team', f'Week_{week}_Win', f'Week_{week}_MOV', f'Week_{week}_Elo']
    home_results = week_sch[['Home', 'Home_Win', 'MOV', 'Home_New_Elo']]
    home_results.columns = ['Team', f'Week_{week}_Win', f'Week_{week}_MOV', f'Week_{week}_Elo']
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

    # fill in elo for teams that didn't play that week in both last_elo and season_results
    if week == 0:
        prev_week_elo = 'Starting_Elo'
    elif week == 'Conf_Champ':
        prev_week_elo = last_elo.columns[-2]
    else:
        prev_week_elo = f'Week_{week - 1}_Elo'

    last_elo[f'Week_{week}_Elo'] = last_elo.apply(
        lambda x: x[f'Week_{week}_Elo'] if ~np.isnan(x[f'Week_{week}_Elo']) else x[prev_week_elo], axis=1)
    season_results[f'Week_{week}_Elo'] = season_results.apply(
        lambda x: x[f'Week_{week}_Elo'] if ~np.isnan(x[f'Week_{week}_Elo']) else x[prev_week_elo], axis=1)

    return season_results, last_elo


def reg_season_sim(season_results, last_elo, sch_df, fav_mov_df, upset_mov_df):
    """
    Simulate the regular season games for the college football season
    """
    # create a list of the difference weeks
    weeks = sch_df['Week'].unique()

    # loop through each of the weeks in the schedule
    for week in weeks:

        # get the schedule of games for that week
        week_sch = sch_df[sch_df['Week'] == week]

        # run the simulation for one week of the regular season
        season_results, last_elo = one_week_sim(season_results, last_elo, week_sch, fav_mov_df, upset_mov_df)

    # create winning percentage column for season results
    season_results['Win_Perc'] = round(season_results['Season_Wins'] / (season_results['Season_Wins'] + season_results['Season_Losses']), 3)

    return season_results, last_elo


def create_conf_champ(conf_results, last_elo, conf):
    """
    Get the top two teams in a conference and create a schedule line to save it
    """
    # determine if there are divisions
    if conf_results.iloc[0]['Has_Div'] == 1:

        # get the winner of the first  and second divisions respectively
        top_div1 = conf_results[conf_results['Div1'] == 1].nlargest(1, ['Win_Perc', last_elo.columns[-1]])
        top_div2 = conf_results[conf_results['Div2'] == 1].nlargest(1, ['Win_Perc', last_elo.columns[-1]])

        # get the teams for the matchup
        home = top_div1.iloc[0]['Team']
        away = top_div2.iloc[0]['Team']

    else:
        # select the top two teams by win percentage using elo as a tiebreaker
        top_two = conf_results.nlargest(2, ['Win_Perc', last_elo.columns[-1]])

        # get the teams for the matchup
        home = top_two.iloc[0]['Team']
        away = top_two.iloc[1]['Team']

    # create dataframe containing the matchup
    matchup = pd.DataFrame().assign(Week=['Conf_Champ'],
                                    Conference=[conf],
                                    Away=[away],
                                    Home=[home],
                                    Neutral=[1])

    return matchup


def conf_champs_sim(season_results, last_elo, fav_mov_df, upset_mov_df):
    """
    Determine which teams make the conference championships and simulate the games for them
    """
    # collect all the different conferences into a list and then remove the independent variable
    conferences = list(season_results['Conference'].unique())
    conferences.remove('Ind')

    # initialize the empty conference championship schedule
    conf_champ_sch = pd.DataFrame()

    # create the matchups for each conference
    for conf in conferences:

        # select the results for the teams from that conference and get the matchup for that conference
        conf_results = season_results[season_results['Conference'] == conf]
        matchup = create_conf_champ(conf_results, last_elo, conf)

        # append the matchup to the conference championship schedule
        conf_champ_sch = pd.concat([conf_champ_sch, matchup], ignore_index=True)

    # run the week simulation for the conference championship
    season_results, last_elo = one_week_sim(season_results, last_elo, conf_champ_sch, fav_mov_df, upset_mov_df)

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
    elif season_losses == 3:
        eos_elo += THREE_L

    # penalize for teams not being Power 5
    if p5 == 0:
        eos_elo += NON_P5

    # adjust for if the team played in a conference title game
    if ~np.isnan(conf_champ):
        eos_elo += TITLE_GAME

    return eos_elo


def one_season_sim(season_results, last_elo, sch_df, fav_mov_df, upset_mov_df):
    """
    Run one season of the simulation
    """
    # run the regular season and conference finals
    season_results, last_elo = reg_season_sim(season_results, last_elo, sch_df, fav_mov_df, upset_mov_df)
    season_results, last_elo = conf_champs_sim(season_results, last_elo, fav_mov_df, upset_mov_df)

    # make the end of season adjustments
    season_results['Final_Elo'] = season_results.apply(lambda x: eos_adjustments(x['Week_Conf_Champ_Elo'],
                                                                                 x['Season_Losses'],
                                                                                 x['P5'],
                                                                                 x['Week_Conf_Champ_Win']), axis=1)
    season_results.sort_values(by=['Final_Elo'], ascending=False, ignore_index=True, inplace=True)

    return season_results


def get_top_teams(season_results):
    """
    Get the top 12 teams that would make the playoffs for a given season
    """
    # collect all the different conferences into a list and then remove the independent variable
    conferences = list(season_results['Conference'].unique())
    conferences.remove('Ind')
    conf_winners = pd.DataFrame()

    # get the top teams in each conference
    for conf in conferences:

        # get the top team from each conference and append top team to the conference winners df
        top_team = season_results[season_results['Conference'] == conf].iloc[0]
        conf_winners = pd.concat([conf_winners, top_team], ignore_index=False, axis=1)

    # transpose the data into the correct format
    conf_winners = conf_winners.transpose()

    # get teams outside the top number that are auto qualified
    outside_aq = conf_winners.iloc[:AQ].query(f'index > {PLAYOFF - 1}')

    # get the top teams plus the automatic qualifiers outside the playoffs
    top_teams = pd.concat([season_results.iloc[:PLAYOFF - len(outside_aq)], outside_aq])

    return top_teams, outside_aq


def add_stats(team_playoff_stats, conf_playoff_stats, teams, aq_teams, confs, aq_confs):
    """
    Add the stats to the dictionaries to aggregate
    """
    # add the playoffs and automatic qualifier stats to the teams dictionary
    for team in teams:
        if team not in team_playoff_stats['Playoffs']:
            team_playoff_stats['Playoffs'][team] = 1
        else:
            team_playoff_stats['Playoffs'][team] += 1
    for team in aq_teams:
        if team not in team_playoff_stats['AQ']:
            team_playoff_stats['AQ'][team] = 1
        else:
            team_playoff_stats['AQ'][team] += 1

    # add the playoffs and automatic qualifier stats to the conferences dictionary
    for conf in confs:
        if conf not in conf_playoff_stats['Playoffs']:
            conf_playoff_stats['Playoffs'][conf] = 1
        else:
            conf_playoff_stats['Playoffs'][conf] += 1
    for conf in aq_confs:
        if conf not in conf_playoff_stats['AQ']:
            conf_playoff_stats['AQ'][conf] = 1
        else:
            conf_playoff_stats['AQ'][conf] += 1

    return team_playoff_stats, conf_playoff_stats


def clean_results(team_playoff_stats, conf_playoff_stats, conf_df):
    """
    Clean up the dictionaries into dataframes and take the averages for the stats
    """
    # turn the dictionaries into dataframes
    team_playoff_stats = pd.DataFrame(team_playoff_stats)
    conf_playoff_stats = pd.DataFrame(conf_playoff_stats)

    # fill the NA's with zeroes
    team_playoff_stats.fillna(0, inplace=True)
    conf_playoff_stats.fillna(0, inplace=True)

    # create new calculated fields for percentage of the time the team is an auto qualifier
    team_playoff_stats['%AQ'] = team_playoff_stats['AQ'] / team_playoff_stats['Playoffs']
    conf_playoff_stats['%AQ'] = conf_playoff_stats['AQ'] / conf_playoff_stats['Playoffs']

    # take the percentages for teams for playoff appearances and average playoff appearances for conferences
    team_playoff_stats['Playoffs'] = team_playoff_stats['Playoffs'].apply(lambda x: round((x / N) * 100, 4))
    team_playoff_stats['AQ'] = team_playoff_stats['AQ'].apply(lambda x: round((x / N) * 100, 4))
    team_playoff_stats['%AQ'] = team_playoff_stats['%AQ'].apply(lambda x: round(x * 100, 4))
    conf_playoff_stats['Playoffs'] = conf_playoff_stats['Playoffs'].apply(lambda x: round(x / N, 2))
    conf_playoff_stats['AQ'] = conf_playoff_stats['AQ'].apply(lambda x: round(x / N, 2))
    conf_playoff_stats['%AQ'] = conf_playoff_stats['%AQ'].apply(lambda x: round(x * 100, 2))

    # turn the team index into a column, sort by values, and rename columns
    team_playoff_stats.reset_index(inplace=True)
    conf_playoff_stats.reset_index(inplace=True)
    team_playoff_stats.sort_values(by=['Playoffs'], ascending=False, ignore_index=True, inplace=True)
    conf_playoff_stats.sort_values(by=['Playoffs'], ascending=False, ignore_index=True, inplace=True)
    team_playoff_stats.columns = ['Team', '% Time in Playoffs', '% Make Playoffs Due to AQ',
                                  '% of Appearances Due to AQ']
    conf_playoff_stats.columns = ['Conference', 'Avg Num Playoff Teams', 'Avg Num Playoffs Due to AQ',
                                  '% of Teams Due to AQ']

    # add the conferences for the individual teams
    team_ind_conf = conf_df.set_index('School')
    team_playoff_stats['Conference'] = team_playoff_stats['Team'].apply(lambda x: team_ind_conf.loc[x]['Acronym'])
    team_playoff_stats = team_playoff_stats[['Team', 'Conference', '% Time in Playoffs', '% Make Playoffs Due to AQ',
                                             '% of Appearances Due to AQ']]

    return team_playoff_stats, conf_playoff_stats


def run_sim(conf_df, elo_df, sch_df, fav_mov_df, upset_mov_df):
    """
    Run the full simulation
    """
    # create baseline dataframe for results
    season_results = pd.DataFrame().assign(Team=conf_df['School'],
                                           Conference=conf_df['Acronym'],
                                           Has_Div=conf_df['Has_Div'],
                                           Div1=conf_df['Div1'],
                                           Div2=conf_df['Div2'],
                                           P5=conf_df['P5'],
                                           Season_Wins=0,
                                           Season_Losses=0)

    # pull the most recent elo data
    last_elo_yr = list(elo_df.columns)[-1]
    last_elo = elo_df[['Team', last_elo_yr]]

    # add starting elo to the season results df
    season_results['Starting_Elo'] = None
    for e_idx, e_row in last_elo.iterrows():
        for idx, row in season_results.iterrows():
            if e_row['Team'] == row['Team']:
                season_results.loc[idx, 'Starting_Elo'] = e_row[last_elo_yr]

    # rename the columns of the elo table
    last_elo.columns = ['Team', 'Starting_Elo']

    # initialize the dictionaries to track total playoffs for teams and conferences in the simulations
    team_playoff_stats = {'Playoffs': {}, 'AQ': {}}
    conf_playoff_stats = {'Playoffs': {}, 'AQ': {}}

    # create the widgets for the progress bar
    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.GranularBar(), ' ',
        progressbar.Percentage(), ' ',
        progressbar.ETA(),
    ]

    # run the number of simulations with a progress bar
    with progressbar.ProgressBar(max_value=N, widgets=widgets) as bar:
        for i in range(N):

            # reset the season results in each run
            temp_season_results = season_results

            # simulate one season of games
            temp_season_results = one_season_sim(temp_season_results, last_elo, sch_df, fav_mov_df, upset_mov_df)

            # get the top teams from that simulation
            top_teams, outside_aq = get_top_teams(temp_season_results)

            # get the teams and conferences for playoffs and AQs
            teams = top_teams['Team'].tolist()
            aq_teams = outside_aq['Team'].tolist()
            confs = top_teams['Conference'].tolist()
            aq_confs = outside_aq['Conference'].to_list()

            # add the stats to the overall tracking
            team_playoff_stats, conf_playoff_stats = add_stats(team_playoff_stats,
                                                               conf_playoff_stats,
                                                               teams,
                                                               aq_teams,
                                                               confs,
                                                               aq_confs)

            # update the bar
            if i % 5 == 0:
                bar.update(i)

    # clean the dictionaries into better displayed dataframes
    team_playoff_stats, conf_playoff_stats = clean_results(team_playoff_stats, conf_playoff_stats, conf_df)

    return team_playoff_stats, conf_playoff_stats


def main():

    # load the data into dataframes
    conf_df = pd.read_excel(CONFERENCES)
    elo_df = pd.read_excel(ELO)
    sch_df = pd.read_excel(SCHEDULE)
    fav_mov_df = pd.read_excel(FAV_MOV, sheet_name=None)
    upset_mov_df = pd.read_excel(UPSET_MOV, sheet_name=None)

    # clean the datasets
    sch_df.drop(['Game', 'Year'], axis=1, inplace=True)

    # run the full simulation
    team_playoff_stats, conf_playoff_stats = run_sim(conf_df, elo_df, sch_df, fav_mov_df, upset_mov_df)

    # create unique csv name to save file
    file_name_add = f'_{datetime.now().strftime("%b%y")}_AQ{AQ}_P{PLAYOFF}_N{N}'

    # save the dataframes to csv
    team_playoff_stats.to_csv(f"./Simulation Outputs/Team Stats/team_stats{file_name_add}.csv")
    conf_playoff_stats.to_csv(f"./Simulation Outputs/Conference Stats/conference_stats{file_name_add}.csv")

    # display the tables within python
    print(team_playoff_stats, "\n")
    print(conf_playoff_stats)


if __name__ == '__main__':

    main()
