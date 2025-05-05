import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# 1) Create a Session and set your headers once
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
})



def extract_team_info(team_json):
    """
    Given a JSON dictionary from the team API response,
    extracts and returns a dictionary with the team id,
    abbreviation, and display name.
    """
    team_id = team_json.get("id")
    abbreviation = team_json.get("abbreviation")
    display_name = team_json.get("displayName")
    return {"id": team_id, "abbreviation": abbreviation, "displayName": display_name}

def get_all_nfl_teams():
    """
    Get a list of all NFL teams and their IDs from the ESPN API.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing team IDs, abbreviations, and display names
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = session.get('https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams?limit=32', headers=headers)
    team_info_list = []
    
    if response.status_code == 200:
        team_links_dict = response.json() 
        team_links = team_links_dict.get("items", [])
        
        for item in team_links:
            ref_url = item.get("$ref")
            if ref_url:
                # Fetch the detailed team data from the ref URL
                team_response = session.get(ref_url, headers=headers)
                if team_response.status_code == 200:
                    team_json = team_response.json()
                    team_info = extract_team_info(team_json)
                    team_info_list.append(team_info)
    
    return pd.DataFrame(team_info_list)

def extract_provider_info(provider_json):
    """
    Given a JSON dictionary from the provider API response,
    extracts and returns a dictionary with the provider id and name.
    """
    provider_id = provider_json.get("id")
    provider_name = provider_json.get("name")
    return {"id": provider_id, "name": provider_name}

def get_betting_providers():
    """
    Get a list of all betting providers from the ESPN API.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing provider IDs and names
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    bet_prov_endpoint = 'https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/providers?limit=100'
    response = session.get(bet_prov_endpoint, headers=headers)
    provider_info_list = []
    
    if response.status_code == 200:
        provider_data = response.json()
        provider_items = provider_data.get("items", [])
        
        for item in provider_items:
            ref_url = item.get("$ref")
            if ref_url:
                prov_response = session.get(ref_url, headers=headers)
                if prov_response.status_code == 200:
                    provider_json = prov_response.json()
                    provider_info = extract_provider_info(provider_json)
                    provider_info_list.append(provider_info)
            else:
                # If no $ref is present, use the item directly
                provider_info = extract_provider_info(item)
                provider_info_list.append(provider_info)
    
    return pd.DataFrame(provider_info_list)

def get_next_page_url(resp_json):
    # 1) first look for the simple “next” object
    nxt = resp_json.get("next")
    if isinstance(nxt, dict) and "$ref" in nxt:
        return nxt["$ref"]

    # 2) fall back to any links array
    for link in resp_json.get("links", []):
        if link.get("rel") == "next" and ("$ref" in link or "href" in link):
            return link.get("$ref") or link.get("href")

    return None


def get_team_betting_data(team_id, limit=99999, provider_id=1002):
    """
    Get historical betting data for a specific team.
    
    Parameters:
    -----------
    team_id : str
        The ESPN API team ID
    limit : int, optional (default=99999)
        The number of past games to retrieve
    provider_id : int, optional (default=1002)
        The betting provider ID
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the team's betting and game data
    """
    # URL for the odds endpoint
    #event_odds_api = f'http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/odds/{provider_id}/past-performances?limit=100'
    event_odds_api = (
     f'https://sports.core.api.espn.com/v2/sports/football/'
     f'leagues/nfl/teams/{team_id}/odds/{provider_id}/past-performances')
    # Add debug output
    print(f"  - Fetching betting data for team {team_id}")
    
    # Add headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Sometimes ESPN API defaults to a lower limit, so let's verify what we get
    #response = session.get(event_odds_api, headers=headers)
    response = session.get(event_odds_api, params={"limit": min(limit, 190)})

    odds_info = []  # This list will store one row per odds item
    
    if response.status_code == 200:
        odds_data = response.json()
        game_info_items = odds_data.get('items', [])
        
        # Debug check for count and potential pagination
        print(f"  - Initial fetch returned {len(game_info_items)} games")
        
        # Process the first page of results
        for item in game_info_items:
            process_item(item, team_id, odds_info)
        
        # Get total count from API response (if available)
        total = odds_data.get('count', 0)
        print(f"  - API indicates {total} total games available")
        
        # Check for pagination using the new helper function
        next_page_url = get_next_page_url(odds_data)
        page_counter = 1
        
        # Continue fetching pages until we have all games or reach the limit
        while next_page_url and len(odds_info) < limit:
            page_counter += 1
            print(f"  - Fetching page {page_counter}, currently have {len(odds_info)} games")
            
            try:
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
                next_resp = session.get(next_page_url, headers=headers)
                if next_resp.status_code == 200:
                    next_data = next_resp.json()
                    new_items = next_data.get('items', [])
                    
                    # Process each item on this page
                    for item in new_items:
                        process_item(item, team_id, odds_info)
                        
                    print(f"  - Added {len(new_items)} more games, total now: {len(odds_info)}")
                    
                    # Get the next page URL
                    next_page_url = get_next_page_url(next_data)
                else:
                    print(f"  - Failed to fetch next page: {next_resp.status_code}")
                    break
            except Exception as e:
                print(f"  - Error fetching next page: {str(e)}")
                break
        
        print(f"  - Finished processing {len(odds_info)} games for team {team_id}")
        
        # Convert to DataFrame and reorder columns if we have data
        if odds_info:
            team_odds_df = pd.DataFrame(odds_info)
            
            # Reorder columns for better readability
            cols_order = [
                'game_id', 'game_date', 'final_lineDate', 'api_team_id', 'is_home', 'is_away', 
                'spread', 'overOdds', 'underOdds', 'totalLine', 'totalResult',
                'moneyLineOdds', 'moneylineWinner', 'spreadOdds', 'spreadWinner',
                'api_team_score', 'opponent_score', 'api_team_won',
                'home_team_id', 'home_winner', 'home_score', 'home_curr_rank', 
                'home_prev_rank', 'home_rank_summary', 'away_team_id', 'away_winner',
                'away_score', 'away_curr_rank', 'away_prev_rank', 'away_rank_summary'
            ]
            
            # Only include columns that exist in the DataFrame
            existing_cols = [col for col in cols_order if col in team_odds_df.columns]
            team_odds_df = team_odds_df[existing_cols]
            
            return team_odds_df
        else:
            print(f"  - No odds data found for team {team_id}")
            return pd.DataFrame()  # Return empty DataFrame if no odds data
    else:
        print(f"  - Error fetching odds data for team {team_id}: {response.status_code}")
        return pd.DataFrame()  # Return empty DataFrame on error

def process_item(item, team_id, odds_info):
    """
    Process a single game item from the API response and add it to odds_info list.
    
    Parameters:
    -----------
    item : dict
        The game item from the API response
    team_id : str
        The ESPN API team ID
    odds_info : list
        The list to append the processed item to
    """
    # Extract base odds data
    spread = item.get('spread')
    overOdds = item.get('overOdds')
    underOdds = item.get('underOdds')
    lineDate = item.get('lineDate')
    totalLine = item.get('totalLine')
    totalResult = item.get('totalResult')
    moneyLineOdds = item.get('moneyLineOdds')
    moneylineWinner = item.get('moneylineWinner')
    spreadOdds = item.get('spreadOdds')
    spreadWinner = item.get('spreadWinner')
    
    # Initialize variables for game and competitor details
    game_id = None
    game_date = None
    home_team_id = None
    home_winner = None
    home_score = None
    home_curr_rank = None
    home_prev_rank = None
    home_rank_summary = None
    away_team_id = None
    away_winner = None
    away_score = None
    away_curr_rank = None
    away_prev_rank = None
    away_rank_summary = None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Get the game details via the pastCompetition URL
    game_url = item.get('pastCompetition', {}).get('$ref')
    if game_url.startswith("http://"):
        game_url = game_url.replace("http://", "https://", 1)
    if game_url:
        try:
            game_response = session.get(game_url, headers=headers)
            if game_response.status_code == 200:
                game_data = game_response.json()
                game_id = game_data.get('id')
                game_date = game_data.get('date')
                
                # Extract competitors info
                competitors = game_data.get('competitors', [])
                for t in competitors:
                    team_id_comp = t.get('id')
                    homeAway = t.get('homeAway')
                    winner = t.get('winner')
                    
                    # Get the score by following the score ref URL (if present)
                    score_value = None
                    score_ref_url = t.get('score', {}).get('$ref')
                    if score_ref_url:
                        try:
                            score_response = session.get(score_ref_url, headers=headers)
                            if score_response.status_code == 200:
                                score_data = score_response.json()
                                score_value = score_data.get('value')
                        except requests.exceptions.RequestException:
                            pass
                    
                    # Get ranking information (if available)
                    curr_rank = None
                    prev_rank = None
                    rank_summary = None
                    if t.get('ranks') is not None:
                        rank_ref_url1 = t.get('ranks', {}).get('$ref')
                        if rank_ref_url1:
                            try:
                                rank_list_response = session.get(rank_ref_url1, headers=headers)
                                if rank_list_response.status_code == 200:
                                    rank_list_json = rank_list_response.json()
                                    items_list = rank_list_json.get('items', [])
                                    if items_list:
                                        first_rank_ref = items_list[0].get('$ref')
                                        if first_rank_ref:
                                            rank_detail_response = session.get(first_rank_ref, headers=headers)
                                            if rank_detail_response.status_code == 200:
                                                rank_json = rank_detail_response.json()
                                                rank_info = rank_json.get('rank', {})
                                                curr_rank = rank_info.get('current')
                                                prev_rank = rank_info.get('previous')
                                                rank_summary = rank_info.get('summary')
                            except requests.exceptions.RequestException:
                                pass
                    
                    # Separate home and away competitor details
                    if homeAway == 'home':
                        home_team_id = team_id_comp
                        home_winner = winner
                        home_score = score_value
                        home_curr_rank = curr_rank
                        home_prev_rank = prev_rank
                        home_rank_summary = rank_summary
                    elif homeAway == 'away':
                        away_team_id = team_id_comp
                        away_winner = winner
                        away_score = score_value
                        away_curr_rank = curr_rank
                        away_prev_rank = prev_rank
                        away_rank_summary = rank_summary
        except requests.exceptions.RequestException as e:
            print(f"  - Error fetching game data: {str(e)}")

    # Combine all information into one row (dictionary)
    row = {
        'api_team_id': team_id,
        'game_id': game_id,
        'game_date': game_date,
        'spread': spread,
        'overOdds': overOdds,
        'underOdds': underOdds,
        'final_lineDate': lineDate,
        'totalLine': totalLine,
        'totalResult': totalResult,
        'moneyLineOdds': moneyLineOdds,
        'moneylineWinner': moneylineWinner,
        'spreadOdds': spreadOdds,
        'spreadWinner': spreadWinner,
        # Home competitor details
        'home_team_id': home_team_id,
        'home_winner': home_winner,
        'home_score': home_score,
        'home_curr_rank': home_curr_rank,
        'home_prev_rank': home_prev_rank,
        'home_rank_summary': home_rank_summary,
        # Away competitor details
        'away_team_id': away_team_id,
        'away_winner': away_winner,
        'away_score': away_score,
        'away_curr_rank': away_curr_rank,
        'away_prev_rank': away_prev_rank,
        'away_rank_summary': away_rank_summary
    }

    # Add derived fields
    row['is_home'] = 1 if home_team_id == team_id else 0
    row['is_away'] = 1 if away_team_id == team_id else 0
    row['api_team_score'] = home_score if home_team_id == team_id else away_score
    row['opponent_score'] = away_score if home_team_id == team_id else home_score
    row['api_team_won'] = home_winner if home_team_id == team_id else away_winner
    
    odds_info.append(row)

def fetch_line_movement(game_id: str, limit: int = 99999) -> pd.DataFrame:
    """
    Fetch line movement data for a specific game.
    
    Parameters:
    -----------
    game_id : str
        The ESPN API game ID
    limit : int, optional (default=100)
        The number of movement records to retrieve
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing line movement data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    url = (
        f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
        f"events/{game_id}/competitions/{game_id}/"
        f"odds/1002/history/0/movement?limit={limit}"
    )
    
    try:
        resp = session.get(url, headers=headers)
        resp.raise_for_status()
        
        data = resp.json().get("items", [])
        movement_data = []
        
        # Process initial page of results
        movement_data.extend(data)
        
        # Check for pagination
        next_page_url = get_next_page_url(resp.json())
        
        # Continue fetching pages until we have all movements or reach the limit
        while next_page_url and len(movement_data) < limit:
            try:
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
                next_resp = session.get(next_page_url, headers=headers)
                next_resp.raise_for_status()
                
                next_data = next_resp.json()
                new_items = next_data.get('items', [])
                movement_data.extend(new_items)
                
                # Get the next page URL
                next_page_url = get_next_page_url(next_data)
            except Exception as e:
                print(f"Error fetching next page of movements for game {game_id}: {str(e)}")
                break
        
        if movement_data:
            df = pd.DataFrame(movement_data)
            df["lineDate"] = pd.to_datetime(df["lineDate"])
            df = df.sort_values("lineDate").reset_index(drop=True)
            # Keep only the three columns
            return df[["lineDate", "awayOdds", "homeOdds"]]
        else:
            return pd.DataFrame(columns=["lineDate", "awayOdds", "homeOdds"])
            
    except Exception as e:
        print(f"Error fetching line movement for game {game_id}: {str(e)}")
        return pd.DataFrame(columns=["lineDate", "awayOdds", "homeOdds"])

def merge_line_movements(baseline_df, movement_limit=9999):
    """
    Given a baseline DataFrame with one row per game (must contain 'game_id'),
    fetch the line-movement series for each game_id via fetch_line_movement(),
    concatenate them into one DataFrame, and then merge back.
    
    Parameters:
    -----------
    baseline_df : pd.DataFrame
        DataFrame returned by get_team_betting_data(), with a 'game_id' column.
    movement_limit : int, optional (default=100)
        How many movement records to pull per game.
        
    Returns:
    --------
    full_df : pd.DataFrame
        A DataFrame in "long" form: one row per (game_id, timestamp) with all
        baseline fields plus `lineDate`, `awayOdds`, and `homeOdds`.
    """
    # Build movement_df_all
    movement_list = []
    # iterate unique game_ids
    for game_id in baseline_df["game_id"].dropna().unique():
        if not game_id:
            continue
        try:
            mv = fetch_line_movement(game_id, limit=movement_limit)
            mv = mv.copy()
            mv["game_id"] = game_id
            movement_list.append(mv)
        except Exception as e:
            print(f"Error fetching line movement for game {game_id}: {e}")
    
    # if no movements found, return baseline with empty movement cols
    if not movement_list:
        baseline_df[["lineDate", "awayOdds", "homeOdds"]] = pd.NA
        return baseline_df
        
    movement_df_all = pd.concat(movement_list, ignore_index=True)
    
    # merge on game_id
    full_df = baseline_df.merge(
        movement_df_all,
        on="game_id",
        how="inner"          
    )
    
    # Reorder columns
    cols_to_include = [
        'game_id', 'game_date', 'final_lineDate', 'api_team_id', 'is_home', 'is_away',
        # movement columns inserted here
        'lineDate', 'awayOdds', 'homeOdds',
        # then the rest of your original fields
        'spread', 'overOdds', 'underOdds', 'totalLine', 'totalResult',
        'moneyLineOdds', 'moneylineWinner', 'spreadOdds', 'spreadWinner',
        'api_team_score', 'opponent_score', 'api_team_won',
        'home_team_id', 'home_winner', 'home_score', 'home_curr_rank',
        'home_prev_rank', 'home_rank_summary', 'away_team_id', 'away_winner',
        'away_score', 'away_curr_rank', 'away_prev_rank', 'away_rank_summary'
    ]
    
    # Only include columns that exist in the DataFrame
    existing_cols = [col for col in cols_to_include if col in full_df.columns]
    full_df = full_df[existing_cols]
    
    return full_df

def collect_all_games_data(provider_id=1002, games_per_team=99999, max_workers=2):
    """
    Collect betting data for all NFL teams and their games.
    
    Parameters:
    -----------
    provider_id : int, optional (default=1002)
        The betting provider ID to use
    games_per_team : int, optional (default=99999)
        Number of games to fetch per team
    max_workers : int, optional (default=2)
        Maximum number of parallel workers for data collection
        
    Returns:
    --------
    tuple
        (team_games_df, unique_games_df, game_perspectives_df, game_movements_df)
        - team_games_df: Raw data with team perspective
        - unique_games_df: Deduplicated game data
        - game_perspectives_df: Combined team perspectives for each game
        - game_movements_df: Line movement data for each game
    """
    # Get all NFL teams
    teams_df = get_all_nfl_teams()
    print(f"Found {len(teams_df)} NFL teams")
    
    # Collect data for each team sequentially to avoid rate limiting
    all_team_data = []
    
    def process_team(team_row):
        team_id = team_row['id']
        team_name = team_row['displayName']
        print(f"Processing team: {team_name} (ID: {team_id})")
        
        try:
            # Add explicit retry logic with backoff
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    team_data = get_team_betting_data(team_id, limit=games_per_team, provider_id=provider_id)
                    
                    # Print how many games were found for debugging
                    num_games = len(team_data) if not team_data.empty else 0
                    print(f"  - Found {num_games} games for {team_name}")
                    
                    if not team_data.empty:
                        return team_data
                    elif attempt < max_retries - 1:
                        print(f"  - Empty result, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        print(f"  - Request error: {e}, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
            
            # If we get here with no data, return empty DataFrame
            print(f"  - No data found for {team_name} after {max_retries} attempts")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error processing team {team_name}: {str(e)}")
            return pd.DataFrame()
    
    # Option 1: Process teams in parallel with fewer workers to avoid rate limiting
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_team, row) for _, row in teams_df.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if not result.empty:
                all_team_data.append(result)
    
    # Option 2: Process teams sequentially if parallel is still problematic
    # Uncomment this and comment out the ThreadPoolExecutor block if needed
    """
    for _, team_row in teams_df.iterrows():
        result = process_team(team_row)
        if not result.empty:
            all_team_data.append(result)
        time.sleep(1)  # Add delay between teams to avoid rate limiting
    """
    
    # Combine all team data
    if not all_team_data:
        print("No team data collected.")
        return None, None, None, None
    
    team_games_df = pd.concat(all_team_data, ignore_index=True)
    print(f"Total games collected: {len(team_games_df)}")
    
    # Create a unique games dataset (deduplicated)
    unique_games_df = team_games_df.drop_duplicates(subset=['game_id'])[
        ['game_id', 'game_date', 'final_lineDate', 'home_team_id', 'away_team_id', 
         'home_score', 'away_score', 'home_winner', 'away_winner', 'totalLine', 
         'totalResult', 'spread', 'spreadWinner']
    ]
    print(f"Unique games after deduplication: {len(unique_games_df)}")
    
    # Create a game perspectives dataset (one row per team per game)
    game_perspectives_df = team_games_df[
        ['game_id', 'api_team_id', 'is_home', 'is_away', 'api_team_score',
         'opponent_score', 'api_team_won', 'spread', 'moneyLineOdds', 'spreadOdds']
    ]
    
    # Get line movement data with reduced parallelism for each unique game
    game_movements_df = []
    
    def process_game_movement(game_id):
        try:
            movement_df = fetch_line_movement(game_id)
            if not movement_df.empty:
                movement_df['game_id'] = game_id
                return movement_df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching movement for game {game_id}: {str(e)}")
            return pd.DataFrame()
    
    # Use a smaller number of workers for movement data to avoid rate limiting
    movement_max_workers = min(2, max_workers)
    with ThreadPoolExecutor(max_workers=movement_max_workers) as executor:
        game_ids = unique_games_df['game_id'].unique()
        print(f"Fetching line movements for {len(game_ids)} unique games...")
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(game_ids), batch_size):
            batch_ids = game_ids[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(game_ids) + batch_size - 1)//batch_size}")
            
            futures = [executor.submit(process_game_movement, game_id) for game_id in batch_ids]
            for future in as_completed(futures):
                result = future.result()
                if not result.empty:
                    game_movements_df.append(result)
            
            # Add a small delay between batches
            if i + batch_size < len(game_ids):
                print("Short delay between batches...")
                time.sleep(2)
    
    if not game_movements_df:
        print("No movement data collected.")
        return team_games_df, unique_games_df, game_perspectives_df, None
    
    game_movements_df = pd.concat(game_movements_df, ignore_index=True)
    print(f"Collected {len(game_movements_df)} line movement records")
    
    return team_games_df, unique_games_df, game_perspectives_df, game_movements_df

def create_features_dataset(team_games_df, game_movements_df, teams_df):
    """
    Create a feature-rich dataset for bet prediction modeling.
    
    Parameters:
    -----------
    team_games_df : pd.DataFrame
        The team-perspective games data
    game_movements_df : pd.DataFrame
        Line movement data for games
    teams_df : pd.DataFrame
        Teams information DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        Feature-rich dataset for modeling
    """
    # Merge team games with teams info
    team_games_with_names = pd.merge(
        team_games_df,
        teams_df.rename(columns={
            'id': 'api_team_id',
            'displayName': 'team_name',
            'abbreviation': 'team_abbr'
        }),
        on='api_team_id',
        how='left'
    )
    
    # Add home team and away team names
    home_teams = teams_df.rename(columns={
        'id': 'home_team_id', 
        'displayName': 'home_team_name',
        'abbreviation': 'home_team_abbr'
    })[['home_team_id', 'home_team_name', 'home_team_abbr']]
    
    away_teams = teams_df.rename(columns={
        'id': 'away_team_id', 
        'displayName': 'away_team_name',
        'abbreviation': 'away_team_abbr'
    })[['away_team_id', 'away_team_name', 'away_team_abbr']]
    
    team_games_with_names = pd.merge(team_games_with_names, home_teams, on='home_team_id', how='left')
    team_games_with_names = pd.merge(team_games_with_names, away_teams, on='away_team_id', how='left')
    
    # Initialize feature-rich dataframe
    team_games_with_features = team_games_with_names.copy()
    
    # Compute derived features based on line movements
    if game_movements_df is not None and not game_movements_df.empty:
        # Process line movements to create features
        movement_features = []
        
        for game_id in game_movements_df['game_id'].unique():
            game_movements = game_movements_df[game_movements_df['game_id'] == game_id]
            
            # Skip if we have fewer than 2 movements
            if len(game_movements) < 2:
                continue
                
            # Sort by date
            game_movements = game_movements.sort_values('lineDate')
            
            # Calculate features
            start_home_odds = game_movements['homeOdds'].iloc[0]
            end_home_odds = game_movements['homeOdds'].iloc[-1]
            start_away_odds = game_movements['awayOdds'].iloc[0]
            end_away_odds = game_movements['awayOdds'].iloc[-1]
            
            home_odds_change = end_home_odds - start_home_odds
            away_odds_change = end_away_odds - start_away_odds
            home_odds_pct_change = home_odds_change / start_home_odds if start_home_odds != 0 else 0
            away_odds_pct_change = away_odds_change / start_away_odds if start_away_odds != 0 else 0
            
            # Volatility features
            home_odds_std = game_movements['homeOdds'].std()
            away_odds_std = game_movements['awayOdds'].std()
            
            # Direction changes
            home_direction_changes = np.sum(np.diff(game_movements['homeOdds']) * np.roll(np.diff(game_movements['homeOdds']), 1) < 0)
            away_direction_changes = np.sum(np.diff(game_movements['awayOdds']) * np.roll(np.diff(game_movements['awayOdds']), 1) < 0)
            
            # Save features
            movement_features.append({
                'game_id': game_id,
                'start_home_odds': start_home_odds,
                'end_home_odds': end_home_odds,
                'start_away_odds': start_away_odds,
                'end_away_odds': end_away_odds,
                'home_odds_change': home_odds_change,
                'away_odds_change': away_odds_change,
                'home_odds_pct_change': home_odds_pct_change,
                'away_odds_pct_change': away_odds_pct_change,
                'home_odds_std': home_odds_std,
                'away_odds_std': away_odds_std,
                'home_direction_changes': home_direction_changes,
                'away_direction_changes': away_direction_changes,
                'movement_count': len(game_movements)
            })
        
        # Convert to DataFrame and merge with team games
        if movement_features:
            movement_features_df = pd.DataFrame(movement_features)
            team_games_with_features = pd.merge(
                team_games_with_names,
                movement_features_df,
                on='game_id',
                how='left'
            )
    
    # Create consistent game outcome indicators
    # For home/away win outcomes
    team_games_with_features['home_team_won'] = np.where(
        team_games_with_features['is_home'] == 1,  # For home team rows
        team_games_with_features['home_winner'] == True,  # Use home_winner value
        team_games_with_features['away_winner'] == False  # For away team rows, use inverse of away_winner
    ).astype(int)

    team_games_with_features['away_team_won'] = np.where(
        team_games_with_features['is_away'] == 1,  # For away team rows
        team_games_with_features['away_winner'] == True,  # Use away_winner value
        team_games_with_features['home_winner'] == False  # For home team rows, use inverse of home_winner
    ).astype(int)

    # For over/under - these are game-level outcomes, should be the same for both teams
    team_games_with_features['over_hit'] = (team_games_with_features['totalResult'] == 'O').astype(int)
    team_games_with_features['under_hit'] = (team_games_with_features['totalResult'] == 'U').astype(int)

    # For spread coverage - create consistent indicators across both team rows
    team_games_with_features['home_covered_spread'] = np.where(
        team_games_with_features['is_home'] == 1,  # For home team rows
        team_games_with_features['spreadWinner'] == True,  # Use spreadWinner directly
        team_games_with_features['spreadWinner'] == False  # For away team rows, use opposite of spreadWinner
    ).astype(int)

    team_games_with_features['away_covered_spread'] = np.where(
        team_games_with_features['is_away'] == 1,  # For away team rows
        team_games_with_features['spreadWinner'] == True,  # Use spreadWinner directly
        team_games_with_features['spreadWinner'] == False  # For home team rows, use opposite of spreadWinner
    ).astype(int)

    # Add a flag for whether the team in this specific row covered the spread
    team_games_with_features['this_team_covered'] = (team_games_with_features['spreadWinner'] == True).astype(int)
    
    # Ensure all feature columns have appropriate data types
    # Convert any boolean columns to integers (0/1)
    bool_cols = team_games_with_features.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        team_games_with_features[col] = team_games_with_features[col].astype(int)
    
    return team_games_with_features

def main():
    """
    Main function to collect and process NFL betting data with pagination.
    """
    print("Starting NFL betting data collection...")
    
    # Get teams data
    teams_df = get_all_nfl_teams()
    print(f"Found {len(teams_df)} NFL teams")
    
    # Get providers data
    providers_df = get_betting_providers()
    print(f"Found {len(providers_df)} betting providers")
    
    # Primary provider ID (default: 1002 for Caesars)
    provider_id = 1002
    provider_name = providers_df[providers_df['id'] == str(provider_id)]['name'].iloc[0] if not providers_df.empty else "Unknown"
    print(f"Using provider: {provider_name} (ID: {provider_id})")
    
    # Optional: Test a single team first to verify function works as expected
    print("\nTesting with a single team first...")
    # Use the Buffalo Bills as a test team (replace with a valid team ID)
    test_team_id = '2' # Buffalo Bills
    test_team_data = get_team_betting_data(test_team_id, limit=99999, provider_id=provider_id)
    print(f"Test single team returned {len(test_team_data)} games")
    
    # Collect comprehensive data with reduced parallelism and better error handling
    print("\nCollecting comprehensive game data (this may take a while)...")
    team_games_df, unique_games_df, game_perspectives_df, game_movements_df = collect_all_games_data(
        provider_id=provider_id,
        games_per_team=99999,
        max_workers=3  # Reduced parallelism to avoid rate limiting
    )
    
    if team_games_df is not None:
        print(f"Collected data for {len(unique_games_df)} unique games from {team_games_df['api_team_id'].nunique()} teams")
        
        # Create modeling dataset
        print("Creating feature-rich dataset for modeling...")
        modeling_df = create_features_dataset(team_games_df, game_movements_df, teams_df)
        
        # Save the datasets
        print("Saving datasets...")
        team_games_df.to_csv("team_games_data.csv", index=False)
        unique_games_df.to_csv("unique_games_data.csv", index=False)
        game_perspectives_df.to_csv("game_perspectives_data.csv", index=False)
        
        if game_movements_df is not None and not game_movements_df.empty:
            game_movements_df.to_csv("game_movements_data.csv", index=False)
        
        modeling_df.to_csv("betting_model_features.csv", index=False)
        
        print("Data collection and processing complete!")
        
        # Print summary statistics
        print("\n===== Data Collection Summary =====")
        print(f"Total teams: {teams_df.shape[0]}")
        print(f"Total games: {team_games_df.shape[0]}")
        print(f"Unique games: {unique_games_df.shape[0]}")
        print(f"Games per team (avg): {team_games_df.shape[0] / teams_df.shape[0]:.1f}")
        print(f"Team perspectives: {game_perspectives_df.shape[0]}")
        if game_movements_df is not None and not game_movements_df.empty:
            print(f"Line movements: {game_movements_df.shape[0]}")
        print(f"Final modeling dataset: {modeling_df.shape[0]} rows, {modeling_df.shape[1]} columns")
        print("===================================")
        
        return {
            'team_games_df': team_games_df,
            'unique_games_df': unique_games_df,
            'game_perspectives_df': game_perspectives_df,
            'game_movements_df': game_movements_df,
            'modeling_df': modeling_df
        }
    else:
        print("Data collection failed. Please check the logs for errors.")
        return None
    
    
if __name__ == "__main__":
    main()