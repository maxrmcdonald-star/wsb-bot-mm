import praw
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
import easyocr
import io

# --- Import config.py ---
import config

# --- Display Settings and Initializations ---
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1700)
analyzer = SentimentIntensityAnalyzer()

# --- Initialize the OCR Reader ---
print("Initializing OCR reader... (this may take a moment on first run)")
reader = easyocr.Reader(['en'])
print("OCR reader initialized.")

# --- Global Cache for API results ---
earnings_calendar_df = None

# --- API Key and Ticker Loading (from config.py) ---
MARKETSTACK_API_KEY = config.MARKETSTACK_API_KEY
FINNHUB_API_KEY = config.FINNHUB_API_KEY
def load_valid_tickers(api_key):
    print("Loading valid tickers from API... (Phase 1 of 3)")
    if not api_key or api_key == "YOUR_MARKETSTACK_API_KEY_HERE":
        print("Error: MarketStack API key not found in config.py.")
        return set()
    try:
        params = {'access_key': api_key, 'limit': 10000}
        response = requests.get('http://api.marketstack.com/v1/tickers', params=params)
        response.raise_for_status()
        data = response.json()
        ticker_list = data.get('data', [])
        valid_tickers = {item['symbol'] for item in ticker_list if '.' not in item['symbol'] and '$' not in item['symbol']}
        print(f"Successfully loaded {len(valid_tickers)} tickers.")
        return valid_tickers
    except requests.exceptions.RequestException as e:
        print(f"Error making API request to marketstack: {e}")
        return set()
VALID_TICKERS = load_valid_tickers(MARKETSTACK_API_KEY)

# --- Function to load the earnings calendar from Finnhub ---
def load_earnings_calendar():
    global earnings_calendar_df
    print("Loading 1-week earnings calendar from Finnhub...")
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_NEW_FINNHUB_KEY":
        print("Finnhub API key not found in config.py. Earnings dates will be unavailable.")
        return
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        one_week_later = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        url = f'https://finnhub.io/api/v1/calendar/earnings?from={today}&to={one_week_later}&token={FINNHUB_API_KEY}'
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if 'earningsCalendar' in data and data['earningsCalendar']:
            earnings_calendar_df = pd.DataFrame(data['earningsCalendar'])
            earnings_calendar_df.rename(columns={'symbol': 'ticker', 'date': 'reportDate', 'hour': 'earningsHour'}, inplace=True)
            earnings_calendar_df['reportDate'] = pd.to_datetime(earnings_calendar_df['reportDate'])
            print(f"Successfully loaded earnings calendar with {len(earnings_calendar_df)} upcoming events.")
        else:
            print("No upcoming earnings found in the next 7 days.")
    except Exception as e:
        print(f"Could not load or parse earnings calendar from Finnhub: {e}")
        print("Earnings dates will be unavailable for this run.")
load_earnings_calendar()

# --- Reddit Connection (from config.py) ---
***REMOVED***
    client_id=config.CLIENT_ID, client_secret=config.CLIENT_SECRET,
    user_agent=config.USER_AGENT, username=config.USERNAME, password=config.PASSWORD
)
print("Successfully connected to Reddit.")

# --- Helper Functions ---
def find_tickers(text):
    if not VALID_TICKERS: return []
    blacklist = {'AI', 'DD', 'CEO', 'ATH', 'FOR', 'ETH', 'DR', 'ONE', 'EDIT', 'BUY', 'SELL'}
    pattern = r'\b[A-Z]{1,5}\b'
    potential_tickers = re.findall(pattern, str(text))
    return list(set([t for t in potential_tickers if len(t) > 1 and t in VALID_TICKERS and t not in blacklist]))
def find_plays(text):
    plays_found = []
    tickers_in_text = find_tickers(text)
    if not tickers_in_text: return []
    instrument_pattern = r'(\$?\d+\.?\d*)\s*([CcPp])'
    date_pattern = r'(\d{1,2}[\/-]\d{1,2})'
    for ticker in tickers_in_text:
        ticker_pattern = r'(?i)\b' + re.escape(ticker) + r'\b'
        for match in re.finditer(ticker_pattern, text):
            search_window = text[match.end() : match.end() + 40]
            instrument_match = re.search(instrument_pattern, search_window)
            date_match = re.search(date_pattern, search_window)
            if instrument_match and date_match:
                strike, type = instrument_match.group(1), instrument_match.group(2).upper()
                date = date_match.group(1).replace('-', '/')
                plays_found.append(f"{ticker} {strike}{type} {date}")
    return plays_found
def get_sentiment(text):
    return analyzer.polarity_scores(str(text))['compound']
def get_stock_price(ticker_str):
    try:
        ticker = yf.Ticker(ticker_str)
        price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice')
        return price if price is not None else 0.0
    except Exception:
        print(f"\n[Warning] yfinance: Could not find stock data for '{ticker_str}'.", end='')
        return 0.0
def get_option_price(play_string):
    if not isinstance(play_string, str) or len(play_string.split()) != 3: return 0.0
    try:
        ticker_str, instrument, date_str = play_string.split()
        ticker = yf.Ticker(ticker_str)
        available_expirations = ticker.options
        if not available_expirations: return 0.0
        current_date = datetime.now()
        target_date_this_year = datetime.strptime(f"{date_str}/{current_date.year}", "%m/%d/%Y")
        target_date = target_date_this_year if target_date_this_year > current_date else target_date_this_year.replace(year=current_date.year + 1)
        closest_date = min(available_expirations, key=lambda d: abs(datetime.strptime(d, "%Y-%m-%d") - target_date))
        strike = float(re.findall(r'(\d+\.?\d*)', instrument)[0])
        contract_type = 'calls' if 'C' in instrument else 'puts'
        opt_chain = ticker.option_chain(closest_date)
        chain = getattr(opt_chain, contract_type)
        contract = chain[chain['strike'] == strike]
        if not contract.empty:
            price = contract.iloc[0]['lastPrice']
            if price == 0.0 and 'mark' in contract.columns: price = contract.iloc[0]['mark']
            if price == 0.0 and 'ask' in contract.columns: price = contract.iloc[0]['ask']
            return price
        return 0.0
    except Exception:
        print(f"\n[Warning] yfinance: Error processing option '{play_string}'.", end='')
        return 0.0
def get_next_earnings_info(ticker_str):
    if earnings_calendar_df is None or earnings_calendar_df.empty:
        return 'N/A', 'N/A'
    ticker_earnings = earnings_calendar_df[earnings_calendar_df['ticker'] == ticker_str]
    if ticker_earnings.empty: return 'N/A', 'N/A'
    today = pd.to_datetime(datetime.now().date())
    next_earnings_event = ticker_earnings.loc[ticker_earnings['reportDate'].idxmin()]
    next_earnings_date = next_earnings_event['reportDate']
    earnings_hour = str(next_earnings_event.get('earningsHour', '')).upper()
    if earnings_hour == "BMO": earnings_time = "AM"
    elif earnings_hour == "AMC": earnings_time = "PM"
    else: earnings_time = 'N/A'
    if pd.isna(next_earnings_date): return 'N/A', 'N/A'
    if next_earnings_date.date() == today.date():
        next_earnings_str = "EARNINGS TODAY"
    else:
        next_earnings_str = next_earnings_date.strftime('%Y-%m-%d')
    return next_earnings_str, earnings_time
def extract_text_from_image(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        result = reader.readtext(response.content, detail=0, paragraph=True)
        return " ".join(result)
    except Exception as e:
        print(f"\n[Warning] OCR Error: Could not process image {image_url}. Error: {e}", end='')
        return ""

# --- UPDATED: Function to export the summary with stats ---
def export_summary_to_csv(earnings_df, no_earnings_df, timestamp_str, posts, comments, images):
    filename = f"wsb_summary_{timestamp_str}.csv"
    print(f"\nExporting summary to {filename}...")
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write(f"WallStreetBets Sentiment Summary\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            # NEW: Add stats to the export file
            f.write("\n--- Data Collection Stats ---\n")
            f.write(f"Threads Analyzed: {posts}\n")
            f.write(f"Comments Analyzed: {comments}\n")
            f.write(f"Images Analyzed (OCR): {images}\n\n")
            
            f.write("--- TICKERS WITH UPCOMING EARNINGS (Sorted by Score) ---\n")
            if not earnings_df.empty:
                earnings_df.to_csv(f)
            else:
                f.write("No tickers with upcoming earnings found in this sample.\n")
            f.write("\n\n")
            
            f.write("--- OTHER TICKERS (Sorted by Score) ---\n")
            if not no_earnings_df.empty:
                no_earnings_df.to_csv(f)
            else:
                f.write("No other tickers found in this sample.\n")
            f.write("\n\n")

            f.write("'Breakeven % Change' is the stock price % change needed for the 'Top Play' option to become profitable at expiration.\n")
            f.write("'Sentiment' is now weighted by the upvotes of each mention.\n")
        print("Export complete.")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

# --- Data Collection Loop ---
subreddit = reddit.subreddit("wallstreetbets")
total_posts_to_fetch = 25
print(f"\nFetching pinned & {total_posts_to_fetch} recent posts... (Phase 2 of 3)")
all_ticker_mentions, all_plays_found = [], []
if VALID_TICKERS:
    # --- NEW: Initialize counters ---
    comments_processed_count = 0
    images_processed_count = 0
    
    stickied_posts = []
    try:
        stickied_posts.append(subreddit.sticky(number=1))
        stickied_posts.append(subreddit.sticky(number=2))
    except Exception: pass
    
    posts_to_process = stickied_posts
    for post in subreddit.new(limit=total_posts_to_fetch):
        if not post.stickied:
            posts_to_process.append(post)
    
    total_posts_processed = len(posts_to_process)
    
    for i, post in enumerate(posts_to_process):
        if post:
            post_score = post.score
            post_text = post.title + " " + post.selftext
            image_text = ""
            if hasattr(post, 'url') and post.url.endswith(('.jpg', '.jpeg', '.png')):
                print(f"\nFound image in post: {post.title[:50]}... Analyzing with OCR...", end='')
                image_text = extract_text_from_image(post.url)
                images_processed_count += 1 # Increment counter
            
            full_text = post_text + " " + image_text
            sentiment = get_sentiment(full_text)
            tickers = find_tickers(full_text)
            plays = find_plays(full_text)
            for ticker in tickers: all_ticker_mentions.append({'ticker': ticker, 'sentiment': sentiment, 'score': post_score})
            all_plays_found.extend(plays)
            
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                comments_processed_count += 1 # Increment counter
                comment_score = comment.score
                sentiment = get_sentiment(comment.body)
                tickers = find_tickers(comment.body)
                plays = find_plays(comment.body)
                for ticker in tickers: all_ticker_mentions.append({'ticker': ticker, 'sentiment': sentiment, 'score': comment_score})
                all_plays_found.extend(plays)
        print(f"Analysis progress: {int(((i + 1) / total_posts_processed) * 100)}% complete...", end='\r')

print("\n\nAggregating results and fetching prices... (Phase 3 of 3)")

if not all_ticker_mentions:
    print("No valid tickers found to analyze.")
else:
    # (Aggregation is the same and correct)
    df_sentiment = pd.DataFrame(all_ticker_mentions)
    df_sentiment['weighted_sentiment'] = df_sentiment['sentiment'] * df_sentiment['score']
    df_sentiment['abs_score'] = df_sentiment['score'].abs().replace(0, 1)
    agg_df = df_sentiment.groupby('ticker').agg(mention_count=('ticker', 'count'), total_weighted_sentiment=('weighted_sentiment', 'sum'), total_abs_score=('abs_score', 'sum'))
    agg_df['average_sentiment'] = agg_df['total_weighted_sentiment'] / agg_df['total_abs_score']
    summary_df = agg_df[['mention_count', 'average_sentiment']].copy()
    summary_df['hype_score'] = summary_df['mention_count'] * summary_df['average_sentiment']
    
    if all_plays_found:
        df_plays = pd.DataFrame(all_plays_found, columns=['play_string'])
        df_plays['ticker'] = df_plays['play_string'].str.split(' ').str[0]
        play_summary_df = df_plays.groupby('ticker').agg(
            Top_Play=('play_string', lambda x: x.value_counts().index[0]),
            Top_Play_Mentions=('play_string', lambda x: x.value_counts().iloc[0]),
            Total_Options_Chatter=('play_string', 'count')
        ).reset_index()
        final_df = summary_df.merge(play_summary_df, on='ticker', how='left')
    else:
        final_df = summary_df.copy()
        final_df[['Top_Play', 'Top_Play_Mentions', 'Total_Options_Chatter']] = 'N/A'

    final_df.reset_index(inplace=True)
    final_df[['Stock Price', 'Option Price', 'Breakeven Price', 'Breakeven % Change']] = 0.0
    final_df[['Next Earnings', 'Earnings Time']] = 'N/A'
    
    total_tickers_to_fetch = len(final_df)
    for index, row in final_df.iterrows():
        stock_price = get_stock_price(row['ticker'])
        option_price = get_option_price(row['Top_Play'])
        next_earnings, earnings_time = get_next_earnings_info(row['ticker'])
        final_df.loc[index, 'Stock Price'] = stock_price
        final_df.loc[index, 'Option Price'] = option_price
        final_df.loc[index, 'Next Earnings'] = next_earnings
        final_df.loc[index, 'Earnings Time'] = earnings_time
        if row['Top_Play'] != 'N/A' and stock_price > 0 and option_price > 0:
            try:
                instrument = row['Top_Play'].split()[1]
                strike_price = float(re.findall(r'(\d+\.?\d*)', instrument)[0])
                option_type = 'C' if 'C' in instrument.upper() else 'P'
                if option_type == 'C': breakeven_price = strike_price + option_price
                else: breakeven_price = strike_price - option_price
                percent_change = ((breakeven_price - stock_price) / stock_price) * 100
                final_df.loc[index, 'Breakeven Price'] = breakeven_price
                final_df.loc[index, 'Breakeven % Change'] = percent_change
            except Exception: pass
        print(f"Fetching market data: {int(((index + 1) / total_tickers_to_fetch) * 100)}% complete...", end='\r')

    final_df.set_index('ticker', inplace=True)
    final_df = final_df.rename(columns={
        'mention_count': 'Mentions', 'average_sentiment': 'Sentiment',
        'hype_score': 'Score', 'Top_Play': 'Top Play',
        'Top_Play_Mentions': 'Top Play Mentions', 'Total_Options_Chatter': 'Options Chatter'
    })
    final_df.fillna({'Top Play': 'N/A', 'Top Play Mentions': 0, 'Options Chatter': 0, 'Next Earnings': 'N/A', 'Earnings Time': 'N/A'}, inplace=True)
    
    numeric_cols = ['Score', 'Breakeven Price', 'Breakeven % Change']
    for col in numeric_cols: final_df[col] = final_df[col].round(2)
    final_df['Sentiment'] = final_df['Sentiment'].round(4)
    
    final_cols = ['Mentions', 'Sentiment', 'Score', 'Stock Price', 'Top Play', 'Top Play Mentions', 'Options Chatter', 'Option Price', 'Breakeven Price', 'Breakeven % Change', 'Next Earnings', 'Earnings Time']
    final_df = final_df[final_cols]

    df_with_earnings = final_df[final_df['Next Earnings'] != 'N/A']
    df_without_earnings = final_df[final_df['Next Earnings'] == 'N/A']
    df_with_earnings = df_with_earnings.sort_values(by='Score', ascending=False)
    df_without_earnings = df_without_earnings.sort_values(by='Score', ascending=False)
    
    timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    print("\n" + "="*170)
    print("                                                                         WALLSTREETBETS COMPREHENSIVE SUMMARY")
    print("="*170)
    
    print("\n" + "--- TICKERS WITH UPCOMING EARNINGS (Sorted by Score) ---".center(170))
    if not df_with_earnings.empty:
        print(df_with_earnings)
    else:
        print("No tickers with upcoming earnings found in this sample.")

    print("\n" + "--- OTHER TICKERS (Sorted by Score) ---".center(170))
    if not df_without_earnings.empty:
        print(df_without_earnings)
    else:
        print("No other tickers found in this sample.")

    # --- NEW: Print the final data collection stats ---
    print("\n" + "--- Data Collection Stats ---".center(170))
    print(f"Threads Analyzed: {total_posts_processed}")
    print(f"Comments Analyzed: {comments_processed_count}")
    print(f"Images Analyzed (OCR): {images_processed_count}")

    print("\n'Breakeven % Change' is the stock price % change needed for the 'Top Play' option to become profitable at expiration.")
    print("'Sentiment' is now weighted by the upvotes of each mention.")
    
    # --- NEW: Call the export function with the stats ---
    export_summary_to_csv(df_with_earnings, df_without_earnings, timestamp_str, total_posts_processed, comments_processed_count, images_processed_count)