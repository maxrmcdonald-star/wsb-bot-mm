# worker.py
import bot
import redis
import os
import json

# Get the Redis URL from the environment variables
REDIS_URL = os.environ.get('REDIS_URL')

def run_and_cache():
    print("Worker starting: Running WSB analysis...")
    if not REDIS_URL:
        print("Error: REDIS_URL not found in environment. Cannot save results.")
        return

    # Run the main analysis function from bot.py
    earnings_df, no_earnings_df = bot.run_analysis()

    # Convert the results to JSON
    earnings_json = earnings_df.reset_index().to_dict(orient='records')
    no_earnings_json = no_earnings_df.reset_index().to_dict(orient='records')
    
    response_data = {
        "tickers_with_earnings": earnings_json,
        "other_tickers": no_earnings_json,
        "last_updated": bot.datetime.now().isoformat()
    }
    
    # Save the final JSON to our Redis database
    try:
        r = redis.from_url(REDIS_URL)
        r.set('wsb_summary', json.dumps(response_data))
        print("Analysis complete. Results saved to Redis cache.")
    except Exception as e:
        print(f"Error connecting to or saving to Redis: {e}")

if __name__ == '__main__':
    run_and_cache()
