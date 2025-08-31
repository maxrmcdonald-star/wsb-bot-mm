# app.py - The Flask Web Server
from flask import Flask, jsonify
import bot # Import our analysis script

app = Flask(__name__)

# This is our main API endpoint.
# When someone visits our website's URL, this function will run.
@app.route('/')
def get_summary():
    print("Request received! Running WSB analysis...")
    # Call the main function from our bot script
    earnings_df, no_earnings_df = bot.run_analysis()
    
    # Convert the pandas DataFrames to a JSON-friendly format (a list of dictionaries)
    earnings_json = earnings_df.reset_index().to_dict(orient='records')
    no_earnings_json = no_earnings_df.reset_index().to_dict(orient='records')
    
    # Create the final JSON response for our mobile app
    response_data = {
        "tickers_with_earnings": earnings_json,
        "other_tickers": no_earnings_json,
        "last_updated": bot.datetime.now().isoformat()
    }
    
    print("Analysis complete. Sending JSON response.")
    # The jsonify function correctly formats our data as a web response
    return jsonify(response_data)

# This allows us to run the server locally for testing
if __name__ == '__main__':
    app.run(debug=True)