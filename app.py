# app.py
from flask import Flask, jsonify
import redis
import os
import json

app = Flask(__name__)
REDIS_URL = os.environ.get('REDIS_URL')

@app.route('/')
def get_summary():
    if not REDIS_URL:
        return jsonify({"error": "Redis not configured"}), 500
    try:
        r = redis.from_url(REDIS_URL)
        # Get the latest summary from the cache
        cached_data = r.get('wsb_summary')
        if cached_data:
            # If data exists, return it
            return jsonify(json.loads(cached_data))
        else:
            # If cache is empty (worker hasn't run yet), return a waiting message
            return jsonify({"status": "Analysis in progress. Please check back in a few minutes."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)