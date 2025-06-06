#!/usr/bin/env python3
import json
from flask import Flask, request, jsonify
import openai
from datetime import datetime

from src.config import OPENAI_API_KEY, API_HOST, API_PORT
from src.db.models import Adjustment, init_db
from src.utils.logger import logger

app = Flask(__name__)

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Initialize database session
db_session = init_db()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.route('/feedback', methods=['POST'])
def get_feedback():
    """
    Endpoint to receive trading data and return AI-generated adjustments.
    
    Expected payload:
    {
        "recent_trades": [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "entry_price": 50000,
                "exit_price": 50100,
                "pnl": 0.1,
                "exit_reason": "take_profit"
            }
        ],
        "last_50_bars": [
            {
                "timestamp": "2023-01-01T00:00:00",
                "open": 50000,
                "high": 50100,
                "low": 49900,
                "close": 50050,
                "volume": 10
            }
        ],
        "perf": {
            "win_rate": 0.6,
            "avg_return": 0.3
        }
    }
    
    Returns:
    {
        "adjustments": [
            {
                "type": "parameter",
                "parameter": "RSI_LONG_THRESHOLD",
                "old_value": 10,
                "new_value": 15,
                "reason": "Recent market conditions show earlier exhaustion of downtrends"
            }
        ]
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        if not all(k in data for k in ["recent_trades", "last_50_bars", "perf"]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Get AI feedback
        adjustments = get_ai_feedback(data)
        
        # Store adjustments in database
        for adjustment in adjustments:
            db_adjustment = Adjustment(
                adjustment_type=adjustment["type"],
                parameter=adjustment.get("parameter"),
                old_value=str(adjustment.get("old_value")),
                new_value=str(adjustment.get("new_value")),
                reason=adjustment.get("reason"),
                applied=False
            )
            db_session.add(db_adjustment)
        
        db_session.commit()
        
        return jsonify({"adjustments": adjustments})
    
    except Exception as e:
        logger.error(f"Error processing feedback request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_ai_feedback(data):
    """
    Get AI feedback using OpenAI API.
    
    Args:
        data: Dictionary with trading data
        
    Returns:
        list: Suggested adjustments
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set, returning empty adjustments")
        return []
    
    try:
        # Format data for the AI
        recent_trades_summary = summarize_trades(data["recent_trades"])
        market_conditions = analyze_market_conditions(data["last_50_bars"])
        performance = data["perf"]
        
        # Create prompt for OpenAI
        prompt = f"""
        You are an expert cryptocurrency trading advisor analyzing a BTC/USDT intra-day scalping strategy.

        The strategy uses:
        - EMA crossover (9 vs 21)
        - RSIâ‚‚ (long when < 10, short when > 90)
        - Volume filter (current volume > 1.5Ã— 20-period MA)
        
        Current performance:
        - Win rate: {performance['win_rate']*100:.1f}%
        - Average return per trade: {performance['avg_return']*100:.2f}%
        
        Recent trade summary:
        {recent_trades_summary}
        
        Market condition analysis:
        {market_conditions}
        
        Based on this information, suggest up to 2 specific parameter or rule adjustments to improve the strategy performance.
        Format your response as a JSON array of adjustments with fields:
        - type: "parameter" or "rule"
        - parameter: name of the parameter to adjust (only for "parameter" type)
        - old_value: current value
        - new_value: suggested value
        - reason: brief explanation for the adjustment
        
        Return ONLY the JSON array without any other text or explanation.
        """
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a cryptocurrency trading strategy advisor."},
                     {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract and parse the response
        ai_response = response.choices[0].message.content.strip()
        
        # Handle potential non-JSON responses
        try:
            # Try to find JSON array in the response
            start_idx = ai_response.find('[')
            end_idx = ai_response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > 0:
                json_str = ai_response[start_idx:end_idx]
                adjustments = json.loads(json_str)
            else:
                logger.warning(f"Failed to extract JSON from AI response: {ai_response}")
                adjustments = []
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse AI response as JSON: {ai_response}")
            adjustments = []
        
        return adjustments
    
    except Exception as e:
        logger.error(f"Error getting AI feedback: {str(e)}")
        return []

def summarize_trades(trades):
    """Create a text summary of recent trades."""
    if not trades:
        return "No recent trades."
    
    total = len(trades)
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = total - wins
    
    buy_trades = sum(1 for t in trades if t.get("side") == "buy")
    sell_trades = total - buy_trades
    
    stop_losses = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
    take_profits = sum(1 for t in trades if t.get("exit_reason") == "take_profit")
    
    summary = f"""
    - Total trades: {total}
    - Winning trades: {wins} ({(wins/total*100) if total > 0 else 0:.1f}%)
    - Losing trades: {losses} ({(losses/total*100) if total > 0 else 0:.1f}%)
    - Long positions: {buy_trades}
    - Short positions: {sell_trades}
    - Stop losses hit: {stop_losses} ({(stop_losses/total*100) if total > 0 else 0:.1f}%)
    - Take profits hit: {take_profits} ({(take_profits/total*100) if total > 0 else 0:.1f}%)
    """
    
    return summary

def analyze_market_conditions(bars):
    """Analyze recent market conditions from price bars."""
    if not bars:
        return "No recent price data."
    
    # Calculate basic statistics
    closes = [bar.get("close", 0) for bar in bars]
    highs = [bar.get("high", 0) for bar in bars]
    lows = [bar.get("low", 0) for bar in bars]
    volumes = [bar.get("volume", 0) for bar in bars]
    
    price_range = max(highs) - min(lows)
    avg_price = sum(closes) / len(closes)
    price_volatility = price_range / avg_price * 100
    
    # Calculate price direction
    first_price = closes[0]
    last_price = closes[-1]
    price_change = (last_price - first_price) / first_price * 100
    
    # Analyze volume
    avg_volume = sum(volumes) / len(volumes)
    volume_trend = "increasing" if volumes[-1] > avg_volume else "decreasing"
    
    analysis = f"""
    - Price range: ${min(lows):.0f} to ${max(highs):.0f}
    - Current price: ${last_price:.0f}
    - Price change: {price_change:.2f}%
    - Volatility: {price_volatility:.2f}%
    - Volume trend: {volume_trend}
    """
    
    return analysis

if __name__ == '__main__':
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT) 
