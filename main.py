#!/usr/bin/env python3
"""
Main orchestration script for the CSV-Based Algorithmic Trading Backtester.
This script demonstrates the complete workflow:
1. Data loading from CSV files
2. Strategy configuration and execution
3. Order processing and portfolio management
4. Results analysis and reporting
"""

import logging
from typing import List, Dict, Any
import random
from models import MarketDataPoint, Order, OrderError, ExecutionError
from engine import ExecutionEngine
from reporting import PerformanceAnalyzer
from strategies import SMACrossoverStrategy, PriceChangeMomentumStrategy, RandomBuyAndSellStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_strategy_list():

    test_strat_1 = RandomBuyAndSellStrategy(symbol="AAPL", capital = 100000.0)
    amaz_strat_2 = RandomBuyAndSellStrategy(symbol="AMZN", capital = 20000.0)

    engine = ExecutionEngine(failure_rate=0.0, initial_capital=100000.0)
    engine.load_data("market_data.csv")

    engine.run([test_strat_1, amaz_strat_2])
    #orchestrator = TradingSystemOrchestrator(failure_rate=0.1)

    # print("Loading Market Data")
    # if not orchestrator.load_market_data("market_data.csv"):
    #     print("Failed to load market data")
    #     return

    # print("Executing Backtest")
    # if not orchestrator.execute_backtest():
    #     print("Failed backtest.")
    #     return



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--error-demo":
        # Run just the error handling demo
        demo_error_handling()
    else:
        # Run the complete trading system demonstration
        # run_complete_trading_system()
        run_strategy_list()
        
       