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
from strategies import SMACrossoverStrategy, RandomBuyAndSellStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_strategy_list():

    test_strat_1 = RandomBuyAndSellStrategy(symbol="AAPL", capital = 100000.0)
    amaz_strat_2 = RandomBuyAndSellStrategy(symbol="AMZN", capital = 20000.0)
    sma_amaz_strat = SMACrossoverStrategy(symbol="AMZN", short_window=10, long_window=30, capital = 30000.0)

    # strat_list = [test_strat_1, amaz_strat_2, sma_amaz_strat]
    strat_list = [test_strat_1]


    engine = ExecutionEngine(failure_rate=0.0, initial_capital=100000.0)
    engine.load_data("market_data.csv")

    engine.run(strat_list)
    
    # Generate performance reports and plots
    from reporting import print_backtest_results, generate_strategy_performance_plots
    print_backtest_results(engine, strat_list)
    generate_strategy_performance_plots(engine, strat_list)




if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--error-demo":
        # Run just the error handling demo
        print("Error demo not implemented")
    else:
        # Run the complete trading system demonstration
        run_strategy_list()
        
def run_error_demo():
    """Demonstrate error handling and failure simulation with 20% failure rate."""
    print("=" * 60)
    print("ERROR HANDLING AND FAILURE SIMULATION DEMO")
    print("=" * 60)
    
    # Create strategies
    test_strat = RandomBuyAndSellStrategy(symbol="AAPL", capital=10000.0, probability=0.3)
    sma_strat = SMACrossoverStrategy(symbol="AMZN", short_window=5, long_window=15, capital=15000.0)
    
    print("Testing with 20% failure rate...")
    
    engine = ExecutionEngine(failure_rate=0.2, initial_capital=100000.0)
    engine.load_data("market_data.csv")
    
    # Run backtest
    engine.run([test_strat, sma_strat])
    
    # Analyze results
    total_signals = len(engine.signals)
    total_orders = len(engine.orders)
    failed_orders = len([order for order in engine.orders if order.status == "FAILED"])
    filled_orders = len([order for order in engine.orders if order.status == "FILLED"])
    
    print(f"Total signals generated: {total_signals}")
    print(f"Total orders created: {total_orders}")
    print(f"Successfully filled: {filled_orders}")
    print(f"Failed orders: {failed_orders}")
    print(f"Success rate: {(filled_orders/total_orders*100):.1f}%" if total_orders > 0 else "N/A")
    
    print("\n" + "=" * 60)
    print("ERROR DEMO COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--error-demo":
        # Run the error handling and failure simulation demo
        run_error_demo()
    else:
        # Run the complete trading system demonstration
        run_strategy_list()