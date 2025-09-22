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

class TradingSystemOrchestrator:
    """
    Main orchestrator class that manages the complete trading system workflow.
    Handles data loading, strategy execution, and performance reporting.
    """
    
    def __init__(self, failure_rate: float = 0.0):
        """Initialize the trading system orchestrator."""
        self.engine = ExecutionEngine(failure_rate=failure_rate)
        self.strategies = []
        self.performance_analyzer = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_market_data(self, csv_file_path: str) -> bool:
        """
        Load market data from CSV file.
        
        Args:
            csv_file_path: Path to the CSV file containing market data
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading market data from {csv_file_path}")
            self.engine.load_data(csv_file_path)
            self.logger.info(f"Successfully loaded {len(self.engine._market_data)} market data points")
            return True
        except FileNotFoundError:
            self.logger.error(f"Market data file not found: {csv_file_path}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            return False







    
    
    def configure_strategies(self, strategy_configs: List[Dict[str, Any]]) -> bool:
        """
        Configure trading strategies based on provided configurations.
        
        Args:
            strategy_configs: List of strategy configuration dictionaries
            
        Returns:
            bool: True if strategies configured successfully, False otherwise
        """
        try:
            self.strategies = []
            
            for config in strategy_configs:
                strategy_type = config.get('type')
                symbol = config.get('symbol', 'AAPL')
                
                if strategy_type == 'SMA_CROSSOVER':
                    strategy = SMACrossoverStrategy(
                        symbol=symbol,
                        short_window=config.get('short_window', 5),
                        long_window=config.get('long_window', 20),
                        qty=config.get('quantity', 10)
                    )
                elif strategy_type == 'MOMENTUM':
                    strategy = PriceChangeMomentumStrategy(
                        symbol=symbol,
                        threshold=config.get('threshold', 2.0),
                        qty=config.get('quantity', 5)
                    )
                else:
                    self.logger.error(f"Unknown strategy type: {strategy_type}")
                    continue
                
                self.strategies.append(strategy)
                self.logger.info(f"Configured {strategy_type} strategy for {symbol}")
            
            self.logger.info(f"Successfully configured {len(self.strategies)} strategies")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to configure strategies: {e}")
            return False
    
    def execute_backtest(self) -> bool:
        """
        Execute the complete backtesting process.
        
        Returns:
            bool: True if backtest completed successfully, False otherwise
        """
        try:
            if not self.strategies:
                self.logger.error("No strategies configured. Cannot run backtest.")
                return False
            
            if not self.engine._market_data:
                self.logger.error("No market data loaded. Cannot run backtest.")
                return False
            
            self.logger.info("Starting backtest execution...")
            self.engine.run(self.strategies)
            
            # Initialize performance analyzer after backtest completion
            self.performance_analyzer = PerformanceAnalyzer(self.engine)
            
            self.logger.info("Backtest execution completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            return False
    
    def generate_performance_report(self, filename: str = "performance.md") -> bool:
        """
        Generate comprehensive performance report.
        
        Args:
            filename: Output filename for the report
            
        Returns:
            bool: True if report generated successfully, False otherwise
        """
        try:
            if not self.performance_analyzer:
                self.logger.error("No performance analyzer available. Run backtest first.")
                return False
            
            self.logger.info("Generating comprehensive performance report...")
            
            # Generate the markdown report
            report_path = self.performance_analyzer.generate_markdown_report(filename)
            
            # Print summary to console
            self.performance_analyzer.print_summary()
            
            self.logger.info(f"Performance report generated: {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return False


def demo_error_handling():
    """Demonstrate the custom exception handling system"""
    print("=== Error Handling Demo ===")
    
    # Create execution engine with 20% failure rate for demo
    engine = ExecutionEngine(failure_rate=0.2)
    
    # Test valid orders
    valid_orders = [
        Order("AAPL", 100, 150.50, "PENDING"),
        Order("GOOGL", 50, 2800.75, "PENDING"),
        Order("MSFT", 200, 300.25, "PENDING")
    ]
    
    # Test invalid orders (should raise OrderError)
    print("\n--- Testing Order Validation ---")
    try:
        invalid_order = Order("", 100, 150.50, "PENDING")  # Empty symbol
    except OrderError as e:
        print(f"✓ Caught OrderError: {e}")
    
    try:
        invalid_order = Order("AAPL", -50, 150.50, "PENDING")  # Negative quantity
    except OrderError as e:
        print(f"✓ Caught OrderError: {e}")
    
    try:
        invalid_order = Order("AAPL", 100, -150.50, "PENDING")  # Negative price
    except OrderError as e:
        print(f"✓ Caught OrderError: {e}")
    
    # Test execution engine with error handling
    print("\n--- Testing Execution Engine ---")
    results = engine.process_orders(valid_orders)
    
    print(f"Total orders: {results['total_orders']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print("\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")


def run_strategy_list():

    test_strat_1 = RandomBuyAndSellStrategy(symbol="AAPL", capital = 100000.0)

    orchestrator = TradingSystemOrchestrator(failure_rate=0.1)

    print("Loading Market Data")
    if not orchestrator.load_market_data("market_data.csv"):
        print("Failed to load market data")
        return

    print("Executing Backtest")
    if not orchestrator.execute_backtest():
        print("Failed backtest.")
        return





def run_complete_trading_system():
    """Run a complete demonstration of the trading system."""
    print("🚀 Starting Complete Trading System Demonstration")
    
    # Initialize the orchestrator with initial capital
    orchestrator = TradingSystemOrchestrator(failure_rate=0.1)
    
    # Step 1: Load market data
    print("\n📂 Step 1: Loading Market Data")
    if not orchestrator.load_market_data("market_data.csv"):
        print("❌ Failed to load market data. Exiting.")
        return
    
    # Step 2: Configure strategies
    print("\n⚙️  Step 2: Configuring Trading Strategies")
    strategy_configs = [
        {
            "type": "SMA_CROSSOVER",
            "symbol": "AAPL",
            "short_window": 5,
            "long_window": 15,
            "quantity": 10
        },
        {
            "type": "MOMENTUM", 
            "symbol": "AAPL",
            "threshold": 1.5,
            "quantity": 5
        }
    ]
    
    if not orchestrator.configure_strategies(strategy_configs):
        print("❌ Failed to configure strategies. Exiting.")
        return
    
    # Step 3: Execute backtest
    print("\n🎯 Step 3: Executing Backtest")
    if not orchestrator.execute_backtest():
        print("❌ Backtest execution failed. Exiting.")
        return
    
    # Step 4: Generate comprehensive performance report
    print("\n📊 Step 4: Generating Performance Report")
    if not orchestrator.generate_performance_report("performance.md"):
        print("❌ Failed to generate performance report.")
        return
    
    print("\n✅ Trading system demonstration completed successfully!")
    print("📄 Check 'performance.md' for detailed analysis and recommendations.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--error-demo":
        # Run just the error handling demo
        demo_error_handling()
    else:
        # Run the complete trading system demonstration
        # run_complete_trading_system()
        run_strategy_list()
        
        # Also run the error handling demo
        print("\n" + "="*60)
        demo_error_handling()