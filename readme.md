# CSV-Based Algorithmic Trading Backtester

A comprehensive Python framework for backtesting algorithmic trading strategies using CSV market data. This system provides strategy development, execution simulation, performance analysis, and professional reporting capabilities.

## 🎯 Features

- **Multi-Strategy Support**: Test multiple trading strategies simultaneously
- **Real-Time Position Tracking**: Per-strategy position and capital tracking
- **Performance Analytics**: Comprehensive metrics including Sharpe ratio, drawdown analysis
- **Professional Reporting**: Automated generation of performance reports with charts
- **Error Simulation**: Built-in failure rate simulation for realistic testing
- **Extensible Architecture**: Easy to add new trading strategies and indicators

## 📁 Project Structure

```
CSV-Based-Algorithmic-Trading-Backtester/
├── main.py                    # Main execution script
├── engine.py                  # Core execution engine
├── models.py                  # Data models (MarketDataPoint, Order, Signal)
├── strategies.py              # Trading strategy implementations
├── reporting.py               # Performance analysis and reporting
├── data_generator.py          # Market data generation utilities
├── market_data.csv           # Sample market data
├── requirements.txt          # Python dependencies
├── test/                     # Test suite
│   └── test_module.py        # Unit tests
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/thibaut-dst/CSV-Based-Algorithmic-Trading-Backtester.git
   cd CSV-Based-Algorithmic-Trading-Backtester
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

#### 1. Standard Backtest Execution
Run the complete trading system with default strategies:

```bash
python main.py
```

This will:
- Load market data from `market_data.csv`
- Execute configured trading strategies
- Generate performance reports and charts
- Create `performance.md` with detailed analysis

#### 2. Error Handling Demo
Test the system with simulated execution failures:

```bash
python main.py --error-demo
```

This demonstrates:
- 20% execution failure rate simulation
- Error handling and recovery mechanisms
- Impact analysis of failed orders

#### 3. Custom Strategy Configuration
Modify `main.py` to test different strategies:

```python
def run_strategy_list():
    # Configure your strategies
    apple_strategy = RandomBuyAndSellStrategy(symbol="AAPL", capital=50000.0)
    amazon_sma = SMACrossoverStrategy(symbol="AMZN", short_window=10, long_window=30, capital=50000.0)
    
    strat_list = [apple_strategy, amazon_sma]
    
    # Run backtest
    engine = ExecutionEngine(failure_rate=0.0, initial_capital=100000.0)
    engine.load_data("market_data.csv")
    engine.run(strat_list)
```

## 🧪 Running Tests

### Unit Tests
Execute the test suite using pytest:

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with print statements visible
pytest -s

# Run specific test file
pytest test/test_module.py -v
```

### Test Coverage
The test suite includes:
- Order status updates and lifecycle testing
- Data model immutability verification  
- Market data point validation
- Strategy execution testing

### Creating New Tests
Add new test functions to `test/test_module.py`:

```python
def test_new_functionality():
    # Your test implementation
    assert expected_result == actual_result
```
