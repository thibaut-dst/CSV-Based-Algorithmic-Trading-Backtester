# engine.py
import csv
import logging
import random
from datetime import datetime
from typing import List, Dict

from models import MarketDataPoint, Signal, Order, OrderError, ExecutionError
from strategies import Strategy

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Core engine that ties everything together:
    - Buffers market data
    - Runs strategies to generate signals
    - Converts signals to orders
    - Updates positions when orders are "executed"
    """

    def __init__(self, failure_rate: float = 0.0, initial_capital: float = 100000.0):
        # Containers
        self._market_data: List[MarketDataPoint] = []  # all ticks
        self._signals: List[Signal] = []               # all signals
        self._orders: List[Order] = []                 # all orders
        self._positions: Dict[str, Dict] = {}          # {symbol: {"quantity": int, "avg_price": float}}        
        self._failure_rate = failure_rate              # For simulating execution failures
        
        # Capital and performance tracking
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        
        # Per-strategy tracking
        self._strategy_signals: Dict[str, List[Signal]] = {}  # {strategy_name: [signals]}
        self._strategy_orders: Dict[str, List[Order]] = {}    # {strategy_name: [orders]}
        self._strategy_capital: Dict[str, float] = {}         # {strategy_name: allocated_capital}
        
        # Historical tracking for performance analysis
        self._capital_history: List[Dict] = []                # Track capital over time

    def load_data(self, csv_path: str):
        """Read market data from a CSV file and store as MarketDataPoint list."""
        with open(csv_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                tick = MarketDataPoint(
                    timestamp=datetime.fromisoformat(row[0]),
                    symbol=row[1],
                    price=float(row[2]),
                )
                self._market_data.append(tick)
        logger.info(f"Loaded {len(self._market_data)} ticks from {csv_path}")

    def initialize_strategies(self, strategies: List[Strategy]):
        """Initialize strategy tracking and allocate capital equally among strategies."""
        if not strategies:
            raise ValueError("Cannot initialize with empty strategy list")
        
        capital_per_strategy = self._initial_capital / len(strategies)
        
        for strategy in strategies:
            strategy_name = f"{strategy.__class__.__name__}_{strategy._symbol}"
            self._strategy_signals[strategy_name] = []
            self._strategy_orders[strategy_name] = []
            self._strategy_capital[strategy_name] = capital_per_strategy
            logger.info(f"Allocated ${capital_per_strategy:.2f} to {strategy_name}")
        
        # Record initial capital allocation
        initial_snapshot = {
            "timestamp": None,  # Initial state
            "total_capital": self._current_capital,
            "strategies": self._strategy_capital.copy()
        }
        self._capital_history.append(initial_snapshot)

    def run(self, strategies: List[Strategy]):
        """Run the backtest by processing data symbol-by-symbol for each strategy."""
        logger.info(f"Starting backtest with {len(self._market_data)} ticks")
        
        # Initialize strategies and capital allocation
        self.initialize_strategies(strategies)
        
        self._market_data.sort(key=lambda tick: tick.timestamp)
        for tick in self._market_data: 
            for strategy in strategies:
                # if the strategy trades this symbol it will generate signals otherwise it will return []
                try:
                    signals = strategy.generate_signals(tick)
                    strategy_name = f"{strategy.__class__.__name__}_{strategy._symbol}"
                    
                    for signal in signals:
                        try:
                            order = Order(signal.symbol, signal.qty, tick.price, "PENDING")                                
                            self._execute_order_direct(order, signal.side, strategy_name)                                
                            
                            # Store in global lists
                            self._signals.append(signal) 
                            self._orders.append(order)
                            
                            # Store in per-strategy lists
                            self._strategy_signals[strategy_name].append(signal)
                            self._strategy_orders[strategy_name].append(order)
                            
                        except (OrderError, ExecutionError) as e:
                            logger.error(f"Order failed for {signal.symbol}: {e}")
                            # Still store failed order for analysis
                            if 'order' in locals():
                                order.status = "FAILED"
                                self._orders.append(order)
                                self._strategy_orders[strategy_name].append(order)
                        except Exception as e:
                            logger.error(f"Unexpected error processing signal: {e}")
                            
                except Exception as e:
                    logger.error(f"Strategy {strategy.__class__.__name__} failed on {tick.symbol}: {e}")
                    continue
                    
        logger.info(f"Backtest completed. Generated {len(self._signals)} signals, {len(self._orders)} orders")


    def _execute_order_direct(self, order: Order, signal_side: str, strategy_name: str):
        """Execute order immediately and update positions with capital checks."""
        # Simulate execution failure
        if random.random() < self._failure_rate:
            raise ExecutionError(f"Simulated execution failure for {order.symbol}")
        
        # Calculate order value
        order_value = order.quantity * order.price
        
        # Check if strategy has enough capital for BUY orders
        if signal_side == "BUY":
            strategy_capital = self._strategy_capital[strategy_name]
            if order_value > strategy_capital:
                raise ExecutionError(f"Insufficient capital for {strategy_name}: need ${order_value:.2f}, have ${strategy_capital:.2f}")
            
            # Deduct capital for BUY orders
            self._strategy_capital[strategy_name] -= order_value
            self._current_capital -= order_value
        
        # Update position directly
        pos = self._positions.setdefault(order.symbol, {"quantity": 0, "avg_price": 0.0})
        
        if signal_side == "BUY":
            if pos["quantity"] == 0:
                pos["quantity"] = order.quantity
                pos["avg_price"] = order.price
            else:
                total_cost = pos["quantity"] * pos["avg_price"] + order.quantity * order.price
                pos["quantity"] += order.quantity
                pos["avg_price"] = total_cost / pos["quantity"]
        elif signal_side == "SELL":
            # Add capital back for SELL orders
            self._strategy_capital[strategy_name] += order_value
            self._current_capital += order_value
            
            pos["quantity"] -= order.quantity
            if pos["quantity"] == 0:
                pos["avg_price"] = 0.0
            elif pos["quantity"] < 0:
                pos["avg_price"] = order.price
        
        order.status = "FILLED"
        logger.info(f"Executed {signal_side}: {order.symbol} {order.quantity}@{order.price:.2f} | Strategy: {strategy_name} | Capital: ${self._strategy_capital[strategy_name]:.2f}")
        
        # Record capital snapshot after each execution
        self._record_capital_snapshot(order.symbol)

    def _record_capital_snapshot(self, symbol: str):
        """Record current capital state for performance tracking."""
        # Calculate total holdings value for each strategy
        strategy_holdings = {}
        for strategy_name in self._strategy_capital.keys():
            cash = self._strategy_capital[strategy_name]
            
            # Calculate position value for this strategy's symbol
            # Extract symbol from strategy name (assumes format: StrategyName_SYMBOL)
            strategy_symbol = strategy_name.split('_')[-1]
            holdings_value = 0.0
            
            if strategy_symbol in self._positions:
                pos = self._positions[strategy_symbol]
                holdings_value = pos["quantity"] * pos["avg_price"] if pos["quantity"] > 0 else 0.0
            
            strategy_holdings[strategy_name] = {
                "cash": cash,
                "holdings": holdings_value,
                "total": cash + holdings_value
            }
        
        snapshot = {
            "symbol": symbol,
            "total_capital": self._current_capital,
            "strategies": strategy_holdings
        }
        self._capital_history.append(snapshot)

    @property
    def positions(self) -> Dict[str, Dict]:
        return self._positions

    @property
    def orders(self) -> List[Order]:
        return self._orders

    @property
    def signals(self) -> List[Signal]:
        return self._signals
    
    @property
    def strategy_signals(self) -> Dict[str, List[Signal]]:
        """Get signals grouped by strategy."""
        return self._strategy_signals
    
    @property
    def strategy_orders(self) -> Dict[str, List[Order]]:
        """Get orders grouped by strategy."""
        return self._strategy_orders
    
    @property
    def strategy_capital(self) -> Dict[str, float]:
        """Get capital allocation per strategy."""
        return self._strategy_capital
    
    @property
    def initial_capital(self) -> float:
        """Get initial capital amount."""
        return self._initial_capital
    
    @property
    def current_capital(self) -> float:
        """Get current available capital."""
        return self._current_capital
    
    @property
    def capital_history(self) -> List[Dict]:
        """Get historical capital tracking data."""
        return self._capital_history
    
    def get_capital_summary(self) -> Dict[str, float]:
        """Get a summary of capital allocation and remaining balances."""
        # Calculate current total holdings for each strategy
        current_strategy_totals = {}
        total_portfolio_value = 0.0
        
        for strategy_name in self._strategy_capital.keys():
            cash = self._strategy_capital[strategy_name]
            
            # Extract symbol from strategy name
            strategy_symbol = strategy_name.split('_')[-1]
            holdings_value = 0.0
            
            if strategy_symbol in self._positions:
                pos = self._positions[strategy_symbol]
                # For current market value, we'd need current prices. Using avg_price as approximation
                holdings_value = pos["quantity"] * pos["avg_price"] if pos["quantity"] > 0 else 0.0
            
            total_value = cash + holdings_value
            current_strategy_totals[strategy_name] = total_value
            total_portfolio_value += total_value
        
        summary = {
            "initial_capital": self._initial_capital,
            "current_total_cash": self._current_capital,
            "current_portfolio_value": total_portfolio_value,
            "allocated_cash": sum(self._strategy_capital.values()),
            "strategies_current": current_strategy_totals,
            "capital_history": self._capital_history
        }
        return summary

