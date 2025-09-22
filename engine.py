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

    def __init__(self, failure_rate: float = 0.0):
        # Containers
        self._market_data: List[MarketDataPoint] = []  # all ticks
        self._signals: List[Signal] = []               # all signals
        self._orders: List[Order] = []                 # all orders
        self._positions: Dict[str, Dict] = {}          # {symbol: {"quantity": int, "avg_price": float}}        
        self._failure_rate = failure_rate              # For simulating execution failures

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

    def run(self, strategies: List[Strategy]):
        """Run the backtest by feeding ticks to each strategy."""
        self._market_data.sort(key=lambda tick: tick.timestamp)
        logger.info(f"Starting backtest with {len(self._market_data)} ticks")
        
        for tick in self._market_data:
            try:
                for strat in strategies:
                    try:
                        # Step 1: Invoke each strategy to generate signals
                        signals = strat.generate_signals(tick)
                        if signals:
                            self._signals.extend(signals)
                            
                            # Step 2: Convert signals to orders with validation
                            for sig in signals:
                                try:
                                    order = self._signal_to_order(sig, tick.price)
                                    self._orders.append(order)
                                    
                                    # Step 3: Execute orders by updating portfolio
                                    self._execute_order(order)
                                    
                                except OrderError as e:
                                    logger.error(f"Order creation failed for signal {sig}: {e}")
                                    continue
                                except ExecutionError as e:
                                    logger.error(f"Order execution failed for {order.symbol}: {e}")
                                    # Mark order as failed but keep it in the list
                                    order.status = "FAILED"
                                    continue
                                except Exception as e:
                                    logger.error(f"Unexpected error processing signal {sig}: {e}")
                                    continue
                                    
                    except Exception as e:
                        logger.error(f"Strategy {strat.__class__.__name__} failed on tick {tick}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Critical error processing tick {tick}: {e}")
                continue
                
        logger.info(f"Backtest completed. Generated {len(self._signals)} signals, {len(self._orders)} orders")


    def _signal_to_order(self, signal: Signal, price: float) -> Order:
        """Convert a trading signal into an order at the current market price."""
        try:
            # Order quantity is always positive (Order validation requires this)
            # We'll store the signal side information separately or handle in execution
            qty = signal.qty  # Always positive as per Order validation
            
            # Create order with validation (Order.__init__ will validate parameters)
            order = Order(signal.symbol, qty, price, status="PENDING")
            
            # Store the original signal side for proper execution
            order._signal_side = signal.side  # Store side for execution logic
            
            logger.info(f"Created {signal.side} order: {order.symbol} {order.quantity}@{order.price} [{order.status}]")
            return order
            
        except Exception as e:
            raise OrderError(f"Failed to create order from signal: {e}")

    def _execute_order(self, order: Order):
        """Execute an order by updating portfolio positions with error handling."""
        # Simulate execution failure based on failure rate
        if random.random() < self._failure_rate:
            raise ExecutionError(f"Simulated execution failure for order {order.symbol}")
            
        try:
            # Update portfolio position
            self._update_positions(order)
            order.status = "FILLED"
            logger.info(f"Successfully executed order: {order.symbol} {order.quantity}@{order.price}")
            
        except Exception as e:
            raise ExecutionError(f"Portfolio update failed: {e}")

    def _update_positions(self, order: Order):
        """Update portfolio positions based on a filled order."""
        try:
            pos = self._positions.setdefault(order.symbol, {"quantity": 0, "avg_price": 0.0})
            
            # Determine the actual quantity change based on signal side
            signal_side = getattr(order, '_signal_side', 'BUY')  # Default to BUY if not set
            
            if signal_side == "BUY":
                # BUY order - increase position
                if pos["quantity"] == 0:
                    # Starting fresh position
                    pos["quantity"] = order.quantity
                    pos["avg_price"] = order.price
                else:
                    # Adding to existing position
                    total_cost = pos["quantity"] * pos["avg_price"] + order.quantity * order.price
                    pos["quantity"] += order.quantity
                    pos["avg_price"] = total_cost / pos["quantity"] if pos["quantity"] != 0 else 0.0
                    
            elif signal_side == "SELL":
                # SELL order - decrease position
                pos["quantity"] -= order.quantity  # Reduce position by order quantity
                if pos["quantity"] == 0:
                    pos["avg_price"] = 0.0  # Reset if flat
                elif pos["quantity"] < 0:
                    # Now we have a short position, use current price
                    pos["avg_price"] = order.price

            logger.info(f"Updated position ({signal_side}): {order.symbol} → Qty: {pos['quantity']}, Avg Price: {pos['avg_price']:.2f}")
            
        except Exception as e:
            raise ExecutionError(f"Failed to update position for {order.symbol}: {e}")
    
    def process_orders(self, orders: List[Order]) -> Dict:
        """Process a list of orders and return execution statistics."""
        results = {
            "total_orders": len(orders),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for order in orders:
            try:
                self._execute_order(order)
                self._orders.append(order)
                results["successful"] += 1
            except (OrderError, ExecutionError) as e:
                results["failed"] += 1
                results["errors"].append(f"Order {order.symbol}: {str(e)}")
                order.status = "FAILED"
                self._orders.append(order)  # Keep failed orders in the list
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Order {order.symbol}: Unexpected error - {str(e)}")
                order.status = "FAILED"
                self._orders.append(order)
                
        return results

    @property
    def positions(self) -> Dict[str, Dict]:
        return self._positions

    @property
    def orders(self) -> List[Order]:
        return self._orders

    @property
    def signals(self) -> List[Signal]:
        return self._signals

