import csv
import dataclasses
import logging
from dataclasses import dataclass, FrozenInstanceError
from datetime import datetime
from models import MarketDataPoint, Order, OrderError, ExecutionError, ExecutionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

market_data_points = []

positions = {
    "AAPL": {"quantity": 0, "avg_price": 0.0},
    "MSFT": {"quantity": 0, "avg_price": 0.0}
}


with open('market_data.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row if present
    for row in reader:
        point = MarketDataPoint(
            timestamp=datetime.fromisoformat(row[0]),
            symbol=row[1],
            price=float(row[2])
        )
        market_data_points.append(point)

# Demo the error handling system
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

if __name__ == "__main__":
    demo_error_handling()