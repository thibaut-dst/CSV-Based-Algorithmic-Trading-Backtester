#!/usr/bin/env python3
"""
Performance Reporting Module for CSV-Based Algorithmic Trading Backtester

This module provides comprehensive performance analysis including:
- Total return calculation
- Periodic returns analysis
- Sharpe ratio computation
- Maximum drawdown analysis
- Professional reporting in Markdown format
- Professional equity curve visualization with matplotlib
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import os

from models import MarketDataPoint, Order, Signal
from engine import ExecutionEngine

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    Calculates key metrics and generates professional reports.
    """
    
    def __init__(self, engine: ExecutionEngine, initial_capital: float = 100000.0):
        """
        Initialize the performance analyzer.
        
        Args:
            engine: ExecutionEngine instance with completed backtest
            initial_capital: Starting capital for performance calculations
        """
        self.engine = engine
        self.initial_capital = initial_capital
        self.portfolio_history = []
        self.returns_series = []
        self.metrics = {}
        
        logger.info(f"Initialized PerformanceAnalyzer with ${initial_capital:,.2f} initial capital")
    
    def calculate_portfolio_history(self) -> List[Dict[str, Any]]:
        """
        Calculate portfolio value at each point in time during the backtest.
        
        Returns:
            List of portfolio snapshots with timestamp, value, positions, etc.
        """
        if not self.engine._market_data:
            logger.warning("No market data available for portfolio calculation")
            return []
        
        logger.info("Calculating portfolio value history...")
        
        # Start with initial portfolio
        current_portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }
        
        # Track orders by timestamp for portfolio updates
        orders_by_time = {}
        for order in self.engine.orders:
            if order.status == "FILLED":
                # Find the corresponding tick for this order
                for tick in self.engine._market_data:
                    # Match orders to ticks (simplified - in reality you'd want more precise matching)
                    if tick.symbol == order.symbol:
                        if tick.timestamp not in orders_by_time:
                            orders_by_time[tick.timestamp] = []
                        orders_by_time[tick.timestamp].append(order)
                        break
        
        portfolio_snapshots = []
        
        # Process each tick chronologically
        for tick in self.engine._market_data:
            # Update portfolio with any orders executed at this time
            if tick.timestamp in orders_by_time:
                for order in orders_by_time[tick.timestamp]:
                    signal_side = getattr(order, '_signal_side', 'BUY')
                    
                    if signal_side == 'BUY':
                        # Buy order: decrease cash, increase position
                        cost = order.quantity * order.price
                        current_portfolio['cash'] -= cost
                        
                        if order.symbol not in current_portfolio['positions']:
                            current_portfolio['positions'][order.symbol] = 0
                        current_portfolio['positions'][order.symbol] += order.quantity
                        
                    elif signal_side == 'SELL':
                        # Sell order: increase cash, decrease position
                        proceeds = order.quantity * order.price
                        current_portfolio['cash'] += proceeds
                        
                        if order.symbol in current_portfolio['positions']:
                            current_portfolio['positions'][order.symbol] -= order.quantity
                            if current_portfolio['positions'][order.symbol] == 0:
                                del current_portfolio['positions'][order.symbol]
            
            # Calculate current portfolio value using current market prices
            portfolio_value = current_portfolio['cash']
            for symbol, quantity in current_portfolio['positions'].items():
                if tick.symbol == symbol:  # Use current tick price for this symbol
                    portfolio_value += quantity * tick.price
                else:
                    # For other symbols, use last known price (simplified)
                    # In a real system, you'd maintain price history for all symbols
                    portfolio_value += quantity * 100  # Placeholder
            
            # Store portfolio snapshot
            snapshot = {
                'timestamp': tick.timestamp,
                'total_value': portfolio_value,
                'cash': current_portfolio['cash'],
                'positions': current_portfolio['positions'].copy(),
                'market_price': {tick.symbol: tick.price}
            }
            portfolio_snapshots.append(snapshot)
        
        self.portfolio_history = portfolio_snapshots
        logger.info(f"Generated {len(portfolio_snapshots)} portfolio snapshots")
        return portfolio_snapshots
    
    def calculate_returns(self) -> List[float]:
        """
        Calculate periodic returns from portfolio history.
        
        Returns:
            List of periodic returns
        """
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for return calculation")
            return []
        
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1]['total_value']
            curr_value = self.portfolio_history[i]['total_value']
            
            if prev_value > 0:
                period_return = (curr_value - prev_value) / prev_value
                returns.append(period_return)
            else:
                returns.append(0.0)
        
        self.returns_series = returns
        logger.info(f"Calculated {len(returns)} periodic returns")
        return returns
    
    def calculate_total_return(self) -> float:
        """Calculate total return over the entire backtest period."""
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if not self.portfolio_history:
            return 0.0
        
        final_value = self.portfolio_history[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")
        return total_return
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        if not self.returns_series:
            self.calculate_returns()
        
        if not self.returns_series or len(self.returns_series) < 2:
            logger.warning("Insufficient returns data for Sharpe ratio calculation")
            return 0.0
        
        # Convert annual risk-free rate to period rate
        # Assuming daily periods (adjust based on your data frequency)
        periods_per_year = 252  # Trading days
        period_risk_free_rate = risk_free_rate / periods_per_year
        
        # Calculate excess returns
        excess_returns = [r - period_risk_free_rate for r in self.returns_series]
        
        if not excess_returns:
            return 0.0
        
        mean_excess_return = statistics.mean(excess_returns)
        
        if len(excess_returns) < 2:
            return 0.0
        
        std_excess_return = statistics.stdev(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        # Annualize the Sharpe ratio
        sharpe_ratio = (mean_excess_return / std_excess_return) * math.sqrt(periods_per_year)
        
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio
    
    def calculate_max_drawdown(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate maximum drawdown and related statistics.
        
        Returns:
            Tuple of (max_drawdown_pct, drawdown_info)
        """
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for drawdown calculation")
            return 0.0, {}
        
        values = [snapshot['total_value'] for snapshot in self.portfolio_history]
        
        max_drawdown = 0.0
        peak_value = values[0]
        peak_index = 0
        trough_index = 0
        drawdown_start = self.portfolio_history[0]['timestamp']
        drawdown_end = self.portfolio_history[0]['timestamp']
        
        for i, value in enumerate(values):
            if value > peak_value:
                peak_value = value
                peak_index = i
            
            drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                trough_index = i
                drawdown_start = self.portfolio_history[peak_index]['timestamp']
                drawdown_end = self.portfolio_history[i]['timestamp']
        
        drawdown_info = {
            'max_drawdown_pct': max_drawdown,
            'peak_value': peak_value,
            'trough_value': values[trough_index] if trough_index < len(values) else values[-1],
            'drawdown_start': drawdown_start,
            'drawdown_end': drawdown_end,
            'duration_days': (drawdown_end - drawdown_start).days if drawdown_end > drawdown_start else 0
        }
        
        logger.info(f"Maximum drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        return max_drawdown, drawdown_info
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all performance metrics and store in metrics dict."""
        logger.info("Calculating comprehensive performance metrics...")
        
        # Basic statistics
        total_signals = len(self.engine.signals)
        total_orders = len(self.engine.orders)
        successful_orders = sum(1 for order in self.engine.orders if order.status == "FILLED")
        failed_orders = total_orders - successful_orders
        
        # Strategy-specific analysis
        strategy_stats = self._analyze_strategy_performance()
        
        # Performance metrics
        total_return = self.calculate_total_return()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, drawdown_info = self.calculate_max_drawdown()
        
        # Portfolio analysis
        final_portfolio_value = self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital
        
        # Return statistics
        if self.returns_series:
            avg_return = statistics.mean(self.returns_series)
            return_volatility = statistics.stdev(self.returns_series) if len(self.returns_series) > 1 else 0
            best_return = max(self.returns_series)
            worst_return = min(self.returns_series)
        else:
            avg_return = return_volatility = best_return = worst_return = 0
        
        # Win/Loss analysis for trades
        winning_trades = 0
        losing_trades = 0
        for i in range(1, len(self.returns_series)):
            if self.returns_series[i] > 0:
                winning_trades += 1
            elif self.returns_series[i] < 0:
                losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        self.metrics = {
            'execution_summary': {
                'total_signals': total_signals,
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'failed_orders': failed_orders,
                'success_rate': (successful_orders / total_orders * 100) if total_orders > 0 else 0
            },
            'portfolio_performance': {
                'initial_capital': self.initial_capital,
                'final_value': final_portfolio_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'profit_loss': final_portfolio_value - self.initial_capital
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'volatility': return_volatility,
                'best_period_return': best_return,
                'worst_period_return': worst_return
            },
            'trading_statistics': {
                'total_periods': len(self.returns_series),
                'winning_periods': winning_trades,
                'losing_periods': losing_trades,
                'win_rate': win_rate * 100,
                'average_return': avg_return,
                'average_return_pct': avg_return * 100
            },
            'strategy_analysis': strategy_stats,
            'drawdown_analysis': drawdown_info,
            'positions': self.engine.positions,
            'time_period': {
                'start': self.engine._market_data[0].timestamp.isoformat() if self.engine._market_data else None,
                'end': self.engine._market_data[-1].timestamp.isoformat() if self.engine._market_data else None,
                'total_ticks': len(self.engine._market_data)
            }
        }
        
        logger.info("Performance metrics calculation completed")
        return self.metrics
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy type."""
        strategy_stats = {}
        
        # Group signals and orders by strategy
        signals_by_strategy = {}
        orders_by_strategy = {}
        
        for signal in self.engine.signals:
            strategy_name = getattr(signal, 'strategy', 'Unknown')
            if strategy_name not in signals_by_strategy:
                signals_by_strategy[strategy_name] = []
            signals_by_strategy[strategy_name].append(signal)
        
        for order in self.engine.orders:
            # Try to infer strategy from order characteristics or add strategy tracking
            strategy_name = getattr(order, 'strategy', 'Unknown')
            if strategy_name not in orders_by_strategy:
                orders_by_strategy[strategy_name] = []
            orders_by_strategy[strategy_name].append(order)
        
        # Calculate stats for each strategy
        for strategy_name in signals_by_strategy.keys():
            signals = signals_by_strategy.get(strategy_name, [])
            orders = orders_by_strategy.get(strategy_name, [])
            
            successful_orders = sum(1 for order in orders if order.status == "FILLED")
            failed_orders = len(orders) - successful_orders
            
            strategy_stats[strategy_name] = {
                'signals_generated': len(signals),
                'orders_placed': len(orders),
                'successful_orders': successful_orders,
                'failed_orders': failed_orders,
                'success_rate': (successful_orders / len(orders) * 100) if orders else 0
            }
        
        return strategy_stats
    
    def generate_equity_curve_chart(self, filename: str = "equity_curve.png") -> str:
        """
        Generate professional equity curve chart using matplotlib.
        
        Args:
            filename: Output filename for the chart
            
        Returns:
            Path to generated chart file
        """
        if not self.portfolio_history:
            logger.warning("No portfolio data available for equity curve chart")
            return ""
        
        if len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for equity curve chart")
            return ""
        
        # Extract data
        timestamps = [snapshot['timestamp'] for snapshot in self.portfolio_history]
        values = [snapshot['total_value'] for snapshot in self.portfolio_history]
        
        # Create the chart
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot equity curve
        ax.plot(timestamps, values, linewidth=2, color='#2E86AB', alpha=0.8)
        ax.fill_between(timestamps, values, alpha=0.3, color='#2E86AB')
        
        # Formatting
        ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis dates
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            if time_span.total_seconds() < 3600:  # Less than 1 hour
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            elif time_span.days < 1:  # Less than 1 day
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add performance metrics as text
        initial_value = values[0]
        final_value = values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Add text box with key metrics
        textstr = f'Initial: ${initial_value:,.0f}\nFinal: ${final_value:,.0f}\nReturn: {total_return:+.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(os.getcwd(), filename)
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close to free memory
        
        logger.info(f"Equity curve chart generated: {filename}")
        return filename
    
    def generate_markdown_report(self, filename: str = "performance.md") -> str:
        """
        Generate comprehensive performance report in Markdown format.
        
        Args:
            filename: Output filename for the report
            
        Returns:
            Path to generated report file
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        logger.info(f"Generating performance report: {filename}")
        
        # Generate equity curve chart
        chart_filename = "equity_curve.png"
        self.generate_equity_curve_chart(chart_filename)
        
        # Generate report content
        report_content = self._create_markdown_content(chart_filename)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Performance report generated: {filename}")
        return filename
    
    def _create_markdown_content(self, chart_filename: str = "") -> str:
        """Create the markdown content for the performance report."""
        execution = self.metrics['execution_summary']
        portfolio = self.metrics['portfolio_performance']
        risk = self.metrics['risk_metrics']
        trading = self.metrics['trading_statistics']
        strategy_analysis = self.metrics['strategy_analysis']
        drawdown = self.metrics['drawdown_analysis']
        time_info = self.metrics['time_period']
        
        content = f"""# Trading Strategy Performance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the performance of the algorithmic trading strategy over the period from {time_info['start']} to {time_info['end']}.

**Key Highlights:**
- **Total Return**: {portfolio['total_return_pct']:+.2f}%
- **Sharpe Ratio**: {risk['sharpe_ratio']:.3f}
- **Maximum Drawdown**: {risk['max_drawdown_pct']:.2f}%
- **Win Rate**: {trading['win_rate']:.1f}%

## Portfolio Performance

| Metric | Value |
|--------|-------|
| Initial Capital | ${portfolio['initial_capital']:,.2f} |
| Final Portfolio Value | ${portfolio['final_value']:,.2f} |
| Total Return | {portfolio['total_return_pct']:+.2f}% |
| Absolute Profit/Loss | ${portfolio['profit_loss']:+,.2f} |

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {risk['sharpe_ratio']:.3f} |
| Maximum Drawdown | {risk['max_drawdown_pct']:.2f}% |
| Volatility (Daily) | {risk['volatility']*100:.2f}% |
| Best Period Return | {risk['best_period_return']*100:+.2f}% |
| Worst Period Return | {risk['worst_period_return']*100:+.2f}% |

## Trading Statistics

| Metric | Value |
|--------|-------|
| Total Signals Generated | {execution['total_signals']} |
| Total Orders Placed | {execution['total_orders']} |
| Successful Orders | {execution['successful_orders']} |
| Failed Orders | {execution['failed_orders']} |
| Order Success Rate | {execution['success_rate']:.1f}% |
| Total Trading Periods | {trading['total_periods']} |
| Winning Periods | {trading['winning_periods']} |
| Losing Periods | {trading['losing_periods']} |
| Win Rate | {trading['win_rate']:.1f}% |
| Average Period Return | {trading['average_return_pct']:+.3f}% |

## Strategy Analysis

"""

        # Add strategy-specific analysis
        if strategy_analysis:
            content += "| Strategy | Signals | Orders | Success Rate | Failed Orders |\n"
            content += "|----------|---------|--------|--------------|---------------|\n"
            for strategy_name, stats in strategy_analysis.items():
                content += f"| {strategy_name} | {stats['signals_generated']} | {stats['orders_placed']} | {stats['success_rate']:.1f}% | {stats['failed_orders']} |\n"
        else:
            content += "*Note: Strategy-specific analysis requires enhanced signal tracking.*\n"

        content += f"""

## Drawdown Analysis

| Metric | Value |
|--------|-------|
| Maximum Drawdown | {risk['max_drawdown_pct']:.2f}% |
| Peak Portfolio Value | ${drawdown.get('peak_value', 0):,.2f} |
| Trough Portfolio Value | ${drawdown.get('trough_value', 0):,.2f} |
| Drawdown Start | {drawdown.get('drawdown_start', 'N/A')} |
| Drawdown End | {drawdown.get('drawdown_end', 'N/A')} |
| Drawdown Duration | {drawdown.get('duration_days', 0)} days |

## Current Positions

"""

        # Add current positions
        positions = self.metrics['positions']
        if positions:
            content += "| Symbol | Quantity | Average Price | Market Value |\n"
            content += "|--------|----------|---------------|-------------|\n"
            for symbol, pos in positions.items():
                if pos['quantity'] != 0:
                    market_value = pos['quantity'] * pos['avg_price']
                    content += f"| {symbol} | {pos['quantity']} | ${pos['avg_price']:.2f} | ${market_value:,.2f} |\n"
        else:
            content += "No open positions.\n"

        content += f"""

## Equity Curve

![Portfolio Equity Curve]({chart_filename})

*Figure: Portfolio value evolution over time showing the cumulative performance of the trading strategy.*

## Performance Interpretation

### Overall Assessment
"""

        # Add narrative interpretation
        if portfolio['total_return'] > 0:
            content += f"The strategy generated a positive return of {portfolio['total_return_pct']:.2f}% over the testing period, "
        else:
            content += f"The strategy experienced a negative return of {portfolio['total_return_pct']:.2f}% over the testing period, "

        if risk['sharpe_ratio'] > 1.0:
            content += "with an excellent risk-adjusted return as indicated by a Sharpe ratio above 1.0."
        elif risk['sharpe_ratio'] > 0.5:
            content += "with a reasonable risk-adjusted return as indicated by a moderate Sharpe ratio."
        else:
            content += "with poor risk-adjusted returns as indicated by a low Sharpe ratio."

        content += f"""

### Risk Assessment
The strategy experienced a maximum drawdown of {risk['max_drawdown_pct']:.2f}%, which represents the largest peak-to-trough decline in portfolio value. """

        if risk['max_drawdown_pct'] < 10:
            content += "This relatively low drawdown suggests good risk management."
        elif risk['max_drawdown_pct'] < 20:
            content += "This moderate drawdown is within acceptable ranges for most strategies."
        else:
            content += "This high drawdown indicates significant risk and potential for large losses."

        content += f"""

### Trading Effectiveness
The strategy achieved a win rate of {trading['win_rate']:.1f}% with an order execution success rate of {execution['success_rate']:.1f}%. """

        if trading['win_rate'] > 60:
            content += "The high win rate suggests the strategy is effective at identifying profitable opportunities."
        elif trading['win_rate'] > 50:
            content += "The win rate is above break-even, indicating the strategy has positive edge."
        else:
            content += "The win rate below 50% suggests the strategy may need refinement, though this could be acceptable if winning trades are larger than losing ones."

        content += f"""

### Recommendations
Based on the analysis:

1. **Risk Management**: """
        
        if risk['max_drawdown_pct'] > 15:
            content += "Consider implementing stricter risk controls to reduce drawdown."
        else:
            content += "Current risk levels appear manageable."

        content += f"""
2. **Strategy Optimization**: """
        
        if trading['win_rate'] < 50:
            content += "Focus on improving trade selection criteria to increase win rate."
        else:
            content += "The strategy shows good trade selection. Consider position sizing optimization."

        content += f"""
3. **Performance**: """
        
        if portfolio['total_return'] > 0 and risk['sharpe_ratio'] > 0.5:
            content += "The strategy shows promise and may be suitable for live trading with proper risk management."
        else:
            content += "The strategy requires further development before considering live implementation."

        content += f"""

---

*This report was generated automatically by the CSV-Based Algorithmic Trading Backtester.*
*Past performance does not guarantee future results.*
"""

        return content

    def print_summary(self):
        """Print a brief performance summary to console."""
        if not self.metrics:
            self.calculate_all_metrics()
        
        portfolio = self.metrics['portfolio_performance']
        risk = self.metrics['risk_metrics']
        trading = self.metrics['trading_statistics']
        
        print("\\n" + "="*60)
        print("           PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital:     ${portfolio['initial_capital']:,.2f}")
        print(f"Final Value:         ${portfolio['final_value']:,.2f}")
        print(f"Total Return:        {portfolio['total_return_pct']:+.2f}%")
        print(f"Sharpe Ratio:        {risk['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {risk['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:            {trading['win_rate']:.1f}%")
        print("="*60)


def generate_strategy_performance_plots(engine: ExecutionEngine, strategies: List):
    """
    Generate PNG plots showing holdings/portfolio value over time for all strategies.
    
    Args:
        engine: ExecutionEngine instance with completed backtest
        strategies: List of strategy instances used in the backtest
    """
    print("\n📊 Generating strategy performance plots...")
    
    if not engine._market_data:
        print("❌ No market data available for plotting")
        return
    
    # Group strategies by symbol and type for easier identification
    strategy_groups = {}
    for strategy in strategies:
        symbol = strategy._symbol
        strategy_type = type(strategy).__name__
        key = f"{symbol}_{strategy_type}"
        strategy_groups[key] = strategy
    
    # Get all unique timestamps from market data
    timestamps = sorted(set(tick.timestamp for tick in engine._market_data))
    initial_capital_per_strategy = engine.initial_capital / len(strategies)
    
    # Use the engine's strategy signals tracking to get better strategy performance data
    strategy_portfolios = {}
    
    # Initialize portfolio tracking for each strategy using strategy names from engine
    for strategy_name in engine._strategy_signals.keys():
        strategy_portfolios[strategy_name] = {
            'timestamps': [],
            'values': [],
            'cash': initial_capital_per_strategy,
            'positions': {},
            'trades': []
        }
    
    # Process signals chronologically to reconstruct portfolio evolution
    for timestamp in timestamps:
        # Get all signals and orders for this timestamp
        timestamp_signals = [s for s in engine.signals if s.timestamp == timestamp]
        timestamp_orders = [o for o in engine.orders if hasattr(o, '_signal_timestamp') and o._signal_timestamp == timestamp]
        
        # Process each strategy's activity
        for strategy_name, signals in engine._strategy_signals.items():
            portfolio = strategy_portfolios[strategy_name]
            
            # Process signals for this strategy at this timestamp
            strategy_signals = [s for s in timestamp_signals if s.strategy == strategy_name.split('_')[-1]]
            
            for signal in strategy_signals:
                # Find corresponding order
                corresponding_orders = [
                    order for order in timestamp_orders 
                    if order.symbol == signal.symbol and 
                       order.quantity == signal.qty and
                       order.status == "FILLED"
                ]
                
                for order in corresponding_orders:
                    if signal.side == 'BUY':
                        cost = order.quantity * order.price
                        portfolio['cash'] -= cost
                        if signal.symbol not in portfolio['positions']:
                            portfolio['positions'][signal.symbol] = 0
                        portfolio['positions'][signal.symbol] += order.quantity
                        portfolio['trades'].append(('BUY', timestamp, order.quantity, order.price))
                    
                    elif signal.side == 'SELL':
                        proceeds = order.quantity * order.price
                        portfolio['cash'] += proceeds
                        if signal.symbol in portfolio['positions']:
                            portfolio['positions'][signal.symbol] -= order.quantity
                            if portfolio['positions'][signal.symbol] <= 0:
                                del portfolio['positions'][signal.symbol]
                        portfolio['trades'].append(('SELL', timestamp, order.quantity, order.price))
            
            # Calculate current portfolio value using latest market prices
            current_value = portfolio['cash']
            for symbol, quantity in portfolio['positions'].items():
                # Get latest price for this symbol at this timestamp
                latest_price = None
                for tick in engine._market_data:
                    if tick.symbol == symbol and tick.timestamp <= timestamp:
                        latest_price = tick.price
                if latest_price:
                    current_value += quantity * latest_price
            
            # Store snapshot
            portfolio['timestamps'].append(timestamp)
            portfolio['values'].append(current_value)
    
    # Create individual plots for each strategy
    plot_count = 0
    for strategy_name, portfolio in strategy_portfolios.items():
        if len(portfolio['timestamps']) < 2:
            continue
            
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio value over time
        plt.subplot(2, 1, 1)
        plt.plot(portfolio['timestamps'], portfolio['values'], linewidth=2, label='Portfolio Value', color='blue')
        plt.axhline(y=initial_capital_per_strategy, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.title(f'{strategy_name} - Portfolio Value Over Time', fontsize=14, fontweight='bold')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add trade markers
        for trade_type, trade_timestamp, quantity, price in portfolio['trades']:
            try:
                # Find index of this timestamp in portfolio timestamps
                if trade_timestamp in portfolio['timestamps']:
                    idx = portfolio['timestamps'].index(trade_timestamp)
                    color = 'green' if trade_type == 'BUY' else 'red'
                    marker = '^' if trade_type == 'BUY' else 'v'
                    plt.scatter(trade_timestamp, portfolio['values'][idx], 
                               color=color, marker=marker, s=50, alpha=0.7, zorder=5)
            except (ValueError, IndexError):
                continue  # Skip if timestamp not found
        
        # Plot cash vs position values over time
        plt.subplot(2, 1, 2)
        
        # Calculate position values over time
        position_values = []
        cash_values = []
        
        for i, timestamp in enumerate(portfolio['timestamps']):
            # Calculate position value at this timestamp
            pos_value = 0
            for symbol, quantity in portfolio['positions'].items():
                # Get price at this timestamp
                price_at_timestamp = None
                for tick in engine._market_data:
                    if tick.symbol == symbol and tick.timestamp <= timestamp:
                        price_at_timestamp = tick.price
                if price_at_timestamp:
                    pos_value += quantity * price_at_timestamp
            
            position_values.append(pos_value)
            cash_values.append(portfolio['cash'])  # Note: this is simplified, cash changes with trades
        
        plt.plot(portfolio['timestamps'], position_values, linewidth=2, color='orange', label='Position Value')
        plt.plot(portfolio['timestamps'], cash_values, linewidth=2, color='green', label='Available Cash')
        plt.title(f'{strategy_name} - Cash vs Position Value', fontsize=14, fontweight='bold')
        plt.ylabel('Value ($)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis
        plt.xticks(rotation=45)
        
        # Save plot
        filename = f"strategy_performance_{strategy_name}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  ✅ Generated: {filename}")
        plot_count += 1
    
    # Create summary plot with all strategies
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (strategy_name, portfolio) in enumerate(strategy_portfolios.items()):
        if len(portfolio['timestamps']) < 2:
            continue
        color = colors[i % len(colors)]
        
        # Calculate total return percentage for legend
        initial_value = portfolio['values'][0] if portfolio['values'] else initial_capital_per_strategy
        final_value = portfolio['values'][-1] if portfolio['values'] else initial_capital_per_strategy
        return_pct = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
        
        label = f'{strategy_name} ({return_pct:+.2f}%)'
        plt.plot(portfolio['timestamps'], portfolio['values'], 
                linewidth=2, label=label, color=color, alpha=0.8)
    
    plt.axhline(y=initial_capital_per_strategy, color='black', linestyle='--', 
               alpha=0.7, label=f'Initial Capital (${initial_capital_per_strategy:,.0f})')
    plt.title('All Strategies Performance Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Save summary plot
    summary_filename = "all_strategies_performance.png"
    plt.tight_layout()
    plt.savefig(summary_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✅ Generated: {summary_filename}")
    print(f"📊 Generated {plot_count + 1} performance plots total")


def print_backtest_results(engine: ExecutionEngine, strategies: List):
    """
    Print comprehensive backtest results to console and generate performance plots.
    
    Args:
        engine: ExecutionEngine instance with completed backtest
        strategies: List of strategy instances used in the backtest
    """
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Print capital summary
    capital_summary = engine.get_capital_summary()
    print(f"\nInitial Capital: ${capital_summary['initial_capital']:,.2f}")
    print(f"Current Portfolio Value: ${capital_summary['current_portfolio_value']:,.2f}")
    print(f"Total Return: ${capital_summary['current_portfolio_value'] - capital_summary['initial_capital']:,.2f}")
    
    print(f"\nStrategy Performance:")
    for strategy_name, total_value in capital_summary['strategies_current'].items():
        initial_per_strategy = capital_summary['initial_capital'] / len(strategies)
        pnl = total_value - initial_per_strategy
        pnl_pct = (pnl / initial_per_strategy) * 100
        print(f"  {strategy_name}: ${total_value:,.2f} (P&L: ${pnl:,.2f}, {pnl_pct:+.2f}%)")
    
    # Print basic stats
    print(f"\nTrade Statistics:")
    print(f"  Total Signals: {len(engine.signals)}")
    print(f"  Total Orders: {len(engine.orders)}")
    print(f"  Filled Orders: {sum(1 for order in engine.orders if order.status == 'FILLED')}")
    print(f"  Failed Orders: {sum(1 for order in engine.orders if order.status == 'FAILED')}")
    
    # Print positions
    print(f"\nFinal Positions:")
    for symbol, position in engine.positions.items():
        if position['quantity'] != 0:
            value = position['quantity'] * position['avg_price']
            print(f"  {symbol}: {position['quantity']} shares @ ${position['avg_price']:.2f} = ${value:,.2f}")
    
    # Generate performance plots
    generate_strategy_performance_plots(engine, strategies)
    
    # Generate comprehensive markdown report
    generate_performance_report(engine, strategies)
    
    print("\n✅ Backtest completed!")


def generate_performance_report(engine: ExecutionEngine, strategies: List):
    """
    Generate a comprehensive performance.md report with metrics tables, equity curves, and narrative.
    
    Args:
        engine: ExecutionEngine instance with completed backtest
        strategies: List of strategy instances used in the backtest
    """
    print("\n📊 Generating comprehensive performance report...")
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(engine, engine.initial_capital)
    analyzer.calculate_portfolio_history()
    analyzer.calculate_returns()
    metrics = analyzer.calculate_all_metrics()
    
    # Generate equity curve chart for all strategies
    equity_chart_filename = "all_strategies_equity_curve.png"
    generate_multi_strategy_equity_curve(engine, strategies, equity_chart_filename)
    
    # Create the markdown report
    report_content = create_comprehensive_markdown_report(engine, strategies, metrics, equity_chart_filename)
    
    # Write to file
    report_filename = "performance.md"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"  ✅ Generated: {report_filename}")
    print(f"  ✅ Generated: {equity_chart_filename}")


def generate_multi_strategy_equity_curve(engine: ExecutionEngine, strategies: List, filename: str):
    """
    Generate an equity curve plot showing all strategies on the same chart.
    
    Args:
        engine: ExecutionEngine instance with completed backtest
        strategies: List of strategy instances used in the backtest
        filename: Output filename for the chart
    """
    if not engine.capital_history:
        print("❌ No capital history available for equity curve")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Extract timestamps and values for each strategy
    timestamps = []
    strategy_values = {}
    
    # Initialize strategy tracking
    for strategy in strategies:
        strategy_name = f"{strategy.__class__.__name__}_{strategy._symbol}"
        strategy_values[strategy_name] = []
    
    # Process capital history to extract strategy performance over time
    for snapshot in engine.capital_history:
        if snapshot.get("strategies"):
            for strategy_name in strategy_values.keys():
                if strategy_name in snapshot["strategies"]:
                    # Handle both old and new formats
                    strategy_data = snapshot["strategies"][strategy_name]
                    if isinstance(strategy_data, dict) and "total" in strategy_data:
                        strategy_values[strategy_name].append(strategy_data["total"])
                    elif isinstance(strategy_data, (int, float)):
                        # Old format - just a number
                        strategy_values[strategy_name].append(strategy_data)
                    else:
                        # Fallback - use last known value
                        if strategy_values[strategy_name]:
                            strategy_values[strategy_name].append(strategy_values[strategy_name][-1])
                        else:
                            initial_capital = engine.initial_capital / len(strategies)
                            strategy_values[strategy_name].append(initial_capital)
                else:
                    # If no data for this snapshot, use last known value or initial
                    if strategy_values[strategy_name]:
                        strategy_values[strategy_name].append(strategy_values[strategy_name][-1])
                    else:
                        initial_capital = engine.initial_capital / len(strategies)
                        strategy_values[strategy_name].append(initial_capital)
    
    # Create timestamps (use index if no actual timestamps available)
    if len(strategy_values) > 0:
        first_strategy = list(strategy_values.keys())[0]
        timestamps = list(range(len(strategy_values[first_strategy])))
    
    # Plot each strategy
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#8E44AD']
    
    for i, (strategy_name, values) in enumerate(strategy_values.items()):
        if len(values) > 1:
            color = colors[i % len(colors)]
            
            # Calculate total return for legend
            initial_value = values[0] if values else engine.initial_capital / len(strategies)
            final_value = values[-1] if values else initial_value
            total_return = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
            
            plt.plot(timestamps, values, linewidth=2.5, color=color, alpha=0.8,
                    label=f'{strategy_name} ({total_return:+.1f}%)')
    
    # Add initial capital reference line
    initial_capital_per_strategy = engine.initial_capital / len(strategies)
    plt.axhline(y=initial_capital_per_strategy, color='black', linestyle='--', 
               alpha=0.7, linewidth=1, label=f'Initial Capital (${initial_capital_per_strategy:,.0f})')
    
    # Formatting
    plt.title('Strategy Performance - Equity Curves Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (Trades)', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Save the chart
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_comprehensive_markdown_report(engine: ExecutionEngine, strategies: List, metrics: Dict, equity_chart_filename: str) -> str:
    """
    Create comprehensive markdown content for the performance report.
    
    Args:
        engine: ExecutionEngine instance with completed backtest
        strategies: List of strategy instances used in the backtest
        metrics: Performance metrics dictionary
        equity_chart_filename: Filename of the equity curve chart
        
    Returns:
        Markdown content string
    """
    # Get current timestamp
    report_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract key metrics
    portfolio = metrics['portfolio_performance']
    risk = metrics['risk_metrics']
    trading = metrics['trading_statistics']
    execution = metrics['execution_summary']
    
    # Calculate per-strategy metrics
    strategy_metrics = calculate_per_strategy_metrics(engine, strategies)
    
    # Create periodic returns table
    analyzer = PerformanceAnalyzer(engine, engine.initial_capital)
    analyzer.calculate_portfolio_history()
    returns_series = analyzer.calculate_returns()
    
    content = f"""# Trading Strategy Performance Report

**Generated:** {report_timestamp}  
**Backtest Period:** {metrics['time_period']['start']} to {metrics['time_period']['end']}  
**Total Market Ticks:** {metrics['time_period']['total_ticks']:,}

---

## Executive Summary

This report analyzes the performance of {len(strategies)} algorithmic trading strategies over the backtest period.

**Key Performance Highlights:**
- **Total Portfolio Return:** {portfolio['total_return_pct']:+.2f}%
- **Absolute Profit/Loss:** ${portfolio['profit_loss']:+,.2f}
- **Risk-Adjusted Return (Sharpe):** {risk['sharpe_ratio']:.3f}
- **Maximum Drawdown:** {risk['max_drawdown_pct']:.2f}%
- **Win Rate:** {trading['win_rate']:.1f}%

---

## Portfolio Performance Metrics

| Metric | Value |
|--------|--------|
| **Initial Capital** | ${portfolio['initial_capital']:,.2f} |
| **Final Portfolio Value** | ${portfolio['final_value']:,.2f} |
| **Total Return** | {portfolio['total_return_pct']:+.2f}% |
| **Absolute P&L** | ${portfolio['profit_loss']:+,.2f} |
| **Average Daily Return** | {trading['average_return_pct']:+.3f}% |

---

## Risk Analysis

| Metric | Value | Interpretation |
|--------|--------|----------------|
| **Sharpe Ratio** | {risk['sharpe_ratio']:.3f} | {'Excellent' if risk['sharpe_ratio'] > 1.0 else 'Good' if risk['sharpe_ratio'] > 0.5 else 'Poor'} risk-adjusted performance |
| **Maximum Drawdown** | {risk['max_drawdown_pct']:.2f}% | {'Low' if risk['max_drawdown_pct'] < 10 else 'Moderate' if risk['max_drawdown_pct'] < 20 else 'High'} maximum loss from peak |
| **Volatility (Daily)** | {risk['volatility']*100:.2f}% | Daily portfolio volatility |
| **Best Period Return** | {risk['best_period_return']*100:+.2f}% | Single best trading period |
| **Worst Period Return** | {risk['worst_period_return']*100:+.2f}% | Single worst trading period |

---

## Strategy Performance Breakdown

| Strategy | Initial Capital | Final Value | Return | P&L |
|----------|----------------|-------------|--------|-----|"""

    # Add strategy-specific rows
    for strategy_name, metrics_data in strategy_metrics.items():
        initial = metrics_data['initial_capital']
        final = metrics_data['final_value']
        return_pct = metrics_data['return_pct']
        pnl = metrics_data['pnl']
        content += f"\n| **{strategy_name}** | ${initial:,.2f} | ${final:,.2f} | {return_pct:+.2f}% | ${pnl:+,.2f} |"

    content += f"""

---

## Trading Statistics

| Metric | Value |
|--------|--------|
| **Total Signals Generated** | {execution['total_signals']} |
| **Total Orders Placed** | {execution['total_orders']} |
| **Successful Orders** | {execution['successful_orders']} |
| **Failed Orders** | {execution['failed_orders']} |
| **Order Success Rate** | {execution['success_rate']:.1f}% |
| **Total Trading Periods** | {trading['total_periods']} |
| **Winning Periods** | {trading['winning_periods']} |
| **Losing Periods** | {trading['losing_periods']} |
| **Win Rate** | {trading['win_rate']:.1f}% |

---

## Periodic Returns Analysis

**Sample of Recent Returns** (Last 10 periods):
"""

    # Add periodic returns table
    if len(returns_series) > 0:
        content += "\n| Period | Return |\n|--------|--------|\n"
        # Show last 10 returns
        recent_returns = returns_series[-10:] if len(returns_series) >= 10 else returns_series
        for i, ret in enumerate(recent_returns, 1):
            content += f"| Period {len(returns_series) - len(recent_returns) + i} | {ret*100:+.3f}% |\n"
        
        # Add summary statistics
        avg_return = sum(returns_series) / len(returns_series) * 100
        positive_periods = sum(1 for r in returns_series if r > 0)
        content += f"""
**Returns Summary:**
- Average Period Return: {avg_return:+.3f}%
- Positive Periods: {positive_periods}/{len(returns_series)} ({positive_periods/len(returns_series)*100:.1f}%)
- Total Periods Analyzed: {len(returns_series)}
"""
    else:
        content += "\n*No periodic returns data available.*\n"

    content += f"""

---

## Equity Curve Analysis

![Strategy Performance Comparison]({equity_chart_filename})

*Figure: Portfolio value evolution over time showing the performance of each strategy.*

---

## Performance Interpretation

### Overall Assessment
"""

    # Add performance interpretation
    if portfolio['total_return'] > 0:
        content += f"The portfolio generated a **positive return of {portfolio['total_return_pct']:+.2f}%**, demonstrating profitable performance over the backtest period. "
    else:
        content += f"The portfolio experienced a **negative return of {portfolio['total_return_pct']:+.2f}%**, indicating losses during the backtest period. "

    # Risk assessment
    if risk['sharpe_ratio'] > 1.0:
        content += f"The Sharpe ratio of {risk['sharpe_ratio']:.3f} indicates **excellent risk-adjusted performance**. "
    elif risk['sharpe_ratio'] > 0.5:
        content += f"The Sharpe ratio of {risk['sharpe_ratio']:.3f} shows **acceptable risk-adjusted returns**. "
    else:
        content += f"The Sharpe ratio of {risk['sharpe_ratio']:.3f} suggests **poor risk-adjusted performance**. "

    content += f"""

### Risk Profile
The strategy experienced a maximum drawdown of **{risk['max_drawdown_pct']:.2f}%**, representing the largest peak-to-trough decline. """

    if risk['max_drawdown_pct'] < 10:
        content += "This is considered a **low-risk** drawdown level."
    elif risk['max_drawdown_pct'] < 20:
        content += "This represents a **moderate risk** level that is acceptable for most strategies."
    else:
        content += "This is a **high-risk** drawdown that may require strategy refinement."

    content += f"""

### Trading Effectiveness
- **Win Rate:** {trading['win_rate']:.1f}% of trading periods were profitable
- **Execution Quality:** {execution['success_rate']:.1f}% order success rate
- **Strategy Diversity:** {len(strategies)} different strategies deployed

"""

    # Strategy-specific insights
    best_strategy = max(strategy_metrics.items(), key=lambda x: x[1]['return_pct'])
    worst_strategy = min(strategy_metrics.items(), key=lambda x: x[1]['return_pct'])
    
    content += f"""### Strategy Analysis
- **Top Performer:** {best_strategy[0]} with {best_strategy[1]['return_pct']:+.2f}% return
- **Underperformer:** {worst_strategy[0]} with {worst_strategy[1]['return_pct']:+.2f}% return
- **Portfolio Effect:** Diversification across strategies {'helped reduce' if len([s for s in strategy_metrics.values() if s['return_pct'] > 0]) > len([s for s in strategy_metrics.values() if s['return_pct'] < 0]) else 'did not prevent'} overall losses

---

## Recommendations

### 1. Strategy Optimization
"""
    
    if trading['win_rate'] < 50:
        content += "- **Improve Win Rate:** Current win rate is below 50%. Consider refining entry/exit criteria."
    else:
        content += "- **Maintain Win Rate:** Good win rate above 50%. Focus on preserving current strategy effectiveness."

    content += f"""

### 2. Risk Management
"""
    
    if risk['max_drawdown_pct'] > 15:
        content += "- **Reduce Drawdown:** Implement tighter stop-losses or position sizing to limit maximum drawdown."
    else:
        content += "- **Risk Control:** Current drawdown levels are acceptable. Maintain existing risk controls."

    content += f"""

### 3. Performance Enhancement
"""
    
    if portfolio['total_return'] > 0 and risk['sharpe_ratio'] > 0.5:
        content += "- **Scale Strategy:** Consider increasing position sizes or capital allocation given positive risk-adjusted returns."
    else:
        content += "- **Strategy Review:** Re-evaluate strategy parameters, market conditions, and implementation before live deployment."

    content += """

---

## Final Positions

"""

    # Add current positions
    if engine.positions:
        content += "| Symbol | Quantity | Avg Price | Current Value |\n|--------|----------|-----------|---------------|\n"
        for symbol, position in engine.positions.items():
            if position['quantity'] > 0:
                current_value = position['quantity'] * position['avg_price']
                content += f"| {symbol} | {position['quantity']} | ${position['avg_price']:.2f} | ${current_value:,.2f} |\n"
    else:
        content += "*No open positions at end of backtest.*\n"

    content += f"""

---

## Appendix

**Data Sources:**
- Market Data: CSV file with {metrics['time_period']['total_ticks']:,} price ticks
- Strategies Tested: {', '.join([f"{s.__class__.__name__}({s._symbol})" for s in strategies])}

**Risk Disclaimers:**
- Past performance does not guarantee future results
- All investments carry risk of loss
- This analysis is for educational purposes only

---

*Report generated by CSV-Based Algorithmic Trading Backtester*  
*Generation Time: {report_timestamp}*
"""

    return content


def calculate_per_strategy_metrics(engine: ExecutionEngine, strategies: List) -> Dict:
    """Calculate performance metrics for each strategy individually."""
    strategy_metrics = {}
    initial_capital_per_strategy = engine.initial_capital / len(strategies)
    
    for strategy in strategies:
        strategy_name = f"{strategy.__class__.__name__}_{strategy._symbol}"
        
        # Get final value for this strategy
        final_value = engine.get_strategy_total_holdings(strategy_name).get('total_holdings', initial_capital_per_strategy)
        
        # Calculate metrics
        pnl = final_value - initial_capital_per_strategy
        return_pct = (pnl / initial_capital_per_strategy) * 100 if initial_capital_per_strategy > 0 else 0
        
        strategy_metrics[strategy_name] = {
            'initial_capital': initial_capital_per_strategy,
            'final_value': final_value,
            'pnl': pnl,
            'return_pct': return_pct
        }
    
    return strategy_metrics