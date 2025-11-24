import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 1. 系统配置与样式
# ============================================================
st.set_page_config(
    page_title="Rolling Strategy Pro - 建控辅助系统",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px;
        margin-bottom: 10px;
    }
    .signal-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .badge-yellow { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #b45309; }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #b91c1c; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. 数据引擎
# ============================================================
class DataEngine:
    def __init__(self):
        self.exchange = ccxt.okx({'enableRateLimit': True})

    def fetch_data(self, symbol, limit_daily=1000, limit_weekly=200):
        """同时拉取日线和周线数据，用于双周期共振分析"""
        try:
            # 日线数据 (用于交易和趋势)
            d_ohlcv = self.exchange.fetch_ohlcv(symbol, '1d', limit=limit_daily)
            df_d = self._process_data(d_ohlcv)
            
            # 周线数据 (用于大周期RSI底部判断)
            w_ohlcv = self.exchange.fetch_ohlcv(symbol, '1w', limit=limit_weekly)
            df_w = self._process_data(w_ohlcv)
            
            return df_d, df_w
        except Exception as e:
            st.error(f"数据拉取失败: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def _process_data(self, ohlcv):
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

# ============================================================
# 3. 策略逻辑核心
# ============================================================
class StrategyEngine:
    def __init__(self, df_daily, df_weekly):
        self.d = df_daily.copy()
        self.w = df_weekly.copy()
        
    def prepare_indicators(self, breakout_days=90, vol_mult=3.0):
        # --- 周线指标 ---
        self.w['RSI_W'] = ta.rsi(self.w['close'], length=14)
        
        # --- 日线指标 ---
        # 1. 趋势均线
        self.d['MA20'] = ta.sma(self.d['close'], length=20)
        self.d['MA200'] = ta.sma(self.d['close'], length=200)
        
        # 2. 震荡突破 (Donchian Channel)
        self.d['High_N'] = self.d['high'].rolling(breakout_days).max().shift(1) # N日高点
        self.d['Low_N'] = self.d['low'].rolling(breakout_days).min().shift(1)
        
        # 3. 成交量异动
        self.d['Vol_MA'] = ta.sma(self.d['volume'], length=20)
        self.d['Vol_Boom'] = self.d['volume'] > (self.d['Vol_MA'] * vol_mult)
        
        # 4. 风控指标
        self.d['ATR'] = ta.atr(self.d['high'], self.d['low'], self.d['close'], length=14)
        
        # --- 融合周线数据到日线 (Forward Fill) ---
        # 为了在日线循环中能看到当时的周线RSI状态
        # 这里做一个简化处理，实际量化需要更严谨的对齐
        self.w_resampled = self.w['RSI_W'].resample('1D').ffill()
        self.d = self.d.join(self.w_resampled.rename('RSI_W_DailyMap'), how='left')
        
        return self.d.dropna()

# ============================================================
# 4. 滚仓回测模拟器 (Event Driven)
# ============================================================
@dataclass
class Position:
    entry_price: float = 0.0
    size: float = 0.0      # 币的数量
    leverage: float = 0.0  # 当前实际杠杆
    stop_loss: float = 0.0
    peak_price: float = 0.0 # 持仓期间最高价

class RollingBacktester:
    def __init__(self, initial_capital=10000.0):
        self.capital = initial_capital
        self.balance = initial_capital
        self.position = None
        self.history = [] # 资金曲线
        self.trades = []  # 交易记录
        
    def run(self, df, params):
        """
        params: {
            'rsi_bottom': 35,
            'max_leverage': 3.0,
            'add_step': 0.10,  # 每涨10%加仓
            'trail_atr': 2.0   # 2倍ATR止损
        }
        """
        for i in range(len(df)):
            bar = df.iloc[i]
            date = df.index[i]
            price = bar['close']
            
            # 更新持仓市值
            equity = self.balance
            if self.position:
                unrealized_pnl = (price - self.position.entry_price) * self.position.size
                equity = self.balance + unrealized_pnl
                # 更新最高价用于移动止损
                if price > self.position.peak_price:
                    self.position.peak_price = price
            
            self.history.append({'date': date, 'equity': equity, 'price': price})
            
            # --- 1. 离场/风控逻辑 (优先级最高) ---
            if self.position:
                # 触发硬止损 或 跌破移动止损
                # 移动止损逻辑：最高价回撤 N * ATR，或者成本保护
                trail_sl = self.position.peak_price - (params['trail_atr'] * bar['ATR'])
                # 核心规则：如果有浮盈，止损线上移至开仓均价，保本！
                if trail_sl < self.position.entry_price and (price > self.position.entry_price * 1.05):
                     trail_sl = self.position.entry_price * 1.01 # 微利保护
                
                actual_sl = max(self.position.stop_loss, trail_sl)
                
                # 跌破MA20趋势线强制离场
                trend_sl = bar['MA20']
                
                if bar['low'] < actual_sl or bar['close'] < trend_sl:
                    exit_price = min(bar['open'], actual_sl) # 简化撮合
                    pnl = (exit_price - self.position.entry_price) * self.position.size
                    self.balance += pnl
                    self.trades.append({
                        'type': 'CLOSE', 'date': date, 'price': exit_price, 
                        'pnl': pnl, 'reason': 'Stop/Trend Break'
                    })
                    self.position = None
                    continue

            # --- 2. 建仓/滚仓逻辑 ---
            
            # 信号A: 底部埋伏 (RSI < 35) -> 建立 1x 底仓
            # 条件：空仓 + 周RSI低 + 价格在MA200下方(熊市深跌)或上方(牛市回调)
            is_bottom = (bar['RSI_W_DailyMap'] < params['rsi_bottom'])
            
            if self.position is None and is_bottom:
                pos_size = (self.balance * 1.0) / price # 1x 杠杆
                self.position = Position(
                    entry_price=price, size=pos_size, leverage=1.0, 
                    stop_loss=price - 2*bar['ATR'], peak_price=price
                )
                self.trades.append({'type': 'OPEN_BASE', 'date': date, 'price': price, 'leverage': 1.0})
                continue
                
            # 信号B: 趋势突破 (Price > 90日新高 & Vol > 3倍) -> 建立/加仓
            is_breakout = (price > bar['High_N']) and bar['Vol_Boom'] and (price > bar['MA20'])
            
            # 情况1: 空仓且突破 -> 这是一个极其强烈的右侧入场点，直接上 1.5x
            if self.position is None and is_breakout:
                pos_size = (self.balance * 1.5) / price
                self.position = Position(
                    entry_price=price, size=pos_size, leverage=1.5,
                    stop_loss=price - 2*bar['ATR'], peak_price=price
                )
                self.trades.append({'type': 'OPEN_BREAKOUT', 'date': date, 'price': price, 'leverage': 1.5})
                continue
                
            # 情况2: 持仓中 -> 滚仓加仓 (Pyramiding)
            # 条件：已持仓 + (再次突破 OR 浮盈达到阈值) + 杠杆未满
            if self.position:
                current_lev = (self.position.size * price) / equity
                pnl_pct = (price - self.position.entry_price) / self.position.entry_price
                
                # 只有当浮盈 > 10% 且 杠杆 < 最大限制时，才允许加仓
                if pnl_pct > params['add_step'] and current_lev < params['max_leverage']:
                    # 加仓逻辑：利用浮盈加仓，保持风险敞口可控
                    add_amt = equity * 0.5 # 每次加本金的50%名义价值
                    add_size = add_amt / price
                    
                    # 重新计算均价
                    new_total_size = self.position.size + add_size
                    new_entry = (self.position.entry_price * self.position.size + price * add_size) / new_total_size
                    
                    self.position.size = new_total_size
                    self.position.entry_price = new_entry
                    self.position.leverage = (new_total_size * price) / equity
                    
                    # 关键：加仓后，止损必须立即上移至新均价上方一点点，防止加仓一把亏光
                    self.position.stop_loss = new_entry * 1.01 
                    
                    self.trades.append({'type': 'ROLL_ADD', 'date': date, 'price': price, 'new_lev': self.position.leverage})

        return pd.DataFrame(self.history), pd.DataFrame(self.trades)

# ============================================================
# 5. 主界面逻辑
# ============================================================
def main():
    st.title("⚔️ 滚仓猎人：趋势突破建控系统")
    st.markdown("此系统基于 **[底部RSI + 突破放量 + 情绪逆转]** 三大黄金法则，结合 **[金字塔加仓 + ATR移动止损]** 风控模型。")
    
    with st.sidebar:
        st.header("⚙️ 策略参数设置")
        symbol = st.text_input("交易对 (OKX)", "BTC/USDT")
        
        st.subheader("信号参数")
        breakout_days = st.slider("震荡突破周期 (天)", 30, 120, 90, help="黄金信号2：突破N日新高")
        rsi_bottom = st.slider("周线底部 RSI 阈值", 20, 45, 35, help="黄金信号1：周线超卖")
        vol_mult = st.slider("量能爆发倍数", 1.5, 5.0, 3.0, help="黄金信号2：成交量 > N倍均量")
        
        st.subheader("滚仓风控")
        max_lev = st.slider("最大允许杠杆", 1.0, 10.0, 3.0, help="滚仓的尽头是爆仓，建议不超过3x")
        trail_atr = st.slider("移动止损 ATR倍数", 1.0, 5.0, 2.5, help="越小离场越快，越大抗波动能力越强")

    # --- 1. 获取与处理数据 ---
    engine = DataEngine()
    with st.spinner("正在从 OKX 抓取数据并进行双周期计算..."):
        df_d, df_w = engine.fetch_data(symbol)
    
    if df_d.empty:
        return

    strategy = StrategyEngine(df_d, df_w)
    df_res = strategy.prepare_indicators(breakout_days=breakout_days, vol_mult=vol_mult)

    # --- 2. 运行滚仓回测 ---
    backtester = RollingBacktester(initial_capital=10000)
    df_equity, df_trades = backtester.run(df_res, {
        'rsi_bottom': rsi_bottom,
        'max_leverage': max_lev,
        'add_step': 0.08, # 每8%涨幅尝试加仓
        'trail_atr': trail_atr
    })
    
    # --- 3. 核心仪表盘 ---
    last_bar = df_res.iloc[-1]
    curr_price = last_bar['close']
    
    # 判断当前状态
    rsi_w_val = last_bar.get('RSI_W_DailyMap', 50)
    is_breakout = (curr_price > last_bar['High_N'])
    is_uptrend = (curr_price > last_bar['MA20'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("当前价格", f"${curr_price:,.2f}")
    with col2:
        delta = rsi_w_val - rsi_bottom
        color = "normal"
        if rsi_w_val < rsi_bottom: color = "inverse"
        st.metric("周线 RSI (底部信号)", f"{rsi_w_val:.1f}", delta=f"{delta:.1f} 距阈值", delta_color=color)
    with col3:
        status = "趋势向下 (空仓)"
        if is_breakout: status = "🔥 突破爆发 (可滚仓)"
        elif is_uptrend: status = "📈 趋势向上 (持仓)"
        st.metric("市场结构状态", status)
    with col4:
        roi = ((df_equity.iloc[-1]['equity'] - 10000) / 10000) * 100
        st.metric("策略模拟收益率", f"{roi:+.2f}%", help="过去2-3年按此策略执行的结果")

    # --- 4. 信号可视化图表 ---
    st.subheader("📊 信号与资金曲线回溯")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=df_res.index, open=df_res['open'], high=df_res['high'], low=df_res['low'], close=df_res['close'],
        name='Price'
    ), row=1, col=1)
    
    # 绘制关键均线
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['MA20'], line=dict(color='orange', width=1), name='MA20 (生死线)'), row=1, col=1)
    
    # 绘制突破线 (Donchian High)
    fig.add_trace(go.Scatter(x=df_res.index, y=df_res['High_N'], line=dict(color='gray', dash='dot', width=1), name=f'{breakout_days}日高点'), row=1, col=1)

    # 标记交易点
    if not df_trades.empty:
        buys = df_trades[df_trades['type'].str.contains('OPEN')]
        adds = df_trades[df_trades['type'] == 'ROLL_ADD']
        sells = df_trades[df_trades['type'] == 'CLOSE']
        
        fig.add_trace(go.Scatter(
            x=buys['date'], y=buys['price'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'),
            name='建仓 (Base)'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=adds['date'], y=adds['price'], mode='markers', marker=dict(symbol='star', size=10, color='gold'),
            name='滚仓 (Roll)'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=sells['date'], y=sells['price'], mode='markers', marker=dict(symbol='x', size=8, color='red'),
            name='离场 (Exit)'
        ), row=1, col=1)

    # 资金曲线
    fig.add_trace(go.Scatter(
        x=df_equity['date'], y=df_equity['equity'], line=dict(color='#6366f1', width=2), fill='tozeroy',
        name='策略净值'
    ), row=2, col=1)
    
    fig.update_layout(height=700, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. 交易记录 ---
    with st.expander("查看详细交易日志"):
        if not df_trades.empty:
            st.dataframe(df_trades.style.format({'price': '{:.2f}', 'pnl': '{:+.2f}', 'new_lev': '{:.2f}x'}))
        else:
            st.info("当前参数下，历史回测未触发交易信号。")

    # --- 6. 实时操作建议 ---
    st.markdown("### 🛡️ 华尔街首席执行建议")
    
    advice = ""
    if is_breakout and last_bar['Vol_Boom']:
        advice = """
        <div class='metric-card' style='border-color: #f59e0b;'>
            <h4 style='color: #fbbf24;'>🚀 触发黄金信号 2：趋势爆发</h4>
            <ul>
                <li><b>检测到：</b>价格突破 90 日震荡区间，且成交量放大。</li>
                <li><b>建议行动：</b>如果当前空仓，建议建立 30% 观察仓位；如果已有底仓且盈利，可开启第一次滚仓加仓。</li>
                <li><b>风控：</b>止损位务必设置在 MA20 均线下方。</li>
            </ul>
        </div>
        """
    elif rsi_w_val < rsi_bottom:
        advice = """
        <div class='metric-card' style='border-color: #10b981;'>
            <h4 style='color: #34d399;'>🌱 触发黄金信号 1：底部超卖</h4>
            <ul>
                <li><b>检测到：</b>周线 RSI 进入历史底部区域。</li>
                <li><b>建议行动：</b>定投买入现货，或建立 1x 低倍合约。<b>严禁重仓滚仓，此时是左侧接刀阶段。</b></li>
                <li><b>心态：</b>做好长期持有的准备，等待趋势反转信号。</li>
            </ul>
        </div>
        """
    elif is_uptrend:
        advice = """
        <div class='metric-card' style='border-color: #6366f1;'>
            <h4 style='color: #818cf8;'>📈 趋势持仓中 (Rolling)</h4>
            <ul>
                <li><b>状态：</b>价格位于 MA20 之上，趋势健康。</li>
                <li><b>建议行动：</b>持有底仓。如果浮盈超过 10%，可检查是否满足加仓条件。</li>
                <li><b>核心：</b>管住手，不要轻易止盈，让利润奔跑；同时紧盯移动止损线。</li>
            </ul>
        </div>
        """
    else:
        advice = """
        <div class='metric-card' style='border-color: #ef4444;'>
            <h4 style='color: #f87171;'>🛑 空仓观望 (Wait)</h4>
            <ul>
                <li><b>状态：</b>未触发底部信号，也未触发突破信号，或趋势已破坏。</li>
                <li><b>建议行动：</b>休息是交易的一部分。不要在垃圾时间里亏掉牛市的本金。</li>
            </ul>
        </div>
        """
    
    st.markdown(advice, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
