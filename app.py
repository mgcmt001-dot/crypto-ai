import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import concurrent.futures
from datetime import datetime

# ==========================================
# 1. 系统配置 (System Config)
# ==========================================
st.set_page_config(
    page_title="Commander-zzjszz [Pro]",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 定义 CSS 样式（压缩为单行或无缩进块，防止渲染错误）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    :root { --bg:#0e1117; --card:#161b22; --border:#30363d; --gold:#d2a656; --green:#2ea043; --red:#da3633; --text:#e6edf3; }
    html,body,[class*="css"]{font-family:'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);}
    
    /* 卡片容器 */
    .pro-card {
        background: var(--card); border: 1px solid var(--border); border-radius: 6px; 
        padding: 16px; margin-bottom: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .pro-card:hover { border-color: var(--gold); }
    
    /* 头部 */
    .pc-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 12px; }
    .pc-title { font-size: 16px; font-weight: 700; color: var(--gold); }
    .pc-tag { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    
    /* 逻辑列表 */
    .pc-logic { font-size: 13px; color: #8b949e; line-height: 1.6; margin-bottom: 15px; }
    .pc-item { display: flex; margin-bottom: 4px; }
    .pc-icon { color: var(--gold); margin-right: 8px; font-weight: bold; }
    
    /* 交易计划表格 */
    .pc-plan { background: #0d1117; border: 1px dashed #30363d; border-radius: 4px; padding: 12px; }
    .pp-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px; }
    .pp-lbl { color: #8b949e; }
    .pp-val { font-family: 'JetBrains Mono'; font-weight: 700; }
    
    /* 颜色定义 */
    .c-bull { color: var(--green); } .bg-bull { background: rgba(46,160,67,0.15); color: var(--green); border:1px solid rgba(46,160,67,0.3); }
    .c-bear { color: var(--red); } .bg-bear { background: rgba(218,54,51,0.15); color: var(--red); border:1px solid rgba(218,54,51,0.3); }
    .c-flat { color: #8b949e; } .bg-flat { background: rgba(139,148,158,0.1); color: #8b949e; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 华尔街深度分析引擎 (The Brain)
# ==========================================
class WallStreetAnalyst:
    @staticmethod
    def deep_scan(df, tf_name):
        if df is None or len(df) < 60: return None
        
        c = df.iloc[-1]
        prev = df.iloc[-2]
        
        # --- 基础指标 ---
        price = c['close']
        ema20 = c['EMA20']
        ema50 = c['EMA50']
        ma200 = c.get('MA200', np.nan)
        atr = c['ATR']
        rsi = c['RSI']
        macd = c['MACD']
        sig = c['SIGNAL']
        vol_ma = df['vol'].mean()
        rvol = c['vol'] / vol_ma if vol_ma > 0 else 1.0
        
        # --- 逻辑推导容器 ---
        logics = []
        score = 0 # 评分系统: >2 做多, <-2 做空
        
        # 1. 趋势结构 (Trend Structure)
        if price > ema20 > ema50:
            logics.append("多头排列：价格 > EMA20 > EMA50，买盘控盘，趋势向上。")
            score += 2
        elif price < ema20 < ema50:
            logics.append("空头排列：价格 < EMA20 < EMA50，卖盘压制，趋势向下。")
            score -= 2
        else:
            logics.append("均线纠缠：EMA短期均线粘合，市场处于震荡蓄势阶段。")
            
        # 2. 动能分析 (Momentum)
        if rsi > 70:
            logics.append(f"RSI超买 ({rsi:.0f})：买力过度消耗，警惕回调风险。")
            score -= 1 # 逆向思维，超买不宜追高
        elif rsi < 30:
            logics.append(f"RSI超卖 ({rsi:.0f})：卖力过度消耗，存在反弹需求。")
            score += 1
            
        if macd > sig and c['HIST'] > 0:
            if c['HIST'] > prev['HIST']:
                logics.append("MACD增强：多头动能正在持续放大。")
                score += 1
        elif macd < sig and c['HIST'] < 0:
             if c['HIST'] < prev['HIST']:
                logics.append("MACD增强：空头动能正在持续放大。")
                score -= 1
                
        # 3. 量价行为 (Price Action & Volume)
        body = abs(c['close'] - c['open'])
        lower_wick = min(c['close'], c['open']) - c['low']
        upper_wick = c['high'] - max(c['close'], c['open'])
        
        if rvol > 1.5:
            term = "放量" if c['close'] > c['open'] else "放量抛压"
            logics.append(f"资金异动：成交量放大 {rvol:.1f}倍 ({term})，机构介入。")
            score += 1 if c['close'] > c['open'] else -1
            
        if lower_wick > body * 2:
            logics.append("金针探底：长下影线显示低位有强力承接。")
            score += 1
        if upper_wick > body * 2:
            logics.append("墓碑线：长上影线显示高位抛压沉重。")
            score -= 1

        # --- 策略生成 ---
        action = "观望 (WAIT)"
        bias_text = "震荡整理"
        css_class = "bg-flat"
        entry, sl, tp = 0, 0, 0
        
        risk_unit = atr * 1.5 if not np.isnan(atr) else price * 0.02
        
        if score >= 3: # 严格门槛
            action = "做多 (LONG)"
            bias_text = "强烈看涨"
            css_class = "bg-bull"
            entry = price
            sl = price - risk_unit
            # 智能止损优化
            if not np.isnan(ma200) and price > ma200 and (price - ma200) < risk_unit:
                sl = ma200 * 0.995
            tp = price + risk_unit * 2
            
        elif score <= -3:
            action = "做空 (SHORT)"
            bias_text = "强烈看跌"
            css_class = "bg-bear"
            entry = price
            sl = price + risk_unit
            if not np.isnan(ma200) and price < ma200 and (ma200 - price) < risk_unit:
                sl = ma200 * 1.005
            tp = price - risk_unit * 2
            
        return {
            "tf": tf_name,
            "bias": bias_text,
            "css": css_class,
            "logics": logics,
            "action": action,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "score": score
        }

# ==========================================
# 3. 稳健数据层 (Data Engine)
# ==========================================
class MarketDataEngine:
    def __init__(self):
        config = {
    'timeout': 20000, 
    'enableRateLimit': True
}
if PROXY:
    config['proxies'] = {'http': PROXY, 'https': PROXY}
    
self.ex = ccxt.binance(config)
    
    def fetch(self, symbol, tf):
        try:
            # 抓取足够数据以确保指标稳定
            bars = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=300)
            if not bars: return None
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # 计算指标
            df['EMA20'] = ta.ema(df['close'], length=20)
            df['EMA50'] = ta.ema(df['close'], length=50)
            df['MA200'] = ta.sma(df['close'], length=200)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            if macd is not None:
                df['MACD'] = macd.iloc[:, 0]
                df['SIGNAL'] = macd.iloc[:, 1]
                df['HIST'] = macd.iloc[:, 2]
            return df
        except: return None

    def get_all(self, symbol):
        d = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = {
                '1m': executor.submit(self.fetch, symbol, '1m'),
                '15m': executor.submit(self.fetch, symbol, '15m'),
                '1h': executor.submit(self.fetch, symbol, '1h'),
                '1d': executor.submit(self.fetch, symbol, '1d'),
                'ticker': executor.submit(self.ex.fetch_ticker, symbol)
            }
            for k, v in tasks.items():
                try: d[k] = v.result()
                except: d[k] = None
        return d

# ==========================================
# 4. 安全渲染层 (Safe HTML Rendering)
# ==========================================
def build_card_html(res):
    if not res: return "<div style='color:red'>数据不足</div>"
    
    # 1. 构建逻辑列表 (String Concatenation only, NO indentation)
    logic_items = ""
    for lg in res['logics']:
        logic_items += f"<div class='pc-item'><span class='pc-icon'>•</span><span>{lg}</span></div>"
    
    # 2. 构建交易计划
    plan_html = ""
    if "观望" in res['action']:
        plan_html = "<div class='pc-plan' style='text-align:center; color:#666;'><div>⚖️ 市场震荡中</div><div style='font-size:12px'>建议空仓等待方向明确</div></div>"
    else:
        c_val = "#2ea043" if "多" in res['action'] else "#da3633"
        # 逐行构建，避免换行符
        p_rows = ""
        p_rows += f"<div class='pp-row'><span class='pp-lbl'>操作建议</span><span class='pp-val' style='color:{c_val}'>{res['action']}</span></div>"
        p_rows += f"<div class='pp-row'><span class='pp-lbl'>建议入场</span><span class='pp-val'>${res['entry']:,.2f}</span></div>"
        p_rows += f"<div class='pp-row'><span class='pp-lbl'>止损位</span><span class='pp-val' style='color:#da3633'>${res['sl']:,.2f}</span></div>"
        p_rows += f"<div class='pp-row'><span class='pp-lbl'>目标位</span><span class='pp-val' style='color:#2ea043'>${res['tp']:,.2f}</span></div>"
        plan_html = f"<div class='pc-plan' style='border-color:{c_val}40'>{p_rows}</div>"

    # 3. 组合最终 HTML (一行流)
    html = f"<div class='pro-card'><div class='pc-header'><span class='pc-title'>{res['tf']}</span><span class='pc-tag {res['css']}'>{res['bias']}</span></div><div class='pc-logic'>{logic_items}</div>{plan_html}</div>"
    
    return html

# ==========================================
# 5. 主程序 (Main)
# ==========================================
def main():
    with st.sidebar:
        st.title("COMMANDER V21")
        st.caption("华尔街深度策略版")
        
        coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'ZEC/USDT', 'DASH/USDT','DOGE/USDT', 'XRP/USDT', 'PEPE/USDT', 'ORDI/USDT']
        sel_coin = st.selectbox("选择标的", coins)
        
        if st.button("⚡ 立即分析市场", use_container_width=True):
            st.rerun()
            
        st.markdown("---")
        st.info("策略模型：\n1. 趋势共振 (Trend)\n2. 动能衰竭/增强 (Momentum)\n3. 机构量能 (VPA)\n4. 智能止损 (Smart SL)")

    eng = MarketDataEngine()
    
    with st.spinner(f"正在进行全周期扫描: {sel_coin} ..."):
        data = eng.get_all(sel_coin)
        
    if not data or not data.get('ticker'):
        st.error("网络连接失败，请检查代理设置。")
        st.stop()
        
    # --- 顶部行情 ---
    tick = data['ticker']
    p_color = "#2ea043" if tick['percentage'] >= 0 else "#da3633"
    # 使用列表拼接而非多行字符串
    head_parts = []
    head_parts.append("<div style='background:#161b22; border:1px solid #d2a656; padding:15px; border-radius:6px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;'>")
    head_parts.append(f"<div><div style='color:#d2a656; font-weight:bold; font-size:18px;'>{sel_coin} 深度研报</div><div style='color:#8b949e; font-size:12px;'>报告时间: {datetime.now().strftime('%H:%M:%S')}</div></div>")
    head_parts.append(f"<div style='text-align:right'><div style='font-size:28px; font-weight:bold; color:#e6edf3'>${tick['last']:,.2f}</div><div style='color:{p_color}; font-weight:bold'>{tick['percentage']:.2f}%</div></div>")
    head_parts.append("</div>")
    st.markdown("".join(head_parts), unsafe_allow_html=True)
    
    # --- 分析卡片 ---
    c1, c2 = st.columns(2)
    
    # 计算逻辑
    r_1m = WallStreetAnalyst.deep_scan(data['1m'], "超短线 (1 Min)")
    r_15m = WallStreetAnalyst.deep_scan(data['15m'], "日内 (15 Min)")
    r_1h = WallStreetAnalyst.deep_scan(data['1h'], "波段 (1 Hour)")
    r_1d = WallStreetAnalyst.deep_scan(data['1d'], "趋势 (1 Day)")
    
    # 渲染
    with c1:
        st.markdown("#### ⚡ 短线博弈")
        st.markdown(build_card_html(r_1m), unsafe_allow_html=True)
        st.markdown(build_card_html(r_15m), unsafe_allow_html=True)
        
    with c2:
        st.markdown("#### 🌊 趋势布局")
        st.markdown(build_card_html(r_1h), unsafe_allow_html=True)
        st.markdown(build_card_html(r_1d), unsafe_allow_html=True)
        
    # --- 最终建议 ---
    # 计分板逻辑
    total_score = 0
    if r_15m: total_score += r_15m['score']
    if r_1h: total_score += r_1h['score'] * 1.5 # 1小时权重更高
    if r_1d: total_score += r_1d['score'] * 2.0 # 日线权重最高
    
    final_text = "市场混沌，建议观望"
    f_bg = "#8b949e"
    
    if total_score >= 4:
        final_text = "💎 极强多头共振 (全仓做多信号)"
        f_bg = "#2ea043"
    elif total_score >= 2:
        final_text = "📈 震荡偏多 (逢低做多)"
        f_bg = "#2ea043"
    elif total_score <= -4:
        final_text = "⚠️ 极强空头共振 (清仓/做空信号)"
        f_bg = "#da3633"
    elif total_score <= -2:
        final_text = "📉 震荡偏空 (逢高做空)"
        f_bg = "#da3633"
        
    sum_html = f"<div style='background:{f_bg}20; border:1px solid {f_bg}; padding:20px; border-radius:6px; text-align:center; margin-top:20px;'><div style='color:{f_bg}; font-weight:bold; font-size:14px;'>首席分析师最终裁决</div><div style='color:#e6edf3; font-size:24px; font-weight:bold; margin:10px 0;'>{final_text}</div><div style='color:#8b949e; font-size:13px'>综合评分: {total_score:.1f} (评分>4为极强信号)</div></div>"
    st.markdown(sum_html, unsafe_allow_html=True)
    
    # --- 图表 ---
    with st.expander("📊 查看 1小时 K线深度图 (Price Action)", expanded=True):
        if data['1h'] is not None:
            df = data['1h']
            fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                increasing_line_color='#2ea043', decreasing_line_color='#da3633')])
            fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":

    main()

