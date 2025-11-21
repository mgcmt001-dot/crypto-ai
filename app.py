import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. 系统配置
# ==========================================
st.set_page_config(
    page_title="Commander Cloud [OKX]",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 云端无需代理
PROXY = None 

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');
    :root { --bg:#0e1117; --card:#161b22; --border:#30363d; --gold:#d2a656; --green:#2ea043; --red:#da3633; --text:#e6edf3; }
    html,body,[class*="css"]{font-family:'Noto Sans SC',sans-serif;background:var(--bg);color:var(--text);}
    .pro-card { background: var(--card); border: 1px solid var(--border); border-radius: 6px; padding: 16px; margin-bottom: 16px; }
    .pc-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #21262d; padding-bottom: 10px; margin-bottom: 12px; }
    .pc-title { font-size: 16px; font-weight: 700; color: var(--gold); }
    .pc-tag { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
    .pc-item { display: flex; margin-bottom: 4px; font-size: 13px; color: #8b949e; }
    .pc-icon { color: var(--gold); margin-right: 8px; }
    .pc-plan { background: #0d1117; border: 1px dashed #30363d; border-radius: 4px; padding: 12px; }
    .pp-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 13px; }
    .pp-lbl { color: #8b949e; }
    .pp-val { font-family: 'JetBrains Mono'; font-weight: 700; }
    .bg-bull { background: rgba(46,160,67,0.15); color: var(--green); border:1px solid rgba(46,160,67,0.3); }
    .bg-bear { background: rgba(218,54,51,0.15); color: var(--red); border:1px solid rgba(218,54,51,0.3); }
    .bg-flat { background: rgba(139,148,158,0.1); color: #8b949e; border:1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心分析逻辑
# ==========================================
class WallStreetAnalyst:
    @staticmethod
    def deep_scan(df, tf_name):
        if df is None or len(df) < 60: return None
        c = df.iloc[-1]
        prev = df.iloc[-2]
        price = c['close']
        ema20, ema50 = c['EMA20'], c['EMA50']
        ma200 = c.get('MA200', np.nan)
        atr, rsi = c['ATR'], c['RSI']
        macd, sig = c['MACD'], c['SIGNAL']
        
        logics = []
        score = 0 
        
        # 趋势
        if price > ema20 > ema50:
            logics.append("多头排列：价格 > EMA20 > EMA50")
            score += 2
        elif price < ema20 < ema50:
            logics.append("空头排列：价格 < EMA20 < EMA50")
            score -= 2
            
        # RSI
        if rsi > 70:
            logics.append(f"RSI超买 ({rsi:.0f})：警惕回调")
            score -= 1 
        elif rsi < 30:
            logics.append(f"RSI超卖 ({rsi:.0f})：存在反弹需求")
            score += 1
            
        # MACD
        if macd > sig and c['HIST'] > prev['HIST']:
            logics.append("MACD动能增强 (多)")
            score += 1
        elif macd < sig and c['HIST'] < prev['HIST']:
            logics.append("MACD动能增强 (空)")
            score -= 1
            
        # K线
        body = abs(c['close'] - c['open'])
        lower = min(c['close'], c['open']) - c['low']
        upper = c['high'] - max(c['close'], c['open'])
        if lower > body * 2: logics.append("金针探底：下影线支撑")
        if upper > body * 2: logics.append("墓碑线：上方抛压")

        action, bias_text, css = "观望 (WAIT)", "震荡", "bg-flat"
        entry, sl, tp = 0, 0, 0
        risk = atr * 1.5 if not np.isnan(atr) else price * 0.02
        
        if score >= 3: 
            action, bias_text, css = "做多 (LONG)", "看涨", "bg-bull"
            entry, sl, tp = price, price - risk, price + risk * 2
        elif score <= -3:
            action, bias_text, css = "做空 (SHORT)", "看跌", "bg-bear"
            entry, sl, tp = price, price + risk, price - risk * 2
            
        return {"tf": tf_name, "bias": bias_text, "css": css, "logics": logics, "action": action, "entry": entry, "sl": sl, "tp": tp, "score": score}

# ==========================================
# 3. 数据引擎 (已切换至 OKX)
# ==========================================
class MarketDataEngine:
    def __init__(self):
        # 切换到 OKX，因为它对美国IP更友好
        config = {'timeout': 30000, 'enableRateLimit': True}
        self.ex = ccxt.okx(config)
    
    def fetch(self, symbol, tf):
        try:
            bars = self.ex.fetch_ohlcv(symbol, timeframe=tf, limit=100)
            if not bars: return None
            df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['EMA20'] = ta.ema(df['close'], length=20)
            df['EMA50'] = ta.ema(df['close'], length=50)
            df['MA200'] = ta.sma(df['close'], length=200)
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            if macd is not None:
                df['MACD'], df['SIGNAL'], df['HIST'] = macd.iloc[:,0], macd.iloc[:,1], macd.iloc[:,2]
            return df
        except Exception as e:
            # 打印错误到前台方便调试
            st.error(f"数据获取失败 {symbol} {tf}: {str(e)}")
            return None

    def get_all(self, symbol):
        # 改为顺序执行，防止云端 CPU 崩溃
        d = {}
        # OKX 同时也支持这些时间周期
        d['1m'] = self.fetch(symbol, '1m')
        d['15m'] = self.fetch(symbol, '15m')
        d['1h'] = self.fetch(symbol, '1h')
        d['1d'] = self.fetch(symbol, '1d')
        try:
            d['ticker'] = self.ex.fetch_ticker(symbol)
        except Exception as e:
            st.error(f"Ticker获取失败: {str(e)}")
            d['ticker'] = None
        return d

# ==========================================
# 4. 渲染与主程序
# ==========================================
def build_html(res):
    if not res: return "<div>数据加载中...</div>"
    items = "".join([f"<div class='pc-item'><span class='pc-icon'>•</span>{l}</div>" for l in res['logics']])
    
    if "观望" in res['action']:
        plan = "<div class='pc-plan' style='text-align:center;color:#666'>市场震荡，建议观望</div>"
    else:
        c = "#2ea043" if "多" in res['action'] else "#da3633"
        plan = f"<div class='pc-plan' style='border-color:{c}40'><div class='pp-row'><span>方向</span><span style='color:{c}'>{res['action']}</span></div><div class='pp-row'><span>入场</span><span>{res['entry']:.4f}</span></div><div class='pp-row'><span>止损</span><span style='color:#da3633'>{res['sl']:.4f}</span></div><div class='pp-row'><span>止盈</span><span style='color:#2ea043'>{res['tp']:.4f}</span></div></div>"
        
    return f"<div class='pro-card'><div class='pc-header'><span class='pc-title'>{res['tf']}</span><span class='pc-tag {res['css']}'>{res['bias']}</span></div>{items}{plan}</div>"

def main():
    with st.sidebar:
        st.title("COMMANDER OKX")
        # 确保这些交易对在 OKX 也是这个名字 (通常是的)
        coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'PEPE/USDT', 'ORDI/USDT', 'WIF/USDT']
        sel_coin = st.selectbox("选择标的 (OKX数据)", coins)
        if st.button("🔄 刷新数据"): st.rerun()

    eng = MarketDataEngine()
    with st.spinner("正在从 OKX 获取数据 (云端模式)..."):
        data = eng.get_all(sel_coin)
        
    if data and data.get('ticker'):
        t = data['ticker']
        clr = "#2ea043" if t['percentage'] >= 0 else "#da3633"
        st.markdown(f"<div style='background:#161b22;border:1px solid #d2a656;padding:15px;border-radius:6px;margin-bottom:20px;display:flex;justify-content:space-between'><div><div style='color:#d2a656;font-weight:bold;font-size:18px'>{sel_coin}</div></div><div style='text-align:right'><div style='font-size:24px;font-weight:bold'>${t['last']:.4f}</div><div style='color:{clr}'>{t['percentage']:.2f}%</div></div></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(build_html(WallStreetAnalyst.deep_scan(data['1m'], "1分钟")), unsafe_allow_html=True)
            st.markdown(build_html(WallStreetAnalyst.deep_scan(data['15m'], "15分钟")), unsafe_allow_html=True)
        with c2:
            st.markdown(build_html(WallStreetAnalyst.deep_scan(data['1h'], "1小时")), unsafe_allow_html=True)
            st.markdown(build_html(WallStreetAnalyst.deep_scan(data['1d'], "日线")), unsafe_allow_html=True)
            
        with st.expander("📊 K线图"):
            if data['1h'] is not None:
                df = data['1h']
                fig = go.Figure(data=[go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], increasing_line_color='#2ea043', decreasing_line_color='#da3633')])
                fig.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=0,b=0), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("数据获取失败，可能是 OKX 接口暂时繁忙，请稍后刷新。")

if __name__ == "__main__":
    main()
