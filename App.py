"""
ChatSolveAI — Streamlit frontend (v3.0) — single page with sidebar nav.

Replaces the previous multipage layout (App.py + pages/1_Admin_Dashboard.py).
Streamlit's pages/ directory caused DOM remnants to persist across page swaps;
folding both views into one script eliminates that class of bug entirely.
"""

from __future__ import annotations
import base64, json, os, time, uuid
from datetime import datetime
from functools import lru_cache
import requests, streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# streamlit-lottie is optional polish; if the package isn't installed
# the empty-state animation is silently skipped instead of crashing.
try:
    from streamlit_lottie import st_lottie
    _HAS_LOTTIE = True
except Exception:
    st_lottie = None  # type: ignore
    _HAS_LOTTIE = False

@lru_cache(maxsize=4)
def _load_lottie(path: str, mtime: float):
    with open(path, "r") as f:
        return json.load(f)

def _lottie_data(path: str):
    return _load_lottie(path, os.path.getmtime(path))

@lru_cache(maxsize=16)
def _img_b64_cached(path: str, mtime: float) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _img_b64(path: str) -> str:
    # Include mtime in cache key so updated images bust the cache.
    return _img_b64_cached(path, os.path.getmtime(path))

load_dotenv()

# ══════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════
_SECRET_API_URL = _SECRET_API_KEY = _SECRET_ADMIN_PW = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _SECRET_API_KEY = st.secrets.get("API_KEY")
    _SECRET_ADMIN_PW = st.secrets.get("ADMIN_PASSWORD")
except: pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""
ADMIN_PASSWORD = _SECRET_ADMIN_PW or os.getenv("ADMIN_PASSWORD") or ""
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT","20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES","2"))
USE_STREAMING    = os.getenv("USE_STREAMING","true").lower() not in {"0","false","no"}

def _api_headers(): return {"X-API-Key": API_KEY} if API_KEY else {}

# ══════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════
st.set_page_config(page_title="ChatSolveAI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lexend:wght@600;700&display=swap" rel="stylesheet">
<style>
:root {--accent:#4F8BF9;--accent-2:#8E6BFF;--font-ui:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;--font-display:'Lexend','Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;}
html,body,[data-testid='stApp'],[data-testid='stSidebar'],[data-testid='stChatInput'] textarea,[data-testid='stChatMessage'],.stMarkdown,.stButton button,.stDownloadButton button,[class*='st-key-iconbtn_'] button,[class*='st-key-chipwrap_'] button,.stTextInput input,.stSelectbox div[role='combobox']{font-family:var(--font-ui)!important;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;text-rendering:optimizeLegibility}
.hero-title,h1,h2,h3,.drill-section h3,[data-testid='stChatMessage'] h1,[data-testid='stChatMessage'] h2,[data-testid='stChatMessage'] h3{font-family:var(--font-display)!important;letter-spacing:-.01em}
code,pre,kbd,samp,tt,[class*='monospace'],[data-testid='stCode']{font-family:'JetBrains Mono','SF Mono',Monaco,Consolas,'Roboto Mono',monospace!important}
body{font-weight:400}
.hero-title{background:linear-gradient(90deg,var(--accent)0%,var(--accent-2)100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-weight:700;font-size:2.1rem;letter-spacing:-.02em;margin-bottom:0.2rem}
.hero-sub{color:#9ea3b0;margin-bottom:1.2rem;font-weight:400}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:999px;font-size:0.72rem;font-weight:600;background:rgba(79,139,249,.12);color:var(--accent);margin-right:6px}
.pill-green{background:rgba(76,175,80,.15);color:#81c784}
.pill-amber{background:rgba(255,152,0,.15);color:#ffb74d}
.pill-red{background:rgba(244,67,54,.15);color:#e57373}
.pill-purple{background:rgba(156,39,176,.15);color:#ba68c8}
.meter-wrap{background:rgba(255,255,255,.06);border-radius:6px;height:6px;overflow:hidden;margin-top:4px}
.meter-fill{height:100%;background:linear-gradient(90deg,#4CAF50,#81C784)}
.src-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-left:3px solid rgba(79,139,249,.8);padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:.82rem;color:#d4d7dd;transition:transform .15s ease,box-shadow .2s ease,border-left-color .2s ease,background-color .2s ease}
.src-card:hover{transform:translateY(-2px);box-shadow:0 6px 16px rgba(79,139,249,.15);border-left-color:#4F8BF9;background:rgba(79,139,249,.08)}
.src-card.top{border-left-color:#66bb6a;background:rgba(102,187,106,.08)}
.src-card.top:hover{border-left-color:#81c784;box-shadow:0 6px 16px rgba(102,187,106,.18);background:rgba(102,187,106,.14)}
.src-meta{font-size:.68rem;color:#7a8190;margin-top:4px}
[class*='st-key-chipwrap_']{max-width:800px;width:100%;margin-left:auto!important;margin-right:auto!important;transform-origin:center;opacity:0}
[class*='st-key-chipwrap_'] button{width:100%!important;background:rgba(255,255,255,.03)!important;border:1px solid rgba(255,255,255,.08)!important;color:#D1D5DB!important;font-weight:500!important;text-align:left!important;padding:12px 38px 12px 16px!important;border-radius:12px!important;box-shadow:inset 4px 0 0 transparent!important;position:relative!important;transition:background-color .2s ease-in-out,border-color .2s ease-in-out,box-shadow .2s ease-in-out,transform .15s ease-in-out,color .2s ease-in-out!important}
[class*='st-key-chipwrap_'] button::after{content:'→';position:absolute;right:16px;top:50%;transform:translateY(-50%) translateX(0);color:rgba(255,255,255,.35);font-size:1rem;font-weight:500;transition:color .2s ease-in-out,transform .25s cubic-bezier(.16,1,.3,1);pointer-events:none}
[class*='st-key-chipwrap_'] button:hover{background:rgba(255,255,255,.06)!important;border-color:rgba(79,139,249,.30)!important;color:#E5E7EB!important;box-shadow:inset 4px 0 0 #4F8BF9,0 4px 14px rgba(0,0,0,.18)!important;transform:translateX(5px)!important}
[class*='st-key-chipwrap_'] button:hover::after{color:var(--accent);transform:translateY(-50%) translateX(4px)}
[class*='st-key-chipwrap_'] button:active{transform:translateX(3px) scale(.99)!important}
[class*='st-key-iconbtn_'] button{transition:transform .4s cubic-bezier(.16,1,.3,1),opacity .4s cubic-bezier(.16,1,.3,1),box-shadow .3s cubic-bezier(.16,1,.3,1),background-color .2s ease-in-out,border-color .2s ease-in-out!important}
[class*='st-key-iconbtn_'] button:hover{transform:translateY(-2px)}
[data-testid='stButton'] button,[data-testid='stDownloadButton'] button{transition:background-color .15s ease,border-color .15s ease,transform .12s ease,box-shadow .15s ease!important}
[class*='st-key-btn_new_chat'] button:hover,[data-testid='stDownloadButton'] button:hover,[class*='st-key-up_'] button:hover,[class*='st-key-down_'] button:hover,[class*='st-key-regen_'] button:hover,[class*='st-key-admin_signout'] button:hover{transform:translateY(-1px) scale(1.02);box-shadow:0 4px 12px rgba(79,139,249,.18);border-color:#4F8BF9!important}
[class*='st-key-btn_new_chat'] button:active,[data-testid='stDownloadButton'] button:active,[class*='st-key-up_'] button:active,[class*='st-key-down_'] button:active,[class*='st-key-regen_'] button:active{transform:translateY(0) scale(.98)}
[data-testid='stChatInput'] textarea,[data-baseweb='input'] input{transition:all .3s ease!important}
[data-testid='stChatInput']:focus-within,[data-baseweb='input']:focus-within{box-shadow:0 0 15px rgba(79,139,249,.20)!important;border-color:#4F8BF9!important;border-radius:14px!important}
[data-testid='stChatInput'] textarea:focus,[data-baseweb='input'] input:focus{outline:none!important}
[data-testid='stMain'] .block-container{max-width:1100px!important;margin-left:auto!important;margin-right:auto!important}
[data-testid='stChatInput']{border-radius:14px!important;border:1px solid rgba(255,255,255,.08)!important;background:rgba(28,34,46,.92)!important;transition:all .3s ease!important;backdrop-filter:saturate(140%)}
[data-testid='stChatInput'] textarea::placeholder{color:#6b7280!important;opacity:.85!important}
[data-testid='stChatInput'] button{border-radius:10px!important;transition:background-color .15s ease,transform .12s ease!important}
[data-testid='stChatInput'] button:hover{background:rgba(79,139,249,.22)!important;transform:translateY(-1px)}
[data-testid='stChatInput'] button:active{transform:translateY(0) scale(.96)}
@keyframes inputPulse{0%,100%{box-shadow:0 0 0 0 rgba(79,139,249,0)}50%{box-shadow:0 0 0 4px rgba(79,139,249,.20),0 0 18px rgba(79,139,249,.18)}}
.agent-status{display:flex;align-items:center;gap:9px;padding:8px 12px;border-radius:10px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);font-size:.82rem;font-weight:500;margin-bottom:8px}
.agent-status__dot{width:9px;height:9px;border-radius:50%;display:inline-block;flex-shrink:0}
.agent-status--online .agent-status__dot{background:#34d399;box-shadow:0 0 0 3px rgba(52,211,153,.20),0 0 10px rgba(52,211,153,.55);animation:dotPulse 1.6s ease-in-out infinite}
.agent-status--online .agent-status__label{color:#a7f3d0}
.agent-status--offline .agent-status__dot{background:#ef4444;box-shadow:0 0 0 3px rgba(239,68,68,.18)}
.agent-status--offline .agent-status__label{color:#fca5a5}
@keyframes dotPulse{0%,100%{box-shadow:0 0 0 3px rgba(52,211,153,.20),0 0 10px rgba(52,211,153,.55);transform:scale(1)}50%{box-shadow:0 0 0 6px rgba(52,211,153,.10),0 0 18px rgba(52,211,153,.85);transform:scale(1.08)}}
.session-stats{margin:10px 0 4px;padding:10px 12px;border-radius:12px;background:rgba(79,139,249,.05);border:1px solid rgba(79,139,249,.14)}
.session-stats__title{font-family:var(--font-display);font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.10em;color:#9ea3b0;margin-bottom:6px}
[data-testid='stSidebar'] [data-testid='stMetric']{padding:4px 0!important;background:transparent!important}
[data-testid='stSidebar'] [data-testid='stMetricLabel']{font-size:.62rem!important;text-transform:uppercase;letter-spacing:.08em;color:#7a8190!important;font-weight:600!important}
[data-testid='stSidebar'] [data-testid='stMetricLabel'] p{font-size:.62rem!important}
[data-testid='stSidebar'] [data-testid='stMetricValue']{font-family:var(--font-display)!important;font-size:1.35rem!important;color:#E5E7EB!important;font-weight:600!important;text-shadow:0 0 12px rgba(79,139,249,.28)}
[data-testid='stSidebar'] [data-testid='stMetricDelta']{display:none!important}
*::-webkit-scrollbar{width:8px;height:8px}
*::-webkit-scrollbar-track{background:rgba(255,255,255,.04);border-radius:4px}
*::-webkit-scrollbar-thumb{background:rgba(79,139,249,.40);border-radius:4px;border:1px solid rgba(255,255,255,.04)}
*::-webkit-scrollbar-thumb:hover{background:rgba(79,139,249,.65)}
*::-webkit-scrollbar-corner{background:transparent}
*{scrollbar-width:thin;scrollbar-color:rgba(79,139,249,.40) rgba(255,255,255,.04)}
a,a:visited{transition:color .2s ease-in-out,opacity .2s ease-in-out,text-decoration-color .2s ease-in-out}
a:hover{color:var(--accent)}
[data-testid='stExpander'] summary,[data-testid='stCaptionContainer']{transition:color .2s ease-in-out,background-color .2s ease-in-out}
[data-testid='stExpander'] summary:hover{color:var(--accent)!important}
[data-testid='stMarkdownContainer'] table,.stTable table{width:100%!important;border-collapse:separate!important;border-spacing:0!important;margin:12px 0!important;background:rgba(255,255,255,.02)!important;border:1px solid rgba(255,255,255,.06)!important;border-radius:10px!important;overflow:hidden!important;font-size:.85rem}
[data-testid='stMarkdownContainer'] table th,.stTable table th{padding:10px 14px!important;text-align:left!important;font-family:var(--font-display)!important;font-size:.72rem!important;font-weight:700!important;text-transform:uppercase!important;letter-spacing:.08em!important;color:#cdd5e0!important;background:rgba(79,139,249,.08)!important;border-bottom:2px solid #4F8BF9!important}
[data-testid='stMarkdownContainer'] table td,.stTable table td{padding:9px 14px!important;color:#d4d7dd!important;border-bottom:1px solid rgba(255,255,255,.04)!important;transition:background-color .15s ease-in-out}
[data-testid='stMarkdownContainer'] table tr:nth-child(even) td,.stTable table tr:nth-child(even) td{background:rgba(255,255,255,.02)!important}
[data-testid='stMarkdownContainer'] table tr:hover td,.stTable table tr:hover td{background:rgba(79,139,249,.10)!important}
[data-testid='stMarkdownContainer'] table tr:last-child td,.stTable table tr:last-child td{border-bottom:none!important}
[data-testid='stMarkdownContainer'] pre,pre[class*='language-']{background:#090B10!important;border:1px solid rgba(255,255,255,.06)!important;border-radius:10px!important;padding:14px 18px!important;margin:10px 0!important;font-size:.85rem!important;line-height:1.55!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.02),0 4px 14px rgba(0,0,0,.25)!important;overflow-x:auto}
[data-testid='stMarkdownContainer'] pre code,pre[class*='language-'] code{background:transparent!important;color:#C0CAF5!important;padding:0!important;border-radius:0!important;font-family:'JetBrains Mono','SF Mono',Monaco,Consolas,'Roboto Mono',monospace!important}
[data-testid='stMarkdownContainer'] :not(pre)>code,[data-testid='stMarkdownContainer'] p>code,[data-testid='stMarkdownContainer'] li>code{background:rgba(79,139,249,.14)!important;color:#a5c2f5!important;border:1px solid rgba(79,139,249,.20)!important;padding:1px 6px!important;border-radius:5px!important;font-size:.85em!important;font-family:'JetBrains Mono','SF Mono',Monaco,Consolas,'Roboto Mono',monospace!important}
.token.keyword,.hljs-keyword,.hljs-built_in{color:#7AA2F7!important;font-weight:600}
.token.string,.hljs-string,.hljs-attr{color:#9ECE6A!important}
.token.number,.token.boolean,.hljs-number,.hljs-literal{color:#FF9E64!important}
.token.comment,.hljs-comment{color:#7a8190!important;font-style:italic}
.token.function,.token.class-name,.hljs-function,.hljs-title,.hljs-class{color:#BB9AF7!important}
.token.operator,.token.punctuation,.hljs-operator,.hljs-punctuation{color:#C0CAF5!important}
.token.tag,.hljs-tag,.hljs-name{color:#F7768E!important}
.token.property,.hljs-attribute{color:#E0AF68!important}
[data-testid='stCodeCopyButton'],[data-testid='stCodeCopyButton'] button,button[title='Copy to clipboard']{background:rgba(79,139,249,.10)!important;border:1px solid rgba(79,139,249,.25)!important;border-radius:8px!important;color:#cdd5e0!important;transition:background-color .2s ease-in-out,border-color .2s ease-in-out,transform .12s ease-in-out!important}
[data-testid='stCodeCopyButton']:hover,[data-testid='stCodeCopyButton'] button:hover,button[title='Copy to clipboard']:hover{background:rgba(79,139,249,.22)!important;border-color:#4F8BF9!important;transform:translateY(-1px)}
.drill-section{overflow:hidden;perspective:1000px;perspective-origin:center top;animation:drillExpand .45s cubic-bezier(.16,1,.3,1) both}
@keyframes drillExpand{0%{max-height:0;opacity:0;transform:translateY(-8px) scale(.98)}60%{opacity:.85}100%{max-height:1200px;opacity:1;transform:translateY(0) scale(1)}}
.drill-section h3{display:inline-flex;align-items:center;gap:10px;font-family:var(--font-display)!important;font-weight:700;font-size:1.25rem;letter-spacing:.02em;color:#E5E7EB;margin:8px 0 16px!important;padding-bottom:10px;background-image:linear-gradient(90deg,var(--accent) 0%,rgba(79,139,249,.45) 30%,transparent 75%);background-size:60% 2px;background-repeat:no-repeat;background-position:0 100%;animation:sectionHeaderFade .8s cubic-bezier(.22,1,.36,1) both}
.drill-section h3::before{content:'✦';color:var(--accent);font-size:.9em;text-shadow:0 0 12px rgba(79,139,249,.55);flex-shrink:0;display:inline-block}
@keyframes sectionHeaderFade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideUpFade{0%{opacity:0;transform:translateY(-40px) translateX(-30px) rotate(-4deg) scale(.85);filter:drop-shadow(0 0 0 rgba(79,139,249,0))}70%{filter:drop-shadow(0 0 22px rgba(79,139,249,.55))}100%{opacity:1;transform:translateY(0) translateX(0) rotate(0deg) scale(1);filter:drop-shadow(0 0 0 rgba(79,139,249,0))}}
@keyframes slideUpFadeRight{0%{opacity:0;transform:translateY(-40px) translateX(30px) rotate(4deg) scale(.85);filter:drop-shadow(0 0 0 rgba(79,139,249,0))}70%{filter:drop-shadow(0 0 22px rgba(79,139,249,.55))}100%{opacity:1;transform:translateY(0) translateX(0) rotate(0deg) scale(1);filter:drop-shadow(0 0 0 rgba(79,139,249,0))}}
[data-testid='stChatMessage']{border-radius:14px!important;padding:12px 14px!important;margin-bottom:10px!important;box-shadow:0 4px 18px rgba(0,0,0,.25);transition:box-shadow .2s ease,transform .2s ease}
[data-testid='stChatMessage']:hover{box-shadow:0 6px 22px rgba(0,0,0,.32)}
[class*='st-key-chatmsg-user'] [data-testid='stChatMessage']{background:rgba(79,139,249,.10)!important;border:1px solid rgba(79,139,249,.22)!important;border-left:3px solid #4F8BF9!important;backdrop-filter:blur(10px) saturate(140%)!important;-webkit-backdrop-filter:blur(10px) saturate(140%)!important;animation:bubbleInRight .35s cubic-bezier(.16,1,.3,1) both}
[class*='st-key-chatmsg-asst'] [data-testid='stChatMessage']{background:rgba(20,24,32,.45)!important;border:1px solid rgba(255,255,255,.08)!important;border-left:3px solid rgba(255,255,255,.14)!important;backdrop-filter:blur(10px) saturate(140%)!important;-webkit-backdrop-filter:blur(10px) saturate(140%)!important;animation:bubbleInLeft .35s cubic-bezier(.16,1,.3,1) both}
[data-testid='stAppViewContainer']{position:relative;isolation:isolate;background:#0E1117}
[data-testid='stAppViewContainer']::before{content:'';position:fixed;inset:0;z-index:-1;pointer-events:none;background:radial-gradient(circle at 18% 20%,rgba(79,139,249,.10) 0%,transparent 45%),radial-gradient(circle at 82% 25%,rgba(142,107,255,.08) 0%,transparent 50%),radial-gradient(circle at 30% 85%,rgba(30,41,59,.55) 0%,transparent 55%),radial-gradient(circle at 75% 75%,rgba(79,139,249,.06) 0%,transparent 50%),linear-gradient(180deg,#0E1117 0%,#161A23 60%,#1E293B 100%);background-size:220% 220%,220% 220%,220% 220%,220% 220%,100% 100%;background-position:0% 0%,100% 0%,0% 100%,100% 100%,0 0;animation:meshFlow 28s ease-in-out infinite}
@keyframes meshFlow{0%,100%{background-position:0% 0%,100% 0%,0% 100%,100% 100%,0 0}50%{background-position:100% 50%,0% 50%,100% 0%,0% 100%,0 0}}
[data-testid='stSidebar']{background:rgba(20,24,32,.65)!important;backdrop-filter:blur(14px) saturate(140%);-webkit-backdrop-filter:blur(14px) saturate(140%);border-right:1px solid rgba(255,255,255,.06)}
[data-testid='stSidebar'] > div{background:transparent!important}
[data-testid='stSidebar'][aria-expanded='false']{overflow:hidden!important;width:0!important;min-width:0!important;border-right:none!important}
[data-testid='stSidebar'][aria-expanded='false'] *{visibility:hidden!important;opacity:0!important;pointer-events:none!important}
[data-testid='stSidebarCollapseButton'] *,[data-testid='stSidebarCollapsedControl'] *{visibility:visible!important;opacity:1!important;pointer-events:auto!important}
.st-key-lottie_wrap iframe,.st-key-lottie_wrap{background:transparent!important}
.st-key-lottie_wrap iframe{display:block;margin:0 auto;max-width:360px;width:100%!important}
@media (max-width: 768px){[data-testid='stAppViewContainer']::before{animation:none}[class*='st-key-chatmsg-user'] [data-testid='stChatMessage'],[class*='st-key-chatmsg-asst'] [data-testid='stChatMessage']{backdrop-filter:blur(6px) saturate(120%)!important;-webkit-backdrop-filter:blur(6px) saturate(120%)!important}[data-testid='stSidebar']{backdrop-filter:blur(8px) saturate(120%);-webkit-backdrop-filter:blur(8px) saturate(120%)}}
@keyframes bubbleInRight{from{opacity:0;transform:translateX(14px) translateY(8px)}to{opacity:1;transform:translateX(0) translateY(0)}}
@keyframes bubbleInLeft{from{opacity:0;transform:translateX(-14px) translateY(8px)}to{opacity:1;transform:translateX(0) translateY(0)}}
.typing-dots{display:inline-flex;gap:5px;padding:6px 2px;align-items:center}
.typing-dots span{width:7px;height:7px;border-radius:50%;background:#9ea3b0;display:inline-block;animation:typingBounce 1.1s infinite ease-in-out}
.typing-dots span:nth-child(2){animation-delay:.18s}
.typing-dots span:nth-child(3){animation-delay:.36s}
@keyframes typingBounce{0%,60%,100%{opacity:.35;transform:translateY(0)}30%{opacity:1;transform:translateY(-5px)}}
.pill{animation:pillPulse .9s cubic-bezier(.16,1,.3,1) both}
@keyframes pillPulse{0%{opacity:0;transform:scale(.85)}55%{opacity:1;transform:scale(1.06)}100%{opacity:1;transform:scale(1)}}
.page-entry-1{animation:pageEntryFade .55s cubic-bezier(.16,1,.3,1) both}
.page-entry-2{animation:pageEntryFade .55s cubic-bezier(.16,1,.3,1) .12s both}
.page-entry-3{animation:pageEntryFade .55s cubic-bezier(.16,1,.3,1) .24s both}
@keyframes pageEntryFade{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.sidebar-entry [data-testid='stSidebar']{animation:sidebarSlide .5s cubic-bezier(.16,1,.3,1) both}
@keyframes sidebarSlide{from{transform:translateX(-30px);opacity:0}to{transform:translateX(0);opacity:1}}
@media (prefers-reduced-motion: reduce){.drill-section,.drill-section h3,[class*='st-key-chipwrap_'],[class*='st-key-chipwrap_'] button::after,[data-testid='stChatMessage'],[class*='st-key-chatmsg-user'] [data-testid='stChatMessage'],[class*='st-key-chatmsg-asst'] [data-testid='stChatMessage'],.typing-dots span,.pill,.page-entry-1,.page-entry-2,.page-entry-3,.sidebar-entry [data-testid='stSidebar'],[data-testid='stButton'] button,[data-testid='stDownloadButton'] button,[data-testid='stChatInput'],[data-testid='stAppViewContainer']::before,.agent-status--online .agent-status__dot{animation:none!important;opacity:1!important;transform:none!important;transition:none!important}[class*='st-key-chipwrap_'] button::after{transform:translateY(-50%)!important}}
#MainMenu,footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Session state helpers (chat only)
# ══════════════════════════════════════════════
def _session_id_from_url():
    try: sid = st.query_params.get("sid")
    except: return None
    if isinstance(sid, list): sid = sid[0] if sid else None
    return str(sid).strip() if sid and len(str(sid))<=128 else None

def _sync_session_url():
    try:
        if st.query_params.get("sid") != st.session_state.session_id:
            st.query_params["sid"] = st.session_state.session_id
    except: pass

def _adopt_url_session():
    url_id = _session_id_from_url()
    if not url_id or url_id == st.session_state.get("session_id"): return
    st.session_state["session_id"] = url_id
    st.session_state["conv_id"] = str(uuid.uuid4())[:8]
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"] = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"
INTENT_META = {
    "billing":{"label":"Billing","emoji":"💳"},"account":{"label":"Account","emoji":"🔐"},
    "shipping":{"label":"Shipping","emoji":"📦"},"technical":{"label":"Technical","emoji":"🛠️"},
    "general":{"label":"General","emoji":"💬"}
}
EXAMPLE_QUESTIONS = [
    ("🔐","How do I reset my password?"), ("📦","Where is my order?"),
    ("💳","How do I get a refund?"), ("🚫","How do I cancel my subscription?")
]

# Topical groupings for the start-page quick-question UI.
# Each category expands into 3 representative questions on click.
TOPIC_CATEGORIES = [
    ("logo/account_icon.png", "Account", [
        "How do I reset my password?",
        "How do I unlock my account?",
        "How do I enable two-factor authentication?",
    ]),
    ("logo/orders_icon.png", "Orders", [
        "Where is my order?",
        "How long does delivery take?",
        "Can I cancel my order after it has shipped?",
    ]),
    ("logo/refunds_icon.png", "Refunds", [
        "How do I get a refund?",
        "What is your refund policy?",
        "How long does a refund take to process?",
    ]),
    ("logo/subscription_icon.png", "Subscription", [
        "How do I cancel my subscription?",
        "How do I update my payment method?",
        "Where can I view my billing history?",
    ]),
]

# ──────────────────────────────────────────────────────────────────────────────
# Inject the icon-button background-image CSS once at module load. Static
# class names (.st-key-iconbtn_0 … _3) so this CSS doesn't need to re-emit
# on every rerun (the previous version embedded conv_id in the selector,
# forcing per-render re-injection that contributed to the 'reload' feel).
# ──────────────────────────────────────────────────────────────────────────────
_ICON_CSS_RULES = []
for _i, (_icon_path, _name, _qs) in enumerate(TOPIC_CATEGORIES):
    _b64 = _img_b64(_icon_path)
    _ICON_CSS_RULES.append(
        f".st-key-iconbtn_{_i} button{{"
        f"background-image:url('data:image/png;base64,{_b64}')!important;"
        f"background-position:center 8px!important;"
        f"background-size:160px auto!important;"
        f"background-repeat:no-repeat!important;"
        f"background-color:rgba(79,139,249,0.05)!important;"
        f"height:180px!important;"
        f"padding-top:152px!important;"
        f"padding-bottom:10px!important;"
        f"font-weight:600!important;"
        f"font-size:.95rem!important;"
        f"border:2px solid rgba(255,255,255,0.08)!important;"
        f"border-radius:14px!important;"
        f"}}"
        f".st-key-iconbtn_{_i} button:hover{{"
        f"background-color:rgba(79,139,249,0.18)!important;"
        f"border-color:#4F8BF9!important;"
        f"}}"
    )
st.markdown(f"<style>{''.join(_ICON_CSS_RULES)}</style>", unsafe_allow_html=True)

def _init_state():
    url_id = _session_id_from_url()
    for k,v in {"session_id":url_id or str(uuid.uuid4()),"conv_id":str(uuid.uuid4())[:8],
                "messages":[],"last_sources":[],"last_meta":{},"pending_query":None,
                "pending_append_user":True,"history_loaded_for":None}.items():
        if k not in st.session_state: st.session_state[k] = v
    st.session_state.pop("followups",None)

# ══════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════
@st.cache_data(ttl=15,show_spinner=False)
def api_health():
    for _ in range(HEALTH_RETRIES+1):
        try:
            r = requests.get(f"{API_URL}/health",timeout=HEALTH_TIMEOUT_S)
            if r.ok: return True
        except: pass
        time.sleep(2)
    return False

def call_chat(query):
    for attempt in range(2):
        try:
            r = requests.post(f"{API_URL}/chat",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers(),timeout=60)
            if r.ok: return r.json()
            if 500<=r.status_code<600 and attempt==0: time.sleep(3); continue
        except: time.sleep(3)
    return None

def call_chat_stream(query, output_box=None):
    parts,final,err = [],[],None
    try:
        with requests.post(f"{API_URL}/chat/stream",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers(),timeout=90,stream=True) as r:
            if not r.ok: return None
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                payload = line.removeprefix("data:").strip()
                if payload=="[DONE]": break
                try: ev = json.loads(payload)
                except: continue
                if "token" in ev:
                    parts.append(ev["token"])
                    if output_box: output_box.markdown("".join(parts))
                elif ev.get("event")=="final": final = ev
    except Exception as e: err = str(e)
    if final:
        final.setdefault("answer","".join(parts)); return final
    if parts:
        return {"answer":"".join(parts),"source_documents":[],"confidence":0.0,
                "condensed_query":query,"intent":"general","latency_ms":0}
    if err: st.error(f"Stream error: {err}")
    return None

def call_suggest(answer):
    try:
        r = requests.post(f"{API_URL}/suggest",json={"last_answer":answer,"n":3},headers=_api_headers(),timeout=25)
        if r.ok: return r.json().get("suggestions",[])
    except: pass
    return []

def call_feedback(q,a,rating):
    try:
        r = requests.post(f"{API_URL}/feedback",json={"session_id":st.session_state.session_id,"query":q,"answer":a,"rating":rating},headers=_api_headers(),timeout=10)
        return r.ok
    except: return False

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_admin(path):
    r = requests.get(f"{API_URL}{path}", headers=_api_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

# ══════════════════════════════════════════════
# Chat callbacks & render helpers
# ══════════════════════════════════════════════
def _queue_query(q):
    if q: st.session_state.pending_query = q; st.session_state.pending_append_user = True

def _on_chip_click(q):
    """on_click callback for chip buttons — sets pending query without an
    explicit st.rerun() so we save one rerun cycle (Streamlit reruns
    automatically after a widget callback)."""
    _queue_query(q)

def _on_icon_click(i, conv):
    """on_click callback for category icon buttons — toggles selected_topic
    and purges chip widget state. Cleared on every transition (toggle-off
    OR category switch) so a stale chip key cannot rebind to a new chip
    widget that happens to occupy the same script position."""
    selected = st.session_state.get("selected_topic")
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(f"chip_{conv}_"):
            del st.session_state[k]
    if selected == i:
        st.session_state.pop("selected_topic", None)
        return
    st.session_state["selected_topic"] = i

def _queue_regenerate(idx):
    msgs = st.session_state.messages
    if 0<idx<len(msgs) and msgs[idx]["role"]=="assistant" and msgs[idx-1]["role"]=="user":
        query = msgs[idx-1]["content"]; del msgs[idx]
        st.session_state.pending_query = query; st.session_state.pending_append_user = False

def _record_feedback(idx,rating):
    msgs = st.session_state.messages
    if idx<=0 or idx>=len(msgs): return
    call_feedback(msgs[idx-1]["content"],msgs[idx]["content"],rating)
    st.session_state[f"fb_{st.session_state.conv_id}_{idx}"] = rating

def confidence_class(c):
    if c>=0.75: return "pill-green"
    if c>=0.5: return "pill-amber"
    return "pill-red"

def render_meta(meta):
    if not meta: return
    intent = meta.get("intent","general")
    info = INTENT_META.get(intent,INTENT_META["general"])
    conf = float(meta.get("confidence",0))
    lat = int(meta.get("latency_ms",0))
    pc = confidence_class(conf)
    pills = (f'<span class="pill">{info["emoji"]} {info["label"]}</span>'
             f'<span class="pill {pc}">{int(conf*100)}% confidence</span>'
             f'<span class="pill pill-purple">⚡ {lat} ms</span>')
    st.markdown(f'<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">{pills}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="meter-wrap"><div class="meter-fill" style="width:{conf*100:.0f}%"></div></div>',unsafe_allow_html=True)

def _similarity_from_l2(l2): return max(0,min(1,1-l2**2/2))

def render_sources(sources):
    if not sources: return
    with st.expander(f"📚 Sources ({len(sources)})",expanded=False):
        for i,src in enumerate(sources[:4]):
            cls = "top" if i==0 else ""
            meta = src.get("metadata",{})
            bits = []
            if src.get("score") is not None: bits.append(f'similarity:{_similarity_from_l2(float(src["score"])):.2f}')
            if meta.get("topic"): bits.append(f'topic:{meta["topic"]}')
            if meta.get("source_query"): bits.append(f'matched q:{meta["source_query"][:60]}')
            line = " · ".join(bits) or "retrieved"
            rank = ["1st","2nd","3rd","4th"][i]
            st.markdown(f'<div class="src-card {cls}"><strong>{rank}</strong> — {src["content"][:220]}{"…" if len(src["content"])>220 else ""}<div class="src-meta">{line}</div></div>',unsafe_allow_html=True)

def build_transcript_md():
    lines = [f"# ChatSolveAI Conversation\n_Session `{st.session_state.session_id}` · exported {datetime.utcnow():%Y-%m-%d %H:%M UTC}_\n"]
    for m in st.session_state.messages:
        who = "**🧑 You**" if m["role"]=="user" else "**🤖 ChatSolveAI**"
        lines.append(f"{who}\n\n{m['content']}\n")
        if m["role"]=="assistant" and m.get("meta"):
            meta = m["meta"]
            lines.append(f"> intent:{meta.get('intent','general')} · confidence:{int(float(meta.get('confidence',0))*100)}% · latency:{meta.get('latency_ms',0)} ms\n")
    return "\n".join(lines)

def submit_query(query, append_user=True):
    if not query or not str(query).strip(): return False
    if append_user:
        st.session_state.messages.append({"role":"user","content":query})
    # Wrap each chat_message in a keyed st.container so the role is
    # baked into the wrapper class (st-key-chatmsg-user / -asst).
    # CSS targets these via [class*=...] for role-specific bubble
    # styling without relying on Streamlit's internal class names.
    with st.container(key="chatmsg-user-pending"):
        with st.chat_message("user",avatar=USER_AVATAR):
            st.markdown(query)
    with st.container(key="chatmsg-asst-pending"):
        with st.chat_message("assistant",avatar=ASSISTANT_AVATAR):
            box = st.empty()
            # Typing indicator until the first token arrives. call_chat_stream
            # overwrites this placeholder with streaming text on the first
            # output_box.markdown(...) call, so the dots vanish smoothly.
            box.markdown(
                '<div class="typing-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )
            result = call_chat_stream(query,box) if USE_STREAMING else call_chat(query)
            if not USE_STREAMING and result: box.markdown(result.get("answer",""))
    if not result:
        if append_user and st.session_state.messages and st.session_state.messages[-1]["role"]=="user" and st.session_state.messages[-1]["content"]==query:
            st.session_state.messages.pop()
        return False
    answer = result["answer"]
    sources = result.get("source_documents",[])
    meta = {
        "intent":result.get("intent","general"),"confidence":result.get("confidence",0),
        "latency_ms":result.get("latency_ms",0),"condensed_query":result.get("condensed_query",query)
    }
    # Follow-up suggestions removed from UI — skip the /suggest call to save
    # an LLM round-trip per turn.
    st.session_state.messages.append({
        "role":"assistant","content":answer,"sources":sources,"meta":meta,
    })
    st.session_state.last_sources = sources
    st.session_state.last_meta = meta
    return True

def _perform_full_reset():
    try: st.query_params.clear()
    except: pass
    api_health.clear()
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["conv_id"] = str(uuid.uuid4())[:8]
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"] = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True
    st.session_state["history_loaded_for"] = None
    st.session_state.pop("followups",None)
    st.session_state.pop("selected_topic",None)
    _sync_session_url()
    for k in list(st.session_state.keys()):
        if isinstance(k,str) and k.startswith(("fb_","up_","down_","fu_","chip_","topic_")):
            del st.session_state[k]

# ══════════════════════════════════════════════
# Sidebar — view selector + per-view extras
# ══════════════════════════════════════════════
NAV_CHAT, NAV_ADMIN = "💬 Chat", "📊 Admin Dashboard"

with st.sidebar:
    view = st.radio("View", [NAV_CHAT, NAV_ADMIN], key="nav_view", label_visibility="collapsed")
    st.divider()

# ══════════════════════════════════════════════
# Chat view
# ══════════════════════════════════════════════
def render_chat(sidebar_slot, main_slot):
    _init_state()
    _adopt_url_session()
    _sync_session_url()

    with sidebar_slot:
        st.image("logo/Logo.png", width=256)
        st.title("ChatSolveAI")
        st.divider()
        healthy = api_health()
        # Custom status indicator: pulsing green dot when the backend
        # health check passes, static red dot when not. Replaces the
        # default st.success / st.error pill so we get the 'alive'
        # animated look the design pass asked for.
        if healthy:
            st.markdown(
                '<div class="agent-status agent-status--online">'
                '<span class="agent-status__dot"></span>'
                '<span class="agent-status__label">Agent: Online</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="agent-status agent-status--offline" '
                f'title="API unreachable at {API_URL}">'
                '<span class="agent-status__dot"></span>'
                '<span class="agent-status__label">Agent: Offline</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Session Intelligence: aggregated per-conversation stats. Only
        # renders once we have at least one completed user+assistant
        # round; the sidebar starts compact and grows as the chat builds.
        msgs_now = st.session_state.messages
        assistant_metas = [
            m["meta"] for m in msgs_now
            if m["role"] == "assistant" and m.get("meta")
        ]
        if assistant_metas:
            user_count = sum(1 for m in msgs_now if m["role"] == "user")
            avg_conf = sum(float(x.get("confidence", 0)) for x in assistant_metas) / len(assistant_metas)
            st.markdown('<div class="session-stats">', unsafe_allow_html=True)
            st.markdown(
                '<div class="session-stats__title">Session Intelligence</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Messages", user_count)
            with c2:
                st.metric("Avg confidence", f"{avg_conf*100:.0f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        if st.button("🗑 New chat", key="btn_new_chat", use_container_width=True):
            _perform_full_reset()
            st.rerun()
        if st.session_state.messages:
            st.download_button("⬇️ Export chat (.md)", data=build_transcript_md(),
                               file_name=f"chatsolveai_{st.session_state.session_id[:8]}.md",
                               mime="text/markdown", use_container_width=True)

    # First-render flag drives the page-entry stagger animations (#3.F)
    # and the sidebar slide-in (#3.I). The flag flips False after the
    # very first render in this session, so subsequent reruns don't
    # replay the entry animations.
    is_first_render = not st.session_state.get("_initial_render_seen", False)
    st.session_state["_initial_render_seen"] = True

    if is_first_render:
        # One-shot CSS rule: applies the sidebar slide-in animation on
        # the very first render only. Streamlit's normal markdown DOM
        # will be replaced on subsequent renders, dropping this rule.
        st.markdown(
            "<style>[data-testid='stSidebar']{"
            "animation:sidebarSlide .5s cubic-bezier(.16,1,.3,1) both}</style>",
            unsafe_allow_html=True,
        )

    with main_slot:
        # Page-entry stagger classes only attach on first render.
        hero_cls = "hero-title page-entry-1" if is_first_render else "hero-title"
        sub_cls = "hero-sub page-entry-2" if is_first_render else "hero-sub"
        st.markdown(
            f'<div class="{hero_cls}">💬 ChatSolveAI — Customer Support</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="{sub_cls}">AI-powered customer support assistant. '
            f'Get instant answers about orders, refunds, billing, account, '
            f'and subscriptions — backed by a real-time knowledge base.</p>',
            unsafe_allow_html=True,
        )

        conv = st.session_state.conv_id
        msgs = st.session_state.messages
        asked = {m["content"] for m in msgs if m["role"] == "user"}
        has_pending = bool(st.session_state.pending_query)

        selected = st.session_state.get("selected_topic")

        # SCRIPT-POSITION STABILITY: every block below this point must
        # exist at a constant script index across reruns. Streamlit
        # assigns positional element IDs based on call order, and when
        # script positions shift between runs (e.g., a conditional
        # `st.markdown(<style>)` appearing or disappearing), Streamlit
        # Cloud's reconciler can mount the new element at a fresh ID
        # without unmounting the old one — surfacing the duplicate
        # icon-row + drill-section ghosts seen on category click.
        #
        # Emit ONE always-present <style> tag with rule contents that
        # vary by state. The element identity stays constant; only its
        # innerHTML changes. Same trick for the greeting div.
        focus_css = ""
        if selected is not None:
            focus_css = (
                f"[class*='st-key-iconbtn_'] button{{"
                f"opacity:.4!important;transform:scale(.95)!important"
                f"}}"
                f"[class*='st-key-iconbtn_'] button:hover{{"
                f"opacity:.7!important;transform:scale(.97)!important"
                f"}}"
                f".st-key-iconbtn_{selected} button{{"
                f"opacity:1!important;"
                f"transform:scale(1.05)!important;"
                f"border-color:#4F8BF9!important;"
                f"background-color:rgba(79,139,249,0.18)!important;"
                f"box-shadow:0 0 0 2px rgba(79,139,249,.30),"
                f"0 0 28px rgba(79,139,249,.30)!important"
                f"}}"
                f".st-key-iconbtn_{selected} button:hover{{"
                f"transform:scale(1.07)!important"
                f"}}"
            )
        pending_css = ""
        if has_pending:
            pending_css = (
                "[data-testid='stChatInput']{"
                "animation:inputPulse 1.4s ease-in-out infinite}"
            )
        # Always-present state-conditional rules. Single emission, fixed
        # position. Lottie centering + transparent background lives in
        # the global stylesheet (targets .st-key-lottie_wrap).
        st.markdown(
            f"<style>{focus_css}{pending_css}</style>",
            unsafe_allow_html=True,
        )

        # Greeting renders inside an always-present keyed wrapper so
        # toggling msgs ↔ no-msgs doesn't shift the icon row's script
        # position. Wrapper stays mounted; inner content swaps.
        with st.container(key="greeting_block"):
            if not msgs:
                greet_cls = "page-entry-3" if is_first_render else ""
                st.markdown(
                    f'<div class="{greet_cls}"><strong>'
                    f'👋 What do you need help with?</strong></div>',
                    unsafe_allow_html=True,
                )

        # Stable-key wrapper anchors the icon row at a fixed logical
        # position so Streamlit's reconciler matches by key on reruns
        # rather than by sibling index. Without this anchor, when the
        # next sibling block changes shape (drill collapses → Lottie
        # appears, or vice versa), Streamlit Cloud's reconciler can
        # partial-merge old `st.columns(4)` icon DOM into the new
        # neighboring layout and leak ghost icons into a phantom row.
        with st.container(key="icon_row"):
            icon_cols = st.columns(4)
            for i, (icon_path, name, _qs) in enumerate(TOPIC_CATEGORIES):
                with icon_cols[i]:
                    st.button(name,
                              key=f"iconbtn_{i}",
                              help=name,
                              use_container_width=True,
                              on_click=_on_icon_click,
                              args=(i, conv))

        # Empty-state Lottie + drill section share an `st.empty()` slot
        # pattern. The slot is the canonical Streamlit primitive for
        # swapping a single child cleanly: calling `slot.container()`
        # replaces any prior child; calling `slot.empty()` removes it.
        # Stable script position; explicit unmount semantics. State-
        # keyed `st.container(...)` was insufficient on Streamlit Cloud
        # because component iframes (st_lottie) and chip widgets stayed
        # mounted past the parent's key change, leaving ghosts.
        empty_state = (not msgs) and (selected is None) and (not has_pending)
        empty_slot = st.empty()
        if empty_state and _HAS_LOTTIE:
            try:
                lottie_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "logo", "empty_state.json",
                )
                anim = _lottie_data(lottie_path)
                with empty_slot.container():
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        with st.container(key="lottie_wrap"):
                            # `streamlit_lottie.st_lottie` exposes no
                            # bg_color / transparency option, and its
                            # iframe content paints an opaque body.
                            # Render via the `<lottie-player>` web
                            # component instead — `background="transparent"`
                            # is honored, the iframe document body is
                            # itself transparent, and we keep full
                            # control over sizing.
                            anim_json = json.dumps(anim)
                            components.html(
                                f"""<!doctype html>
                                <html><head>
                                  <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
                                  <style>
                                    html,body{{margin:0;padding:0;background:transparent!important}}
                                    lottie-player{{display:block;margin:0 auto;width:100%;max-width:360px;height:220px;background:transparent!important}}
                                  </style>
                                </head><body>
                                  <lottie-player background="transparent" speed="1" loop autoplay></lottie-player>
                                  <script>
                                    const p = document.querySelector('lottie-player');
                                    p.load({anim_json});
                                  </script>
                                </body></html>""",
                                height=230,
                            )
                        st.markdown(
                            "<div style='text-align:center;color:#7a8190;"
                            "font-size:.85rem;margin-top:-10px'>"
                            "Pick a category above or type a question below to get started"
                            "</div>",
                            unsafe_allow_html=True,
                        )
            except Exception:
                empty_slot.empty()
        else:
            empty_slot.empty()

        drill_slot = st.empty()
        if selected is not None:
            _icon, sel_name, sel_questions = TOPIC_CATEGORIES[selected]
            remaining = [(j, q) for j, q in enumerate(sel_questions) if q not in asked]
            for j, q in enumerate(sel_questions):
                if q in asked:
                    chip_key = f"chip_{conv}_{selected}_{j}"
                    if chip_key in st.session_state:
                        del st.session_state[chip_key]
            with drill_slot.container():
                st.markdown(
                    f'<div class="drill-section"><h3>{sel_name}</h3></div>',
                    unsafe_allow_html=True,
                )
                # Pre-allocated chip slots: each TOPIC_CATEGORIES entry
                # has exactly 3 questions, so 3 fixed-position st.empty()
                # slots cover every state. Fill the first len(remaining)
                # slots with chips; explicitly clear the rest. Without
                # the fixed allocation, dynamic chip-count reruns left
                # ghost chip widgets in DOM on Streamlit Cloud (the
                # most-recently-asked question's slot got reused by the
                # next chip, so the prior chip text lingered as a
                # phantom row beneath the live ones).
                MAX_CHIPS = 3
                chip_slots = [st.empty() for _ in range(MAX_CHIPS)]
                if remaining:
                    # Build per-chip animation-delay rules in a single
                    # <style> tag emitted at a fixed script position
                    # (inside drill_slot.container, which itself is at
                    # a stable index above). One emission, no per-chip
                    # script-position drift. CSS :nth-of-type cannot
                    # work because Streamlit wraps each chip in its own
                    # vertical-block container, so chips are not direct
                    # siblings — keying by wrap_key sidesteps that.
                    delay_rules = []
                    for idx, (j, _q) in enumerate(remaining):
                        wrap_key = f"chipwrap_{conv}_{selected}_{j}_{len(asked)}"
                        delay = 0.10 + (idx * 0.15)
                        delay_rules.append(
                            f".st-key-{wrap_key}{{"
                            f"animation:slideUpFade .45s cubic-bezier(.22,1,.36,1) {delay:.3f}s both"
                            f"}}"
                        )
                    st.markdown(
                        f"<style>{''.join(delay_rules)}</style>",
                        unsafe_allow_html=True,
                    )
                    for slot, (j, q) in zip(chip_slots, remaining):
                        with slot.container():
                            wrap_key = f"chipwrap_{conv}_{selected}_{j}_{len(asked)}"
                            with st.container(key=wrap_key):
                                st.button(q,
                                          key=f"chip_{conv}_{selected}_{j}_{len(asked)}",
                                          use_container_width=True,
                                          disabled=has_pending,
                                          on_click=_on_chip_click,
                                          args=(q,))
                    for slot in chip_slots[len(remaining):]:
                        slot.empty()
                else:
                    for slot in chip_slots:
                        slot.empty()
                    st.caption("All questions in this topic asked.")
        else:
            drill_slot.empty()

        if msgs or has_pending:
            with st.container(height=520):
                for idx, msg in enumerate(msgs):
                    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
                    role_short = "user" if msg["role"] == "user" else "asst"
                    # Keyed wrapper container so the role ('user' / 'asst')
                    # ends up in the wrapper's CSS class (st-key-chatmsg-...);
                    # the global stylesheet uses that to apply per-role
                    # bubble background, border, and slide-in direction.
                    with st.container(key=f"chatmsg-{role_short}-{conv}-{idx}"):
                        with st.chat_message(msg["role"], avatar=avatar):
                            st.markdown(msg["content"])
                            if msg["role"] == "assistant":
                                render_meta(msg.get("meta", {}))
                                render_sources(msg.get("sources", []))
                                fb = f"fb_{conv}_{idx}"
                                if st.session_state.get(fb) is None:
                                    c1, c2, _ = st.columns([1, 1, 8])
                                    with c1:
                                        if st.button("👍", key=f"up_{conv}_{idx}"):
                                            _record_feedback(idx, "up"); st.rerun()
                                    with c2:
                                        if st.button("👎", key=f"down_{conv}_{idx}"):
                                            _record_feedback(idx, "down"); st.rerun()
                                else:
                                    rating = st.session_state[fb]
                                    st.caption(f"You rated: {'👍' if rating == 'up' else '👎'}")
                                    if rating == "down" and st.button("Regenerate", key=f"regen_{conv}_{idx}"):
                                        _queue_regenerate(idx); del st.session_state[fb]; st.rerun()

                if has_pending:
                    if healthy:
                        q = st.session_state.pending_query
                        append = st.session_state.get("pending_append_user", True)
                        st.session_state.pending_query = None
                        st.session_state.pending_append_user = True
                        if submit_query(q, append_user=append):
                            st.rerun()
                    else:
                        st.warning("Backend cold‑starting … try again in ~30s.")
                        st.session_state.pending_query = None

            # Auto-scroll the chat container to the newest message after
            # any rerun that has chat content. Runs in a 0-height iframe;
            # accesses the parent Streamlit document via window.parent and
            # smooth-scrolls every scrollable vertical block to its bottom.
            # No-op if Streamlit Cloud's iframe sandbox blocks parent
            # access — in that case it just doesn't scroll, no error.
            components.html(
                """<script>
                setTimeout(() => {
                  try {
                    const root = window.parent.document;
                    const blocks = root.querySelectorAll(
                      '[data-testid="stVerticalBlockBorderWrapper"]'
                    );
                    blocks.forEach(b => {
                      if (b.scrollHeight > b.clientHeight) {
                        b.scrollTo({ top: b.scrollHeight, behavior: 'smooth' });
                      }
                    });
                  } catch (e) {}
                }, 120);
                </script>""",
                height=0,
            )

        # Copy-last-answer button. Renders only when the most recent
        # message is from the assistant. Wrapped in an st.empty() slot
        # so the iframe component is explicitly removed from DOM when
        # the condition flips false (no chat yet / New chat reset /
        # pending stream). Without the explicit clear, Streamlit Cloud
        # left the prior iframe mounted as a ghost beneath the empty-
        # state Lottie.
        copy_slot = st.empty()
        if msgs and msgs[-1]["role"] == "assistant" and not has_pending:
            last_answer = msgs[-1].get("content", "") or ""
            payload = json.dumps(last_answer)
            with copy_slot.container():
                components.html(
                    f"""<!doctype html>
                    <html><head><style>
                      body {{margin:0;padding:0;background:transparent;
                             font-family:'Inter',-apple-system,system-ui,sans-serif}}
                      .copy-btn {{
                        display:inline-flex;align-items:center;gap:6px;
                        padding:6px 12px;border-radius:10px;
                        background:rgba(79,139,249,.10);
                        border:1px solid rgba(79,139,249,.25);
                        color:#cdd5e0;font-size:.78rem;font-weight:500;
                        cursor:pointer;
                        transition:background-color .2s ease-in-out,
                                   border-color .2s ease-in-out,
                                   transform .12s ease-in-out;
                      }}
                      .copy-btn:hover {{
                        background:rgba(79,139,249,.20);
                        border-color:#4F8BF9;
                        transform:translateY(-1px);
                      }}
                      .copy-btn.copied {{
                        background:rgba(52,211,153,.18);
                        border-color:#34d399;color:#a7f3d0;
                      }}
                      .copy-btn .icon {{font-size:.95rem;line-height:1}}
                    </style></head>
                    <body>
                      <button id="cb" class="copy-btn" type="button">
                        <span class="icon">📋</span><span class="lbl">Copy answer</span>
                      </button>
                      <script>
                        const btn = document.getElementById('cb');
                        const lbl = btn.querySelector('.lbl');
                        const icon = btn.querySelector('.icon');
                        const TEXT = {payload};
                        btn.addEventListener('click', async () => {{
                          try {{
                            await navigator.clipboard.writeText(TEXT);
                          }} catch (e) {{
                            const ta = document.createElement('textarea');
                            ta.value = TEXT; ta.style.position='fixed'; ta.style.opacity='0';
                            document.body.appendChild(ta); ta.select();
                            try {{ document.execCommand('copy'); }} catch (_) {{}}
                            document.body.removeChild(ta);
                          }}
                          btn.classList.add('copied');
                          icon.textContent = '✓';
                          lbl.textContent = 'Copied';
                          setTimeout(() => {{
                            btn.classList.remove('copied');
                            icon.textContent = '📋';
                            lbl.textContent = 'Copy answer';
                          }}, 1500);
                        }});
                      </script>
                    </body></html>""",
                    height=44,
                )
        else:
            copy_slot.empty()

        if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
            if not healthy:
                st.error("API unreachable.")
                st.stop()
            _queue_query(prompt)
            st.rerun()

# ══════════════════════════════════════════════
# Admin view
# ══════════════════════════════════════════════
def render_admin(sidebar_slot, main_slot):
    # Sidebar additions (only Sign-out when password gate is enabled).
    if ADMIN_PASSWORD and st.session_state.get("admin_ok"):
        with sidebar_slot:
            if st.button("Sign out", key="admin_signout"):
                st.session_state.pop("admin_ok", None)
                st.rerun()

    with main_slot:
        # Optional password gate
        if ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
            with st.form("admin_login"):
                pw = st.text_input("Admin password", type="password")
                if st.form_submit_button("Sign in") and pw == ADMIN_PASSWORD:
                    st.session_state["admin_ok"] = True
                    st.rerun()
                if pw and pw != ADMIN_PASSWORD:
                    st.error("Invalid password.")
            return

        try:
            summary    = _fetch_admin("/analytics")
            timeseries = _fetch_admin("/analytics/timeseries?days=14")
            intents    = _fetch_admin("/analytics/intents")
            latency    = _fetch_admin("/analytics/latency")
            feedback   = _fetch_admin("/analytics/feedback")
            sessions   = _fetch_admin("/sessions?limit=20")
        except Exception:
            st.warning("Backend unreachable – analytics unavailable.")
            return

        st.title("ChatSolveAI Admin Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions", summary.get("total_sessions", 0))
        c2.metric("Queries", summary.get("total_queries", 0))
        c3.metric("Today", summary.get("queries_today", 0))
        c4.metric("Avg Session Length", summary.get("avg_session_length", 0))
        f1, f2, f3 = st.columns(3)
        f1.metric("P95 Latency", f"{latency.get('p95', 0)} ms")
        f2.metric("👍 Upvotes", feedback.get("up", 0))
        f3.metric("👎 Downvotes", feedback.get("down", 0))
        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Daily Queries")
            # Guard against all-zero counts: Vega emits 'Infinite extent'
            # warnings when every count value is 0 because it can't compute
            # a non-degenerate scale. Only render the chart if at least one
            # row has a non-zero count.
            if timeseries and any(r.get("count", 0) > 0 for r in timeseries):
                st.bar_chart(timeseries, x="date", y="count")
            else:
                st.info("No data yet.")
        with right:
            st.subheader("Intent Distribution")
            if intents and any(r.get("count", 0) > 0 for r in intents):
                st.bar_chart(intents, x="intent", y="count")
            else:
                st.info("No data yet.")
        st.divider()
        tab1, tab2 = st.tabs(["Top Questions", "Recent Sessions"])
        with tab1:
            top = summary.get("top_questions", [])
            if top: st.dataframe(top, use_container_width=True, hide_index=True)
            else: st.info("No questions logged.")
        with tab2:
            if sessions: st.dataframe(sessions, use_container_width=True, hide_index=True)
            else: st.info("No sessions.")
        st.caption(f"Last refreshed: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")

# ══════════════════════════════════════════════
# Dispatch
# ──────────────────────────────────────────────
# Strategy: ALWAYS create both views' containers with stable keys, but
# only fill the active one. Stable keys mean the container element
# identity is preserved across reruns; when the inactive container has
# no children in the new run, Streamlit reconciles its prior children
# away (old charts / tabs / etc. are removed).
#
# CSS hide on the inactive container is a visual fallback. st.chat_input
# docks to the body so we also hide it via testid when on admin.
# ══════════════════════════════════════════════
with st.sidebar:
    chat_sb_ctx  = st.container(key="sb_chat")
    admin_sb_ctx = st.container(key="sb_admin")
chat_main_ctx  = st.container(key="main_chat")
admin_main_ctx = st.container(key="main_admin")

_inactive = "admin" if view == NAV_CHAT else "chat"
_extra = "[data-testid='stChatInput']{display:none !important;}" if view == NAV_ADMIN else ""
st.markdown(
    f"<style>"
    f"[class*='st-key-sb_{_inactive}'],"
    f"[class*='st-key-main_{_inactive}']{{display:none !important;}}"
    f"{_extra}"
    f"</style>",
    unsafe_allow_html=True,
)

if view == NAV_CHAT:
    render_chat(chat_sb_ctx, chat_main_ctx)
else:
    render_admin(admin_sb_ctx, admin_main_ctx)
