"""
ChatSolveAI — Streamlit frontend (v3.0) — single page with sidebar nav.

Replaces the previous multipage layout (App.py + pages/1_Admin_Dashboard.py).
Streamlit's pages/ directory caused DOM remnants to persist across page swaps;
folding both views into one script eliminates that class of bug entirely.
"""

from __future__ import annotations
import base64, hmac, html, json, os, threading, time, uuid
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
    # SECURITY: callers must pass a hard-coded asset path under logo/.
    # Never pass user-supplied paths — open() would happily serve any
    # readable file via the base64-in-CSS channel.
    # mtime in the lru_cache key busts the cache when an asset changes.
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
except Exception: pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""
ADMIN_PASSWORD = _SECRET_ADMIN_PW or os.getenv("ADMIN_PASSWORD") or ""
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT","20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES","2"))
USE_STREAMING    = os.getenv("USE_STREAMING","true").lower() not in {"0","false","no"}

def _api_headers(): return {"X-API-Key": API_KEY} if API_KEY else {}

# Sentry: production error tracking. No-op when DSN is not set, so
# local dev never tries to report anywhere. `traces_sample_rate=0.05`
# captures 5% of transactions so the free tier doesn't burn through.
try:
    import sentry_sdk
    _SENTRY_DSN = None
    try:
        _SENTRY_DSN = st.secrets.get("SENTRY_DSN")
    except Exception:
        pass
    _SENTRY_DSN = _SENTRY_DSN or os.getenv("SENTRY_DSN")
    if _SENTRY_DSN:
        sentry_sdk.init(
            dsn=_SENTRY_DSN,
            traces_sample_rate=0.05,
            send_default_pii=False,
            release=os.getenv("APP_VERSION", "dev"),
        )
except ImportError:
    pass

# ══════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════
st.set_page_config(page_title="Customer Support AI", page_icon="logo/Logo.png", layout="wide",
                   initial_sidebar_state="expanded")

# Anti-flash: paint a dark background on <html>/<body> BEFORE the main
# stylesheet (with its fonts + heavy selectors) parses, so first paint
# never shows the default white. Tiny inline style; emitted at fixed
# top-of-script position so the element identity is stable across reruns.
st.markdown(
    "<style>html,body,[data-testid='stApp']{background-color:#0E1117}</style>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Theme persistence: read `?theme=light|dark` from URL on the first render of
# a session, so a returning user (or a shared deep-link) lands in their
# preferred mode. Must run BEFORE any toggle widget is created so its
# default state matches the URL.
# ──────────────────────────────────────────────────────────────────────────────
if "_theme_initialized" not in st.session_state:
    try:
        _qp_theme = st.query_params.get("theme")
        if _qp_theme in ("light", "dark"):
            st.session_state["_theme_light"] = (_qp_theme == "light")
    except Exception:
        pass
    st.session_state["_theme_initialized"] = True

def _on_theme_toggle():
    """on_change callback for the Light/Dark toggle. Writes the new
    state back to the URL so a refresh preserves it. Same callback is
    wired to both render_chat and render_admin's toggle widgets."""
    try:
        st.query_params["theme"] = "light" if st.session_state.get("_theme_light") else "dark"
    except Exception:
        pass

# View persistence: `?view=admin|chat` deep-link. Init runs once per session,
# before the nav radio is created so its default state matches the URL.
NAV_CHAT, NAV_ADMIN = "💬 Chat", "📊 Admin Dashboard"
if "_view_initialized" not in st.session_state:
    try:
        _qp_view = st.query_params.get("view")
        if _qp_view == "admin":
            st.session_state["nav_view"] = NAV_ADMIN
        elif _qp_view == "chat":
            st.session_state["nav_view"] = NAV_CHAT
    except Exception:
        pass
    st.session_state["_view_initialized"] = True

def _on_view_change():
    try:
        st.query_params["view"] = "admin" if st.session_state.get("nav_view") == NAV_ADMIN else "chat"
    except Exception:
        pass

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
[data-testid='stChatInput'] textarea,[data-baseweb='input'] input{transition:all .3s ease!important;font-size:15px!important;line-height:1.55!important}
[data-testid='stChatInput']:focus-within,[data-baseweb='input']:focus-within{box-shadow:0 0 15px rgba(79,139,249,.20)!important;border-color:#4F8BF9!important;border-radius:14px!important}
[data-testid='stChatInput'] textarea:focus,[data-baseweb='input'] input:focus{outline:none!important}
[data-testid='stMain'] .block-container{max-width:1100px!important;margin-left:auto!important;margin-right:auto!important}
/* Admin's wider 1500px max-width is emitted conditionally in the
 * view-dispatch block at the bottom of App.py. Using `:has()` here
 * caused the admin width to leak into the chat view because
 * `.st-key-admin_grid` stays mounted (with display:none from the
 * ghost-css) even after the user switches back — `:has()` matches
 * DOM presence regardless of visibility. */
@media (max-width:900px){.st-key-admin_grid [data-testid='stHorizontalBlock']{flex-direction:column!important;gap:1rem!important}.st-key-admin_grid [data-testid='stHorizontalBlock'] [data-testid='stColumn']{width:100%!important;flex:1 1 100%!important;min-width:0!important}.st-key-admin_grid [data-testid='stMetric']{padding:12px 14px!important}.st-key-admin_grid [data-testid='stMetricValue']{font-size:1.4rem!important}}
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
[data-testid='StyledFullScreenButton'],[data-testid='stFullScreenButton'],[data-testid='stElementToolbar'],[data-testid='stHeaderActionElements']{display:none!important}
.vega-actions,details.vega-actions,.vega-embed details,.vega-embed summary,.vega-bindings{display:none!important}
.stMarkdown h1 a,.stMarkdown h2 a,.stMarkdown h3 a,.stMarkdown h4 a,.drill-section h3 a,h1 .anchor-link,h2 .anchor-link,h3 .anchor-link{display:none!important}
[data-testid='stSidebar']{background:rgba(20,24,32,.65)!important;backdrop-filter:blur(14px) saturate(140%);-webkit-backdrop-filter:blur(14px) saturate(140%);border-right:1px solid rgba(255,255,255,.06)}
[data-testid='stSidebar'] > div{background:transparent!important}
[data-testid='stSidebarNav']{display:block!important;visibility:visible!important;opacity:1!important;padding:.5rem .25rem 1rem!important;border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:.5rem}
[data-testid='stSidebarNav'] ul li a{color:#cdd5e0!important;font-weight:500;padding:8px 12px;border-radius:8px;transition:all .2s ease;display:flex;align-items:center;gap:8px}
[data-testid='stSidebarNav'] ul li a:hover{color:#fff!important;background:rgba(79,139,249,.15)!important}
[data-testid='stSidebarNav'] ul li a[aria-current='page']{color:#fff!important;background:rgba(79,139,249,.22)!important;border-left:2px solid #4F8BF9}
.nav-links{display:flex;flex-direction:column;gap:4px;margin:8px 0 4px}
.nav-links .nav-link{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;color:#cdd5e0!important;font-weight:500;font-size:.92rem;text-decoration:none!important;border:1px solid transparent;transition:all .2s ease}
.nav-links .nav-link:hover{background:rgba(79,139,249,.12)!important;color:#fff!important;border-color:rgba(79,139,249,.25)}
.nav-links .nav-link--active{background:rgba(79,139,249,.18)!important;color:#fff!important;border-color:rgba(79,139,249,.40);box-shadow:inset 3px 0 0 #4F8BF9}
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
/* Sidebar cohesion: agent-status pill, new-chat button, and theme toggle
 * sit inside a single visual group via a soft inset box-shadow on their
 * common ancestor (st-key-view_sb_chat / _admin container). Pure CSS,
 * no DOM additions. */
[class*='st-key-view_sb_chat'] .agent-status,[class*='st-key-view_sb_chat'] [class*='st-key-btn_new_chat']{box-shadow:0 1px 0 rgba(255,255,255,.02) inset,0 1px 2px rgba(0,0,0,.10)}
/* Subtle native-tooltip affordance on source cards: cursor change + the
 * browser's built-in title= popup carries the full passage. No JS, no
 * iframe, no ghost surface. */
.src-card[title]{cursor:help}
/* prefers-contrast: bumps border opacity so bubbles + cards stay clearly
 * separated for users who request stronger contrast in OS settings. */
@media (prefers-contrast: more){
  .src-card{border-color:rgba(255,255,255,.22)!important}
  [data-testid='stChatMessage']{border-color:rgba(255,255,255,.24)!important}
  [class*='st-key-chatmsg-user'] [data-testid='stChatMessage']{border-color:rgba(79,139,249,.55)!important}
  [class*='st-key-chatmsg-asst'] [data-testid='stChatMessage']{border-color:rgba(255,255,255,.30)!important}
  .pill{outline:1px solid rgba(255,255,255,.20)}
}
/* prefers-reduced-data: drop the animated mesh background entirely on
 * data-saver connections — the radial-gradient layers cost paint cycles
 * for purely decorative motion. */
@media (prefers-reduced-data: reduce){
  [data-testid='stAppViewContainer']::before{animation:none!important;background:linear-gradient(180deg,#0E1117 0%,#161A23 60%,#1E293B 100%)!important}
}
/* Smooth theme transition: when the light-mode CSS block injects or
 * retracts, animate the swap on top-level surfaces. Selector list is
 * narrow so we don't accidentally fade interactive states. */
[data-testid='stApp'],[data-testid='stSidebar'],[data-testid='stChatMessage'],.src-card,.agent-status,[data-testid='stChatInput'],[data-testid='stChatInput'] textarea,.pill,[data-testid='stHeader']{
  transition:background-color .25s ease, color .25s ease, border-color .25s ease, box-shadow .25s ease !important;
}
/* Focus-visible polish: keyboard nav gets a clear 2px accent ring on
 * every interactive surface. Mouse clicks don't trigger this (uses
 * :focus-visible, not :focus), so it stays out of the way for pointer
 * users while remaining accessible. */
[class*='st-key-iconbtn_'] button:focus-visible,[class*='st-key-chipwrap_'] button:focus-visible,[class*='st-key-up_'] button:focus-visible,[class*='st-key-down_'] button:focus-visible,[class*='st-key-regen_'] button:focus-visible,[class*='st-key-edit_user_'] button:focus-visible,[class*='st-key-admin_signout'] button:focus-visible,[data-testid='stChatInput'] button:focus-visible,[data-testid='stCodeCopyButton']:focus-visible,[data-testid='stExpander'] summary:focus-visible{
  outline:2px solid #4F8BF9 !important;outline-offset:3px !important;border-radius:10px !important;
}
/* Mobile safe-area: respect iPhone home-bar / notch by padding the
 * chat input. env() falls back to 8px when no safe-area is exposed. */
[data-testid='stChatInput']{padding-bottom:max(0px,env(safe-area-inset-bottom))}
@media (max-width: 768px){
  [data-testid='stChatInput']{margin-bottom:max(8px,env(safe-area-inset-bottom))!important}
}
/* Connection quality classes on the agent-status pill. Three colors
 * grade the dot by recent p50 latency: <1.2s green, <3s amber, else red.
 * Reuses existing .agent-status--online keyframe; only the dot color +
 * pill text accent changes. */
.agent-status--medium .agent-status__dot{background:#facc15;box-shadow:0 0 0 3px rgba(250,204,21,.20),0 0 10px rgba(250,204,21,.55);animation:dotPulse 1.6s ease-in-out infinite}
.agent-status--medium .agent-status__label{color:#fde68a}
.agent-status--poor .agent-status__dot{background:#fb923c;box-shadow:0 0 0 3px rgba(251,146,60,.20),0 0 10px rgba(251,146,60,.55);animation:dotPulse 1.6s ease-in-out infinite}
.agent-status--poor .agent-status__label{color:#fed7aa}
/* Per-message animation gate: when a previously-rendered message is
 * re-rendered (e.g., during a streaming rerun), suppress the slide-in
 * keyframe so only the new bubble animates. The Python layer sets
 * .msg-static on every container whose index already appears in
 * _animated_msgs; first-time mounts get the default animation. */
[class*='st-key-chatmsg-'] .msg-static[data-testid='stChatMessage'],[class*='st-key-chatmsg-'].msg-static [data-testid='stChatMessage']{animation:none!important}
/* Edit-pencil button on user bubbles: tiny chromeless 24px square,
 * appears top-right of the bubble on hover. Reuses existing focus
 * outline + transition tokens. */
[class*='st-key-edit_user_'] button{
  min-height:24px!important;height:24px!important;padding:0 8px!important;
  font-size:.72rem!important;background:transparent!important;
  border:1px solid rgba(255,255,255,.10)!important;color:#9ea3b0!important;
  border-radius:8px!important;
}
[class*='st-key-edit_user_'] button:hover{
  background:rgba(79,139,249,.12)!important;border-color:#4F8BF9!important;color:#E5E7EB!important;
}
/* Conversation summary chip row */
.conv-summary{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 12px;align-items:center}
.conv-summary .pill{margin:0}
.conv-summary .label{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:#7a8190;font-weight:600;margin-right:4px}
/* Scroll-to-bottom floating button (injected client-side inside the
 * existing scroll iframe). Pure parent-DOM CSS so it inherits theme. */
.cs-jump-btn{
  position:fixed;right:24px;bottom:96px;z-index:9000;
  display:none;align-items:center;gap:6px;
  padding:8px 14px;border-radius:999px;
  background:rgba(79,139,249,.92);color:#fff;font-weight:600;font-size:.82rem;
  border:1px solid rgba(255,255,255,.20);
  box-shadow:0 8px 24px rgba(79,139,249,.35),0 2px 6px rgba(0,0,0,.25);
  cursor:pointer;transition:transform .18s ease, box-shadow .2s ease, opacity .2s ease;
  opacity:0;pointer-events:none;font-family:'Inter',system-ui,sans-serif;
}
.cs-jump-btn.cs-show{display:inline-flex;opacity:1;pointer-events:auto}
.cs-jump-btn:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(79,139,249,.45)}
.cs-jump-btn:active{transform:translateY(0) scale(.97)}
/* Network online/offline banner (injected client-side at module scope). */
.cs-net-banner{
  position:fixed;top:0;left:0;right:0;z-index:99999;
  padding:8px 16px;text-align:center;font-size:.84rem;font-weight:600;
  font-family:'Inter',system-ui,sans-serif;letter-spacing:.01em;
  transform:translateY(-100%);transition:transform .25s ease, background-color .25s ease;
  box-shadow:0 4px 12px rgba(0,0,0,.18);
}
.cs-net-banner.cs-show{transform:translateY(0)}
.cs-net-banner.cs-offline{background:#fb923c;color:#1c1917}
.cs-net-banner.cs-online{background:#34d399;color:#053b27}
/* Recent-question resume card (injected client-side from localStorage). */
.cs-resume-card{
  display:block;width:100%;max-width:800px;margin:8px auto 14px;
  padding:14px 16px;border-radius:14px;
  background:rgba(142,107,255,.08);border:1px solid rgba(142,107,255,.22);
  color:#cdd5e0;font-family:'Inter',system-ui,sans-serif;text-align:left;cursor:pointer;
  transition:transform .15s ease, background-color .2s ease, border-color .2s ease, box-shadow .2s ease;
}
.cs-resume-card:hover{
  background:rgba(142,107,255,.14);border-color:#8E6BFF;transform:translateY(-2px);
  box-shadow:0 8px 22px rgba(142,107,255,.22);
}
.cs-resume-card .lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.10em;color:#9ea3b0;font-weight:700;margin-bottom:4px}
.cs-resume-card .q{font-size:.95rem;font-weight:500;color:#E5E7EB;line-height:1.4}
.cs-resume-card .meta{font-size:.72rem;color:#7a8190;margin-top:6px}
.cs-resume-card{position:relative}
.cs-resume-x{
  position:absolute;top:6px;right:8px;width:22px;height:22px;
  display:inline-flex;align-items:center;justify-content:center;
  background:transparent;border:none;color:#9ea3b0;font-size:1.05rem;line-height:1;
  border-radius:50%;cursor:pointer;padding:0;
  transition:background-color .15s ease, color .15s ease, transform .15s ease;
}
.cs-resume-x:hover{background:rgba(255,255,255,.08);color:#E5E7EB;transform:scale(1.08)}
.cs-resume-x:focus-visible{outline:2px solid #8E6BFF;outline-offset:2px}
</style>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Network online/offline banner. Single always-mounted iframe at module
# scope (stable script position) emits a sticky banner on the parent
# document when navigator.onLine flips. Auto-dismisses 2s after recovery
# and clears Streamlit's @cache_data on api_health by hitting a noop URL
# fragment — the next sidebar render then re-evaluates health without a
# page reload. Zero ghosts: the iframe never unmounts, the banner DOM
# is owned by the iframe's parent-document hooks (idempotent insert).
#
# Also handles the `?draft=` query-param flow for edit-last-user-message
# and for the recent-question Resume card: reads draft on parent URL,
# fills the chat input textarea via React-friendly native setter, then
# clears the param via history.replaceState (no reload).
#
# Also installs a hover prefetch on the Admin Dashboard radio: when the
# user moves over it, fire the analytics endpoints once (debounced) so
# the tab lands warm even if the user takes >30s to click.
#
# Also persists last user question per session into localStorage and
# renders a "Resume" card client-side on the empty-state landing.
components.html(
    f"""<script>
    (function(){{
      // Bridge v3: idempotent re-install — no bail guard. Each
      // iframe re-mount (post-deploy) replaces the prior bridge's
      // event handlers / observer / DOM hooks via slots stored on
      // window.parent. Without this, old closures from a previous
      // bundle would persist (dead) while the new bundle's code
      // is locked out, producing the "fixes don't appear" pattern
      // even though the deploy itself succeeded.
      const API = {json.dumps(API_URL)};
      const W = window.parent;
      if (!W) return;
      const doc = W.document;
      if (!doc) return;
      const SLOT = W.__cs_bridge__ = (W.__cs_bridge__ || {{}});

      // ─── Helpers ────────────────────────────────────────────────────
      function ensureBanner() {{
        let b = doc.getElementById('cs-net-banner');
        if (b) return b;
        b = doc.createElement('div');
        b.id = 'cs-net-banner';
        b.className = 'cs-net-banner';
        doc.body.appendChild(b);
        return b;
      }}
      function showBanner(text, mode) {{
        const b = ensureBanner();
        b.textContent = text;
        b.classList.remove('cs-offline','cs-online','cs-show');
        b.classList.add(mode === 'offline' ? 'cs-offline' : 'cs-online');
        W.requestAnimationFrame(() => W.requestAnimationFrame(() => b.classList.add('cs-show')));
      }}
      function hideBanner(after) {{
        W.setTimeout(() => {{
          const b = doc.getElementById('cs-net-banner');
          if (b) b.classList.remove('cs-show');
        }}, after || 0);
      }}

      function fillChatInput(text) {{
        // Try repeatedly for up to ~1.2s in case the textarea is
        // still mounting after a rerun. Once found, set value via
        // React-friendly native setter so Streamlit's controlled
        // component picks up the change.
        let attempts = 0;
        const tryFill = () => {{
          const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
          if (!ta) {{
            if (attempts++ < 20) W.setTimeout(tryFill, 60);
            return;
          }}
          const setter = Object.getOwnPropertyDescriptor(
            W.HTMLTextAreaElement.prototype, 'value'
          ).set;
          setter.call(ta, text);
          ta.dispatchEvent(new Event('input', {{bubbles:true}}));
          ta.focus();
          try {{ ta.setSelectionRange(ta.value.length, ta.value.length); }} catch (e) {{}}
        }};
        tryFill();
      }}

      function consumeDraftFromUrl() {{
        try {{
          const url = new URL(W.location.href);
          const draft = url.searchParams.get('draft');
          if (!draft) return;
          fillChatInput(decodeURIComponent(draft));
          url.searchParams.delete('draft');
          W.history.replaceState({{}}, '', url.toString());
        }} catch (e) {{}}
      }}

      const LS_KEY = 'chatsolveai:last_query_v1';
      // Strips streaming cursor (▍, U+258D) + any other block-elements
      // glyph; collapses whitespace; trims leading non-letter / non-
      // number noise (avatars, dingbats) so the saved query stays
      // clean across compounding Resume cycles.
      const BLOCK_RX = /[\\u2580-\\u259F]/g;
      const LEAD_RX  = /^[^\\p{{L}}\\p{{N}}]+/u;
      function sanitize(s) {{
        if (!s) return '';
        let t = String(s).replace(BLOCK_RX, '').replace(/\\u00A0/g, ' ');
        t = t.replace(/\\s+/g, ' ');
        t = t.replace(LEAD_RX, '');
        return t.trim();
      }}

      function clearResumeStorage() {{
        try {{ W.localStorage.removeItem(LS_KEY); }} catch (e) {{}}
        const old = doc.getElementById('cs-resume-card');
        if (old) old.remove();
        SLOT.shownFor = null;
      }}

      // Wipe any leftover draft text from the chat input textarea.
      // Used by the New-chat reset path: a prior Resume click filled
      // the textarea via fillChatInput(), and Streamlit's chat_input
      // keeps that React-controlled value across st.rerun(), so the
      // ghost prompt would otherwise sit in the input on the fresh
      // empty-state landing. Retries briefly to catch the textarea
      // even if it re-mounts during the rerun.
      function clearChatInputDraft() {{
        let attempts = 0;
        const tryClear = () => {{
          const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
          if (!ta) {{
            if (attempts++ < 20) W.setTimeout(tryClear, 60);
            return;
          }}
          if (!ta.value) return;
          const setter = Object.getOwnPropertyDescriptor(
            W.HTMLTextAreaElement.prototype, 'value'
          ).set;
          setter.call(ta, '');
          ta.dispatchEvent(new Event('input', {{bubbles:true}}));
        }};
        tryClear();
      }}

      function consumeClearResume() {{
        try {{
          const url = new URL(W.location.href);
          if (url.searchParams.get('clear_resume') !== '1') return;
          clearResumeStorage();
          clearChatInputDraft();
          url.searchParams.delete('clear_resume');
          W.history.replaceState({{}}, '', url.toString());
        }} catch (e) {{}}
      }}

      function saveLastQueryFromDOM() {{
        try {{
          const userMsgs = doc.querySelectorAll(
            '[class*="st-key-chatmsg-user"] [data-testid="stChatMessage"]'
          );
          if (!userMsgs.length) return;
          const last = userMsgs[userMsgs.length - 1];
          // Strict content selector: the avatar lives in
          // [data-testid="stChatMessageAvatar"] OR as a sibling
          // <img>; the actual text is inside .stMarkdown /
          // [data-testid="stMarkdownContainer"]. Targeting the
          // markdown node only is the only way to keep the avatar
          // emoji from compounding into the saved string.
          const md = last.querySelector(
            '[data-testid="stMarkdownContainer"], .stMarkdown, [data-testid="stChatMessageContent"]'
          );
          const raw = ((md || last).innerText) || '';
          const text = sanitize(raw);
          if (!text) return;
          const prev = W.localStorage.getItem(LS_KEY);
          try {{
            if (prev && JSON.parse(prev).q === text) return;
          }} catch (e) {{}}
          W.localStorage.setItem(
            LS_KEY,
            JSON.stringify({{q: text.slice(0, 300), ts: Date.now()}})
          );
        }} catch (e) {{}}
      }}

      function maybeShowResumeCard() {{
        try {{
          const hasMsgs = doc.querySelector(
            '[class*="st-key-chatmsg-"] [data-testid="stChatMessage"]'
          );
          const greet = doc.querySelector('[class*="st-key-greeting_block"]');
          if (hasMsgs || !greet) {{
            const old = doc.getElementById('cs-resume-card');
            if (old) old.remove();
            return;
          }}
          const raw = W.localStorage.getItem(LS_KEY);
          if (!raw) return;
          let data;
          try {{ data = JSON.parse(raw); }} catch (e) {{ return; }}
          if (!data || !data.q) return;
          // Sanitize legacy contaminated entries on read so a card
          // built from pre-fix data still renders clean text.
          const cleaned = sanitize(data.q);
          if (!cleaned) {{ clearResumeStorage(); return; }}
          if (cleaned !== data.q) {{
            data.q = cleaned;
            try {{ W.localStorage.setItem(LS_KEY, JSON.stringify(data)); }} catch (e) {{}}
          }}
          const ageMs = Date.now() - (data.ts || 0);
          if (ageMs > 7 * 24 * 60 * 60 * 1000) {{ clearResumeStorage(); return; }}
          if (doc.getElementById('cs-resume-card')) return;
          const card = doc.createElement('div');
          card.id = 'cs-resume-card';
          card.className = 'cs-resume-card';
          card.setAttribute('role', 'button');
          card.setAttribute('tabindex', '0');
          const mins = Math.max(1, Math.round(ageMs / 60000));
          const ago = mins < 60 ? `${{mins}} min ago`
                    : mins < 1440 ? `${{Math.round(mins/60)}}h ago`
                    : `${{Math.round(mins/1440)}}d ago`;
          card.innerHTML =
            '<button class="cs-resume-x" type="button" aria-label="Dismiss">×</button>' +
            '<div class="lbl">↻ Resume previous chat</div>' +
            '<div class="q"></div>' +
            '<div class="meta"></div>';
          card.querySelector('.q').textContent = data.q;
          card.querySelector('.meta').textContent = 'Asked ' + ago + ' · click to prefill input';
          const useResume = () => {{
            try {{
              const url = new URL(W.location.href);
              url.searchParams.set('draft', encodeURIComponent(data.q));
              W.history.replaceState({{}}, '', url.toString());
            }} catch (e) {{}}
            fillChatInput(data.q);
            clearResumeStorage();
          }};
          card.addEventListener('click', (ev) => {{
            if (ev.target && ev.target.classList && ev.target.classList.contains('cs-resume-x')) return;
            useResume();
          }});
          card.addEventListener('keydown', (ev) => {{
            if (ev.key === 'Enter' || ev.key === ' ') {{ ev.preventDefault(); useResume(); }}
          }});
          card.querySelector('.cs-resume-x').addEventListener('click', (ev) => {{
            ev.stopPropagation();
            clearResumeStorage();
          }});
          greet.appendChild(card);
        }} catch (e) {{}}
      }}

      const ANALYTICS_PATHS = [
        "/analytics", "/analytics/timeseries?days=14",
        "/analytics/intents", "/analytics/latency", "/analytics/feedback",
      ];
      function prefetchAdmin() {{
        if (SLOT.prefetched) return;
        SLOT.prefetched = true;
        ANALYTICS_PATHS.forEach(p => {{
          try {{ fetch(API + p, {{mode:'no-cors', cache:'no-store'}}); }} catch (e) {{}}
        }});
      }}
      function wireAdminHover() {{
        doc.querySelectorAll('label, [data-testid="stRadio"] label').forEach(l => {{
          if (l.__csWired__) return;
          if ((l.textContent || '').includes('Admin')) {{
            l.__csWired__ = true;
            l.addEventListener('mouseenter', prefetchAdmin, {{once:true}});
            l.addEventListener('focusin', prefetchAdmin, {{once:true}});
          }}
        }});
      }}

      // ─── Re-install handlers / observer (replaces prior bridge) ─────
      if (SLOT.handlers) {{
        try {{
          W.removeEventListener('online',   SLOT.handlers.online);
          W.removeEventListener('offline',  SLOT.handlers.offline);
          W.removeEventListener('popstate', SLOT.handlers.popstate);
        }} catch (e) {{}}
      }}
      const onlineH  = () => {{
        showBanner("Back online. Re-checking backend…", 'online');
        fetch(API + "/health", {{mode:'no-cors', cache:'no-store'}})
          .catch(()=>{{}}).finally(() => hideBanner(1800));
      }};
      const offlineH = () => {{
        showBanner("You're offline — waiting for connection…", 'offline');
      }};
      const popstateH = () => consumeDraftFromUrl();
      W.addEventListener('online',   onlineH);
      W.addEventListener('offline',  offlineH);
      W.addEventListener('popstate', popstateH);
      SLOT.handlers = {{online: onlineH, offline: offlineH, popstate: popstateH}};
      if (!W.navigator.onLine) offlineH();

      consumeClearResume();
      consumeDraftFromUrl();
      wireAdminHover();

      if (SLOT.observer) {{
        try {{ SLOT.observer.disconnect(); }} catch (e) {{}}
      }}
      let ticking = false;
      const tick = () => {{
        if (ticking) return;
        ticking = true;
        W.requestAnimationFrame(() => {{
          ticking = false;
          try {{
            saveLastQueryFromDOM();
            maybeShowResumeCard();
            wireAdminHover();
          }} catch (e) {{}}
        }});
      }};
      const mo = new W.MutationObserver(tick);
      mo.observe(doc.body, {{childList:true, subtree:true}});
      SLOT.observer = mo;
      tick();
    }})();
    </script>""",
    height=0,
)

# ══════════════════════════════════════════════
# Session state helpers (chat only)
# ══════════════════════════════════════════════
# Per-message widget keys that must be cleared on New chat reset so a
# fresh conv_id doesn't collide with stale feedback / regen / chip state.
# Hoisted to module scope so the prefix list is auditable and shared
# between _perform_full_reset and any future cleanup paths.
_RESET_KEY_PREFIXES = ("fb_","up_","down_","fu_","chip_","topic_")

def _session_id_from_url():
    try: sid = st.query_params.get("sid")
    except Exception: return None
    if isinstance(sid, list): sid = sid[0] if sid else None
    return str(sid).strip() if sid and len(str(sid))<=128 else None

def _sync_session_url():
    try:
        if st.query_params.get("sid") != st.session_state.session_id:
            st.query_params["sid"] = st.session_state.session_id
    except Exception: pass

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
ASSISTANT_AVATAR = "logo/Logo.png"
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

# ──────────────────────────────────────────────────────────────────────────────
# "New chat" button uses a custom logo (logo/new_chat_logo.png, ~3.5:1 aspect).
# Image is set as background so the button keeps native click behavior; the
# label is kept in the DOM for accessibility but visually hidden. Same trick
# the category icon buttons use; emitted once at module load.
# ──────────────────────────────────────────────────────────────────────────────
# Two variants: light-mode logo keeps dark text (reads on cream), dark-mode
# logo has off-white text (reads on dark mesh background). Default <style>
# block at module load uses the dark variant; _LIGHT_CSS_GLOBAL overrides
# the background-image when the theme toggle is on.
_NEW_CHAT_LOGO_B64 = _img_b64("logo/new_chat_logo_dark.png")
_NEW_CHAT_LOGO_LIGHT_B64 = _img_b64("logo/new_chat_logo.png")
# The PNG itself contains a rounded white pill, so the cleanest visual is
# to drop the surrounding button chrome (border + background + box-shadow)
# and let the logo BE the button. Hover applies a subtle lift + glow via
# transform / filter so there's still affordance without a second frame.
# `!important` overrides the global :hover rule (line ~127) and the
# light-theme background-color rule, so the look is consistent in both
# themes with no double-border artifact.
_NEW_CHAT_CSS = (
    "[class*='st-key-btn_new_chat']{margin-top:4px}"
    "[class*='st-key-btn_new_chat'] button{"
    f"background-image:url('data:image/png;base64,{_NEW_CHAT_LOGO_B64}')!important;"
    "background-position:center!important;"
    "background-size:contain!important;"
    "background-repeat:no-repeat!important;"
    "background-color:transparent!important;"
    "height:56px!important;"
    "padding:0!important;"
    "border:none!important;"
    "outline:none!important;"
    "border-radius:14px!important;"
    "box-shadow:none!important;"
    "font-size:0!important;"  # hide the accessible "New chat" text label
    "color:transparent!important;"
    "transition:transform .18s cubic-bezier(.16,1,.3,1),"
    "filter .25s ease,box-shadow .25s ease!important;"
    "}"
    "[class*='st-key-btn_new_chat'] button>div,"
    "[class*='st-key-btn_new_chat'] button p{font-size:0!important;color:transparent!important}"
    "[class*='st-key-btn_new_chat'] button:hover{"
    "background-color:transparent!important;"
    "border:none!important;"
    "transform:translateY(-1px) scale(1.02)!important;"
    "filter:brightness(1.04) drop-shadow(0 4px 12px rgba(79,139,249,.28))!important;"
    "box-shadow:none!important;"
    "}"
    "[class*='st-key-btn_new_chat'] button:active{"
    "transform:translateY(0) scale(.98)!important;"
    "filter:brightness(.98)!important;"
    "}"
    "[class*='st-key-btn_new_chat'] button:focus-visible{"
    "outline:2px solid #4F8BF9!important;outline-offset:3px!important;"
    "}"
)
st.markdown(f"<style>{_NEW_CHAT_CSS}</style>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Light-theme CSS. Hoisted to module scope so render_chat AND render_admin can
# both emit it (the toggle now appears in both views' sidebars). Two parts:
#   _LIGHT_CSS_GLOBAL — page chrome (sidebar, hero, chat, pills, ...).
#   _LIGHT_CSS_ADMIN  — admin-grid specific overrides (metric cards, tabs,
#                        dataframes, refresh button, captions).
# ──────────────────────────────────────────────────────────────────────────────
# Warm light palette (replaces earlier cool #f5f7fb / #ffffff scheme):
#   page    #faf6ee  - warm cream, easier on eyes than pure white
#   sidebar #fffdf7  - subtle paper tint
#   surface #fffefa  - cards, chat input, message bubbles
#   border  #ece5d6  - warm beige border
#   text    #1c1917 (primary) / #574f43 (secondary)
#   accent  #4F8BF9 (unchanged so brand colors stay consistent)
_LIGHT_CSS_GLOBAL = (
    # Page surfaces + the Streamlit top header bar (was leaving a black
    # band above the content area).
    "[data-testid='stApp'],[data-testid='stMain'],[data-testid='stAppViewContainer'],.main{background:#faf6ee!important;color:#1c1917!important}"
    "[data-testid='stHeader']{background:#faf6ee!important;border-bottom:1px solid #ece5d6!important}"
    "[data-testid='stToolbar']{background:transparent!important}"
    "[data-testid='stToolbar'] *{color:#1c1917!important}"
    "[data-testid='stApp']::before,[data-testid='stAppViewContainer']::before,[data-testid='stMain']::before{display:none!important;background:none!important;animation:none!important;opacity:0!important}"
    # Sidebar
    "[data-testid='stSidebar']{background:#fffdf7!important;border-right:1px solid #ece5d6!important;backdrop-filter:none!important}"
    "[data-testid='stSidebar'] *{color:#1c1917!important}"
    "[data-testid='stSidebarNav'] a{color:#1c1917!important}"
    "[data-testid='stSidebarNav'] a:hover{background:#f1e9d6!important;color:#1c1917!important}"
    "[data-testid='stSidebarNav'] a[aria-current='page']{background:#e8dfc7!important;color:#1c1917!important}"
    # Hero — keep brand gradient on the title, but use deeper colors
    # that read clearly against the warm cream background.
    ".hero-title{background:linear-gradient(90deg,#1d4ed8 0%,#7c3aed 100%)!important;-webkit-background-clip:text!important;background-clip:text!important;-webkit-text-fill-color:transparent!important;color:transparent!important;filter:none!important}"
    ".hero-sub{color:#574f43!important}"
    ".page-entry-3 strong,.page-entry-3{color:#1c1917!important}"
    # Chat input — broaden selectors so the BaseWeb wrapper (the dark
    # band visible around the textarea) gets the warm surface too.
    "[data-testid='stChatInput'],[data-testid='stChatInput'] > div,[data-testid='stChatInput'] [data-baseweb='base-input'],[data-testid='stChatInput'] [data-baseweb='textarea']{background:#fffefa!important;border-color:#ece5d6!important;box-shadow:0 1px 3px rgba(28,25,23,.05)}"
    "[data-testid='stChatInput']{border:1px solid #ece5d6!important;border-radius:14px!important}"
    "[data-testid='stChatInput'] textarea{color:#1c1917!important;background:#fffefa!important;caret-color:#1c1917!important}"
    "[data-testid='stChatInput'] textarea::placeholder{color:#8a8378!important}"
    "[data-testid='stChatInput'] button{background:#f5ede0!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    "[data-testid='stChatInput'] button:hover{background:#ebe1ce!important;border-color:#4F8BF9!important}"
    "[data-testid='stChatInput'] button svg{fill:#1c1917!important;color:#1c1917!important}"
    # Chat message bubbles
    "[data-testid='stChatMessage']{background:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    "[data-testid='stChatMessage'] *{color:#1c1917!important}"
    "[class*='st-key-chatmsg-user'] [data-testid='stChatMessage']{background:#eef4ff!important;border-color:#cdddfb!important;border-left:3px solid #4F8BF9!important}"
    "[class*='st-key-chatmsg-asst'] [data-testid='stChatMessage']{background:#fffefa!important;border-color:#ece5d6!important;border-left:3px solid #d6cdb8!important}"
    # Use background-color (not the `background` shorthand) so the category
    # icon's background-image set at module load survives.
    "[class*='st-key-iconbtn_'] button{background-color:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important;box-shadow:0 1px 3px rgba(28,25,23,.05)}"
    "[class*='st-key-iconbtn_'] button:hover{background-color:#f5ede0!important;border-color:#4F8BF9!important}"
    "[class*='st-key-chipwrap_'] button{background:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    "[class*='st-key-chipwrap_'] button:hover{background:#f5ede0!important;border-color:#4F8BF9!important;color:#1c1917!important}"
    "[class*='st-key-chipwrap_'] button::after{color:#8a8378!important}"
    # New-chat button: keep chromeless, but swap to the dark-text logo
    # variant (no white pill, original text colors) so it reads on the
    # warm cream background. Hover gets a deeper-blue drop-shadow.
    "[class*='st-key-btn_new_chat'] button{"
    f"background-image:url('data:image/png;base64,{_NEW_CHAT_LOGO_LIGHT_B64}')!important;"
    "background-color:transparent!important;border:none!important"
    "}"
    "[class*='st-key-btn_new_chat'] button:hover{background-color:transparent!important;border:none!important;filter:brightness(1.02) drop-shadow(0 4px 12px rgba(29,78,216,.22))!important}"
    # Sidebar agent status
    ".agent-status{background:#fffefa!important;border:1px solid #ece5d6!important}"
    ".agent-status__label{color:#1c1917!important}"
    ".agent-status--online .agent-status__dot{box-shadow:0 0 0 3px rgba(34,197,94,.18)}"
    # Drill section header
    ".drill-section h3{color:#1c1917!important}"
    # Captions + small text
    ".stMarkdown p,.stMarkdown li,.stCaption,small,[data-testid='stCaptionContainer']{color:#574f43!important}"
    # Pills
    ".pill{background:rgba(79,139,249,.10)!important;color:#1e40af!important}"
    ".pill-green{background:rgba(34,197,94,.12)!important;color:#15803d!important}"
    ".pill-amber{background:rgba(234,179,8,.14)!important;color:#854d0e!important}"
    ".pill-red{background:rgba(239,68,68,.14)!important;color:#991b1b!important}"
    ".pill-purple{background:rgba(168,85,247,.14)!important;color:#6b21a8!important}"
    # Source cards
    ".src-card{background:#fffefa!important;border:1px solid #ece5d6!important;border-left:3px solid #4F8BF9!important;color:#1c1917!important}"
    ".src-meta{color:#574f43!important}"
    # Dividers
    "hr{border-top:1px solid #ece5d6!important;opacity:1!important}"
    # Sign-out button (admin sidebar)
    "[class*='st-key-admin_signout'] button{background:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    # Scrollbars
    "*::-webkit-scrollbar-track{background:#f1e9d6!important}"
    "*::-webkit-scrollbar-thumb{background:#d6cdb8!important;border:1px solid #ece5d6!important}"
    "*::-webkit-scrollbar-thumb:hover{background:#bfb59c!important}"
)

_LIGHT_CSS_ADMIN = (
    ".st-key-admin_grid [data-testid='stMetric']{background:#fffefa!important;border:1px solid #ece5d6!important;backdrop-filter:none!important;box-shadow:0 1px 3px rgba(28,25,23,.05)}"
    ".st-key-admin_grid [data-testid='stMetric']:hover{box-shadow:0 6px 20px rgba(79,139,249,.10)!important;border-color:rgba(79,139,249,.35)!important}"
    ".st-key-admin_grid [data-testid='stMetricLabel'],.st-key-admin_grid [data-testid='stMetricLabel'] *{color:#574f43!important}"
    ".st-key-admin_grid [data-testid='stMetricValue'],.st-key-admin_grid [data-testid='stMetricValue'] *{color:#1c1917!important;text-shadow:none!important}"
    ".st-key-admin_grid h1,.st-key-admin_grid h2,.st-key-admin_grid h3{color:#1c1917!important}"
    ".st-key-admin_grid [data-testid='stTabs'] button[role='tab']{color:#574f43!important}"
    ".st-key-admin_grid [data-testid='stTabs'] button[role='tab'][aria-selected='true']{color:#1c1917!important}"
    ".st-key-admin_grid [data-testid='stTabs'] [data-baseweb='tab-list']{border-bottom:1px solid #ece5d6!important}"
    ".st-key-admin_grid [data-baseweb='select']>div{background:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    ".st-key-admin_grid [data-baseweb='select'] *{color:#1c1917!important}"
    ".st-key-admin_grid label,.st-key-admin_grid [data-testid='stWidgetLabel']{color:#574f43!important}"
    ".st-key-admin_grid [data-testid='stButton'] button{background:#fffefa!important;border:1px solid #ece5d6!important;color:#1c1917!important}"
    ".st-key-admin_grid [data-testid='stButton'] button:hover{background:#f5ede0!important;border-color:#4F8BF9!important}"
    ".st-key-admin_grid [data-testid='stDataFrame']{background:#fffefa!important;border:1px solid #ece5d6!important;border-radius:10px!important}"
    ".st-key-admin_grid [data-testid='stDataFrame'] *{color:#1c1917!important}"
    ".st-key-admin_grid [data-testid='stCaptionContainer']{color:#574f43!important}"
    # Slider track + thumb (Rows to show)
    ".st-key-admin_grid [data-testid='stSlider'] [data-baseweb='slider'] div[role='progressbar']{background:#cdddfb!important}"
    ".st-key-admin_grid [data-testid='stSlider'] [data-baseweb='slider'] div[role='slider']{background:#4F8BF9!important;border:2px solid #fffefa!important}"
)

def _init_state():
    url_id = _session_id_from_url()
    for k,v in {"session_id":url_id or str(uuid.uuid4()),"conv_id":str(uuid.uuid4())[:8],
                "messages":[],"last_sources":[],"last_meta":{},"pending_query":None,
                "pending_append_user":True,"history_loaded_for":None}.items():
        if k not in st.session_state: st.session_state[k] = v
    st.session_state.pop("followups",None)
    # Bind the session_id onto Sentry's scope as an anonymous user id
    # so a chain of breadcrumbs (chat/admin fetches, errors) groups by
    # session in the Sentry UI. Re-set every rerun because Streamlit
    # tears down the per-request scope; cheap, idempotent.
    try:
        import sentry_sdk as _ss
        _ss.set_user({"id": st.session_state["session_id"]})
    except Exception:
        pass

# ══════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════
@st.cache_data(ttl=15,show_spinner=False)
def api_health():
    for _ in range(HEALTH_RETRIES+1):
        try:
            r = requests.get(f"{API_URL}/health",timeout=HEALTH_TIMEOUT_S)
            if r.ok: return True
        except Exception: pass
        time.sleep(2)
    return False

def _record_chat_breadcrumb(endpoint, t0, attempts, ok, status=None, exc=None):
    try:
        import sentry_sdk as _ss
        ms = int((time.perf_counter() - t0) * 1000)
        _ss.add_breadcrumb(
            category="chat_fetch",
            level="info" if ok else "warning",
            message=endpoint,
            data={"latency_ms": ms, "attempts": attempts, "ok": ok,
                  "status": status, "error": repr(exc) if exc else None},
        )
    except Exception:
        pass

def _idempotency_key(query):
    """Stable per-submit key so retries don't double-bill OpenAI. Generated
    once per submit_query() call via session_state and rotated when the
    next user turn starts."""
    if not st.session_state.get("_idempotency_key"):
        st.session_state["_idempotency_key"] = uuid.uuid4().hex
    return st.session_state["_idempotency_key"]

def _api_headers_for_chat(query):
    h = _api_headers()
    h["X-Idempotency-Key"] = _idempotency_key(query)
    return h

def call_chat(query):
    t0 = time.perf_counter()
    last_status = None
    last_exc = None
    for attempt in range(2):
        try:
            r = requests.post(f"{API_URL}/chat",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers_for_chat(query),timeout=60)
            last_status = r.status_code
            if r.ok:
                _record_chat_breadcrumb("/chat", t0, attempt+1, ok=True, status=r.status_code)
                return r.json()
            if 500<=r.status_code<600 and attempt==0: time.sleep(3); continue
        except Exception as e:
            last_exc = e
            time.sleep(3)
    _record_chat_breadcrumb("/chat", t0, 2, ok=False, status=last_status, exc=last_exc)
    return None

def call_chat_stream(query, output_box=None):
    parts,final,err = [],[],None
    t0 = time.perf_counter()
    last_flush = 0.0
    FLUSH_INTERVAL_S = 0.04  # ~25 fps; batches token paints to cut markdown re-render cost
    CURSOR = " ▍"

    def _paint(force=False):
        nonlocal last_flush
        if not output_box: return
        now = time.perf_counter()
        if force or (now - last_flush) >= FLUSH_INTERVAL_S:
            output_box.markdown("".join(parts) + CURSOR)
            last_flush = now

    try:
        with requests.post(f"{API_URL}/chat/stream",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers_for_chat(query),timeout=90,stream=True) as r:
            if not r.ok:
                _record_chat_breadcrumb("/chat/stream", t0, 1, ok=False, status=r.status_code)
                return None
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                payload = line.removeprefix("data:").strip()
                if payload=="[DONE]": break
                try: ev = json.loads(payload)
                except Exception: continue
                if "token" in ev:
                    parts.append(ev["token"])
                    _paint()
                elif ev.get("event")=="final": final = ev
    except Exception as e:
        err = str(e)
        _record_chat_breadcrumb("/chat/stream", t0, 1, ok=False, exc=e)
    if output_box and parts:
        # final paint without the cursor
        output_box.markdown("".join(parts))
    if final:
        final.setdefault("answer","".join(parts))
        _record_chat_breadcrumb("/chat/stream", t0, 1, ok=True)
        return final
    if parts:
        return {"answer":"".join(parts),"source_documents":[],"confidence":0.0,
                "condensed_query":query,"intent":"general","latency_ms":0}
    if err: st.error(f"Stream error: {err}")
    return None

def call_suggest(answer):
    try:
        r = requests.post(f"{API_URL}/suggest",json={"last_answer":answer,"n":3},headers=_api_headers(),timeout=25)
        if r.ok: return r.json().get("suggestions",[])
    except Exception: pass
    return []

def _post_feedback_async(payload):
    """Run feedback POST off the request thread so the UI rerun isn't
    gated on backend latency. Daemon thread; failures are swallowed —
    feedback is best-effort and surfacing errors here would just nag."""
    def _send():
        try:
            requests.post(
                f"{API_URL}/feedback",
                json=payload,
                headers=_api_headers(),
                timeout=10,
            )
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()

def call_feedback(q,a,rating):
    _post_feedback_async({
        "session_id": st.session_state.session_id,
        "query": q, "answer": a, "rating": rating,
    })
    return True

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_admin(path):
    """Fetch admin endpoint with exponential-backoff retry. Handles
    HuggingFace Spaces cold-start delays (the backend can take 20-60s
    to wake from sleep). Three attempts: immediate, +2s, +5s.

    Wraps the call in a Sentry breadcrumb (when DSN configured) so
    slow analytics endpoints surface in error reports without
    dragging Streamlit's normal logging into Sentry's noisy default
    capture."""
    last_exc = None
    t0 = time.perf_counter()
    attempts = 0
    for delay in (0, 2, 5):
        if delay:
            time.sleep(delay)
        attempts += 1
        try:
            r = requests.get(f"{API_URL}{path}", headers=_api_headers(), timeout=25)
            r.raise_for_status()
            _record_admin_breadcrumb(path, t0, attempts, ok=True)
            return r.json()
        except Exception as e:
            last_exc = e
    _record_admin_breadcrumb(path, t0, attempts, ok=False, exc=last_exc)
    raise last_exc if last_exc else RuntimeError(f"fetch failed: {path}")


def _record_admin_breadcrumb(path, t0, attempts, ok, exc=None):
    try:
        import sentry_sdk as _ss  # local import: silently skipped if not installed
        ms = int((time.perf_counter() - t0) * 1000)
        _ss.add_breadcrumb(
            category="admin_fetch",
            level="info" if ok else "warning",
            message=path,
            data={"latency_ms": ms, "attempts": attempts, "ok": ok,
                  "error": repr(exc) if exc else None},
        )
    except Exception:
        pass

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
    condensed = (meta.get("condensed_query") or "")[:140]
    # Free native tooltips on each pill — gives power users a one-hover
    # diagnostic without adding any new DOM element (zero ghost risk).
    intent_t  = html.escape(f"Intent classifier: {info['label']} ({intent})", quote=True)
    conf_t    = html.escape(f"Model confidence in this answer: {conf:.2f}. Higher = retrieval found a clear match.", quote=True)
    lat_t     = html.escape(f"End-to-end latency for this turn: {lat} ms (includes retrieval + LLM stream).", quote=True)
    if condensed:
        conf_t += " · q: " + html.escape(condensed, quote=True)
    pills = (f'<span class="pill" title="{intent_t}">{info["emoji"]} {info["label"]}</span>'
             f'<span class="pill {pc}" title="{conf_t}">{int(conf*100)}% confidence</span>'
             f'<span class="pill pill-purple" title="{lat_t}">⚡ {lat} ms</span>')
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
            full = src.get("content","") or ""
            # title= gives a free native browser tooltip with the full passage
            # text on hover, w/o adding any iframe / new DOM (zero ghost risk).
            tooltip = html.escape(full[:800], quote=True)
            snippet = html.escape(full[:220]) + ("…" if len(full) > 220 else "")
            line_safe = html.escape(line)
            st.markdown(
                f'<div class="src-card {cls}" title="{tooltip}">'
                f'<strong>{rank}</strong> — {snippet}'
                f'<div class="src-meta">{line_safe}</div></div>',
                unsafe_allow_html=True,
            )

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
    # Rotate the idempotency key per submit so a new turn (or a
    # Regenerate) gets a fresh dedup window on the backend, while
    # internal retries within call_chat / call_chat_stream still
    # reuse the same key (set on first read inside this submit).
    st.session_state.pop("_idempotency_key", None)
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
            try:
                result = call_chat_stream(query,box) if USE_STREAMING else call_chat(query)
                if not USE_STREAMING and result: box.markdown(result.get("answer",""))
            except Exception as exc:
                result = None
                st.toast(f"Chat request failed: {exc}", icon="⚠️")
    if not result:
        st.toast("No response from backend. Try again in a moment.", icon="⚠️")
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
    # Connection-quality tracker: keep the most recent 5 turn latencies
    # so the sidebar dot can grade end-to-end perf as green/amber/red
    # by trailing p50, not just binary online/offline.
    try:
        lat_ms = int(meta.get("latency_ms", 0))
        if lat_ms > 0:
            buf = st.session_state.get("_latency_buf", [])
            buf.append(lat_ms); buf = buf[-5:]
            st.session_state["_latency_buf"] = buf
    except Exception:
        pass
    return True

def _perform_full_reset():
    try: st.query_params.clear()
    except Exception: pass
    api_health.clear()
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["conv_id"] = str(uuid.uuid4())[:8]
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"] = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True
    st.session_state["history_loaded_for"] = None
    for k in ("followups","selected_topic","_idempotency_key",
              "_animated_msgs","_latency_buf"):
        st.session_state.pop(k, None)
    _sync_session_url()
    for k in list(st.session_state.keys()):
        if isinstance(k,str) and k.startswith(_RESET_KEY_PREFIXES):
            del st.session_state[k]

# ══════════════════════════════════════════════
# Chat view
# ══════════════════════════════════════════════
def render_chat(sidebar_slot, main_slot):
    _init_state()
    _adopt_url_session()
    _sync_session_url()

    with sidebar_slot:
        st.image("logo/Logo.png", width=256)
        st.title("Customer Support AI")
        st.divider()
        healthy = api_health()
        # Custom status indicator: pulsing green dot when the backend
        # health check passes, static red dot when not. Replaces the
        # default st.success / st.error pill so we get the 'alive'
        # animated look the design pass asked for.
        if healthy:
            buf = st.session_state.get("_latency_buf") or []
            if buf:
                # p50 of last 5 turns
                p50 = sorted(buf)[len(buf)//2]
                if p50 < 1200:
                    mod, label, tip = "online", "Agent: Online", f"Fast — recent p50 {p50} ms"
                elif p50 < 3000:
                    mod, label, tip = "medium", "Agent: Slower", f"Backend slower than usual — p50 {p50} ms"
                else:
                    mod, label, tip = "poor", "Agent: Lagging", f"High latency — p50 {p50} ms"
            else:
                mod, label, tip = "online", "Agent: Online", "Backend reachable"
            st.markdown(
                f'<div class="agent-status agent-status--{mod}" title="{html.escape(tip)}">'
                '<span class="agent-status__dot"></span>'
                f'<span class="agent-status__label">{label}</span>'
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

        st.divider()
        # Visual: logo image background (set in module-level CSS).
        # Label kept as accessible name for screen readers.
        if st.button("New chat", key="btn_new_chat", use_container_width=True,
                     help="Start a new chat — also forgets the saved Resume card"):
            _perform_full_reset()
            # Signal the network-bridge iframe to drop the cached
            # last-query in localStorage (and any rendered Resume
            # card). The bridge consumes this query param via
            # history.replaceState — no page reload, no ghost.
            try:
                st.query_params["clear_resume"] = "1"
            except Exception:
                pass
            st.rerun()

        # Light/Dark theme toggle. Label flips with state so the user
        # always sees the OPPOSITE of the current mode (label = the
        # mode they'd switch INTO if they clicked).
        _current_light = st.session_state.get("_theme_light", False)
        _toggle_label = "☀️ Light mode" if _current_light else "🌙 Dark mode"
        is_light = st.toggle(
            _toggle_label,
            key="_theme_light",
            help="Switch between light and dark color schemes.",
            on_change=_on_theme_toggle,
        )

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

    # Light theme overrides. Always emit the style tag with conditional
    # contents so its script position stays stable across reruns.
    _light_css = (_LIGHT_CSS_GLOBAL + _LIGHT_CSS_ADMIN) if is_light else ""
    st.markdown(f"<style>{_light_css}</style>", unsafe_allow_html=True)

    # Backend warmup: fire a background fetch to /health on first
    # render so the HuggingFace Space wakes from sleep before the
    # user clicks Admin Dashboard. Pure browser-side, doesn't block
    # Python rendering.
    if is_first_render:
        # Background warmup: hit /health to wake the HF Space, and
        # opportunistically prefetch the admin analytics endpoints so
        # clicking "Admin Dashboard" lands on a populated view instead
        # of a 20-30s cold-start spinner. Sequenced so /health primes
        # the worker before analytics kicks off; failures are silent.
        components.html(
            f"""<script>
            (async () => {{
              const opts = {{mode:'no-cors', cache:'no-store', credentials:'omit'}};
              const paths = [
                "/health",
                "/analytics",
                "/analytics/timeseries?days=14",
                "/analytics/intents",
                "/analytics/latency",
                "/analytics/feedback",
              ];
              try {{ await fetch("{API_URL}" + paths[0], opts); }} catch (e) {{}}
              // Fan out the remaining prefetches; HTTP/2 multiplexes
              // them so this is one cheap round of warmup.
              paths.slice(1).forEach(p => {{
                try {{ fetch("{API_URL}" + p, opts); }} catch (e) {{}}
              }});
            }})();
            </script>""",
            height=0,
        )
        # Onboarding hint: shown once per session via toast. Doesn't
        # add persistent DOM and auto-dismisses, so it cannot ghost.
        if not st.session_state.get("_seen_onboarding"):
            st.toast(
                "Pick a category above for sample questions, "
                "or type your own in the input below.",
                icon="✨",
            )
            st.session_state["_seen_onboarding"] = True

    with main_slot, st.container(key="chat_root"):
        # Page-entry stagger classes only attach on first render.
        hero_cls = "hero-title page-entry-1" if is_first_render else "hero-title"
        sub_cls = "hero-sub page-entry-2" if is_first_render else "hero-sub"
        st.markdown(
            f'<div class="{hero_cls}">💬 Customer Support AI</div>',
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
                    f'What do you need help with?</strong></div>',
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
                    # Stable chip keys (no asked-count suffix). The
                    # pre-allocated st.empty() slots already handle
                    # ghost-chip cleanup, and stable keys mean the
                    # entry animation fires exactly once per chip
                    # mount instead of re-firing on every rerun (which
                    # made the post-answer transition feel like a
                    # reload). Animation rules emitted once at fixed
                    # script position above the slots; targets full
                    # wrap_key with computed delay.
                    delay_rules = []
                    for idx, (j, _q) in enumerate(remaining):
                        wrap_key = f"chipwrap_{conv}_{selected}_{j}"
                        delay = 0.08 + (idx * 0.10)
                        delay_rules.append(
                            f".st-key-{wrap_key}{{"
                            f"animation:slideUpFade .35s cubic-bezier(.22,1,.36,1) {delay:.3f}s both"
                            f"}}"
                        )
                    st.markdown(
                        f"<style>{''.join(delay_rules)}</style>",
                        unsafe_allow_html=True,
                    )
                    for slot, (j, q) in zip(chip_slots, remaining):
                        with slot.container():
                            wrap_key = f"chipwrap_{conv}_{selected}_{j}"
                            with st.container(key=wrap_key):
                                st.button(q,
                                          key=f"chip_{conv}_{selected}_{j}",
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

        # Chat-history slot. `st.empty()` ensures that when New chat
        # clears the message list, the prior chat_message bubbles and
        # any embedded iframe components (typing dots, sources)
        # unmount cleanly. Without the slot, Streamlit Cloud retained
        # the chat-history subtree across the reset, surfacing as
        # ghost messages on the landing screen.
        # Per-message animation gate: stable script position. Always
        # emits the style tag (empty when nothing to suppress) so its
        # element identity never shifts across reruns.
        #
        # Rule: the two most-recent messages (last user + last asst)
        # ALWAYS get their slide-in keyframe. Older messages get the
        # keyframe suppressed via per-key rules, so a streaming rerun
        # doesn't replay every prior bubble's entrance. This is the
        # most predictable model — every new turn animates, history
        # stays still.
        last_user_idx_a = max(
            (i for i, m in enumerate(msgs) if m["role"] == "user"),
            default=-1,
        )
        last_asst_idx_a = max(
            (i for i, m in enumerate(msgs) if m["role"] == "assistant"),
            default=-1,
        )
        anim_keep = {last_user_idx_a, last_asst_idx_a}
        anim_rules = []
        for idx, msg in enumerate(msgs):
            if idx in anim_keep:
                continue
            role_short_a = "user" if msg["role"] == "user" else "asst"
            anim_rules.append(
                f"[class*='st-key-chatmsg-{role_short_a}-{conv}-{idx}'] [data-testid='stChatMessage']"
                "{animation:none!important}"
            )
        st.markdown(f"<style>{''.join(anim_rules)}</style>", unsafe_allow_html=True)

        chat_slot = st.empty()
        if msgs or has_pending:
            with chat_slot.container():
                with st.container(key="chat_history_box", height=520):
                    # Conversation summary chip — derived stats, no API
                    # call. Appears only once the conversation has some
                    # weight (>=3 messages) so single-turn chats aren't
                    # cluttered by a header.
                    if len(msgs) >= 3:
                        asst_metas = [m["meta"] for m in msgs
                                      if m["role"] == "assistant" and isinstance(m.get("meta"), dict)]
                        user_n = sum(1 for m in msgs if m["role"] == "user")
                        if asst_metas:
                            avg_lat = sum(int(m.get("latency_ms", 0)) for m in asst_metas) // len(asst_metas)
                            high = sum(1 for m in asst_metas if float(m.get("confidence", 0)) >= 0.75)
                            pct = int(100 * high / len(asst_metas))
                        else:
                            avg_lat, pct = 0, 0
                        st.markdown(
                            f'<div class="conv-summary">'
                            f'<span class="label">Session</span>'
                            f'<span class="pill">{user_n} question{"s" if user_n != 1 else ""}</span>'
                            f'<span class="pill pill-purple">⚡ avg {avg_lat} ms</span>'
                            f'<span class="pill pill-green">{pct}% high-confidence</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    last_user_idx = max(
                        (i for i, m in enumerate(msgs) if m["role"] == "user"),
                        default=-1,
                    )
                    for idx, msg in enumerate(msgs):
                        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
                        role_short = "user" if msg["role"] == "user" else "asst"
                        with st.container(key=f"chatmsg-{role_short}-{conv}-{idx}"):
                            with st.chat_message(msg["role"], avatar=avatar):
                                st.markdown(msg["content"])
                                if msg["role"] == "user" and idx == last_user_idx and not has_pending:
                                    # Edit-and-resend pencil on the most
                                    # recent user message only. Drops
                                    # everything from idx onward and
                                    # routes the text into the chat
                                    # input via ?draft= which the
                                    # network-bridge iframe consumes
                                    # (history.replaceState, no reload).
                                    _ec1, _ec2 = st.columns([10, 1])
                                    with _ec2:
                                        if st.button("✎", key=f"edit_user_{conv}_{idx}",
                                                     help="Edit and resend"):
                                            draft_q = msg["content"]
                                            st.session_state.messages = msgs[:idx]
                                            try:
                                                st.query_params["draft"] = draft_q
                                            except Exception:
                                                pass
                                            st.rerun()
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

                # Scroll-to-bottom: rAF + IntersectionObserver beats the
                # prior 120ms setTimeout. rAF guarantees we paint AFTER
                # Streamlit's reconciler has appended the new message;
                # the observer watches the last bubble and re-anchors
                # while the streaming answer grows so the view doesn't
                # lag behind incoming tokens. Idempotent across reruns
                # via a window-level flag stored on the parent document.
                #
                # Also installs a floating "↓ Jump to latest" pill on
                # the parent document. The pill is hidden by default
                # and shown when the user scrolls more than 120px above
                # the bottom of the chat scroller. Click jumps back.
                # All DOM lives on the parent so it isn't affected by
                # iframe unmounts; idempotent via window-level flag.
                components.html(
                    """<script>
                    (function(){
                      try {
                        const doc = window.parent.document;
                        const pickScroller = () => {
                          const candidates = doc.querySelectorAll(
                            '[data-testid="stVerticalBlockBorderWrapper"]'
                          );
                          for (const b of candidates) {
                            if (b.scrollHeight > b.clientHeight) return b;
                          }
                          return null;
                        };
                        const ensureJumpBtn = () => {
                          let btn = doc.getElementById('cs-jump-btn');
                          if (btn) return btn;
                          btn = doc.createElement('button');
                          btn.id = 'cs-jump-btn';
                          btn.className = 'cs-jump-btn';
                          btn.type = 'button';
                          btn.innerHTML = '↓ Jump to latest';
                          btn.addEventListener('click', () => {
                            const s = pickScroller();
                            if (!s) return;
                            s.scrollTo({ top: s.scrollHeight, behavior: 'smooth' });
                          });
                          doc.body.appendChild(btn);
                          return btn;
                        };
                        const updateBtnVisibility = (s) => {
                          if (!s) return;
                          const btn = ensureJumpBtn();
                          const dist = s.scrollHeight - s.scrollTop - s.clientHeight;
                          btn.classList.toggle('cs-show', dist > 120);
                        };
                        const anchor = () => {
                          const s = pickScroller();
                          if (!s) return;
                          s.scrollTo({ top: s.scrollHeight, behavior: 'smooth' });
                          ensureJumpBtn();
                          // Wire scroll-position listener once per scroller.
                          if (!s.__csScrollWired__) {
                            s.__csScrollWired__ = true;
                            s.addEventListener('scroll', () => updateBtnVisibility(s), {passive:true});
                          }
                          updateBtnVisibility(s);
                          const last = s.querySelector(
                            '[data-testid="stChatMessage"]:last-of-type'
                          );
                          if (!last) return;
                          // Re-anchor as the last bubble grows mid-stream.
                          // ResizeObserver fires on each token paint; cap
                          // to one rAF so we don't spam scrollTo per char.
                          if (window.__chatRO__) window.__chatRO__.disconnect();
                          let scheduled = false;
                          window.__chatRO__ = new ResizeObserver(() => {
                            if (scheduled) return;
                            scheduled = true;
                            requestAnimationFrame(() => {
                              scheduled = false;
                              // Only auto-anchor if user is already near
                              // the bottom — respects manual scroll-up.
                              const dist = s.scrollHeight - s.scrollTop - s.clientHeight;
                              if (dist < 200) {
                                s.scrollTo({ top: s.scrollHeight, behavior: 'auto' });
                              }
                              updateBtnVisibility(s);
                            });
                          });
                          window.__chatRO__.observe(last);
                        };
                        requestAnimationFrame(() => requestAnimationFrame(anchor));
                      } catch (e) {}
                    })();
                    </script>""",
                    height=0,
                )
        else:
            # Force iframe-DOM replacement to flush any prior chat
            # subtree on Streamlit Cloud (slot.empty() alone left
            # message bubbles ghosting after a New chat reset).
            with chat_slot.container():
                components.html("", height=0)

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
            # Render an empty 0-height iframe in place of the copy button.
            # `slot.empty()` alone left the prior iframe component mounted
            # on Streamlit Cloud, so the copy button persisted on the
            # landing screen after a New chat reset. Replacing the iframe
            # with a fresh empty one forces the old DOM out cleanly.
            with copy_slot.container():
                components.html("", height=0)

        if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
            if not healthy:
                st.error("API unreachable.")
                st.stop()
            _queue_query(prompt)
            st.rerun()

# ══════════════════════════════════════════════
# Admin view (defined inline; rendered via the dispatch below).
# ══════════════════════════════════════════════
def render_admin(sidebar_slot, main_slot):
    if ADMIN_PASSWORD and st.session_state.get("admin_ok"):
        with sidebar_slot:
            if st.button("Sign out", key="admin_signout"):
                st.session_state.pop("admin_ok", None)
                st.rerun()

    # Light/Dark toggle, also rendered in the admin sidebar so users
    # don't lose access to the theme switch when on this view. Same
    # session_state key as the chat toggle ("_theme_light") so state
    # persists across view switches; only one of render_chat /
    # render_admin runs per dispatch, so the duplicate widget key is
    # not a runtime collision.
    with sidebar_slot:
        _current_light = st.session_state.get("_theme_light", False)
        _toggle_label = "☀️ Light mode" if _current_light else "🌙 Dark mode"
        st.toggle(
            _toggle_label,
            key="_theme_light",
            help="Switch between light and dark color schemes.",
            on_change=_on_theme_toggle,
        )

    is_light = st.session_state.get("_theme_light", False)
    # Always emit a stable <style> block (empty when dark) so the script
    # position of subsequent elements doesn't shift between renders.
    _admin_light_css = (_LIGHT_CSS_GLOBAL + _LIGHT_CSS_ADMIN) if is_light else ""
    st.markdown(f"<style>{_admin_light_css}</style>", unsafe_allow_html=True)

    with main_slot:
        if ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
            with st.form("admin_login"):
                pw = st.text_input("Admin password", type="password")
                # hmac.compare_digest defeats timing-based password
                # probing by ensuring the comparison runs in constant
                # time regardless of how many leading characters match.
                submitted = st.form_submit_button("Sign in")
                if submitted and hmac.compare_digest(
                    (pw or "").encode("utf-8"),
                    ADMIN_PASSWORD.encode("utf-8"),
                ):
                    st.session_state["admin_ok"] = True
                    st.rerun()
                if pw and submitted:
                    st.error("Invalid password.")
            return

        # Row-count control. Slider value persists via the widget key.
        # Clamp to 50 in case a prior session stored a value above the
        # backend's `limit` cap (the API returns 422 for higher values).
        row_limit = min(50, st.session_state.get("adm_row_limit", 25))
        st.session_state["adm_row_limit"] = row_limit

        st.markdown(
            "<style>"
            ".st-key-admin_grid [data-testid='stMetric']{background:rgba(20,24,32,.55);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:16px 20px;backdrop-filter:blur(8px) saturate(140%);transition:transform .25s ease,box-shadow .25s ease}"
            ".st-key-admin_grid [data-testid='stMetric']:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(79,139,249,.12);border-color:rgba(79,139,249,.25)}"
            ".st-key-admin_grid [data-testid='stMetricLabel']{color:#7a8190!important;font-size:.7rem!important;letter-spacing:.08em;text-transform:uppercase;font-weight:600!important}"
            ".st-key-admin_grid [data-testid='stMetricValue']{font-family:var(--font-display)!important;font-size:1.85rem!important;font-weight:600!important;color:#E5E7EB!important;text-shadow:0 0 14px rgba(79,139,249,.22)}"
            ".st-key-admin_grid h3{font-family:var(--font-display)!important;font-weight:600;letter-spacing:.01em;color:#E5E7EB;margin:.4rem 0 1rem!important;font-size:1.1rem}"
            ".st-key-admin_grid h1{font-size:1.85rem!important;margin-bottom:1.2rem!important}"
            ".st-key-admin_grid hr{margin:1.6rem 0!important;opacity:.4}"
            ".st-key-admin_grid [data-testid='stHorizontalBlock']{gap:1rem}"
            "</style>",
            unsafe_allow_html=True,
        )
        with st.container(key="admin_grid"):
            # Status slot at the top of the grid. Streamlit Cloud
            # was retaining the prior render's `st.warning` DOM when
            # the next render skipped the warning, so the yellow
            # "Backend unreachable" banner kept ghosting above
            # successfully-loaded charts. An `st.empty()` slot whose
            # content is explicitly replaced (with an empty 0-height
            # iframe on success) forces the prior warning out.
            status_slot = st.empty()
            with st.spinner("Loading analytics…"):
                fetch_error = None
                summary = timeseries = intents = latency = feedback = sessions = None
                try:
                    summary    = _fetch_admin("/analytics")
                    timeseries = _fetch_admin("/analytics/timeseries?days=14")
                    intents    = _fetch_admin("/analytics/intents")
                    latency    = _fetch_admin("/analytics/latency")
                    feedback   = _fetch_admin("/analytics/feedback")
                    sessions   = _fetch_admin(f"/sessions?limit={row_limit}")
                except Exception as exc:
                    fetch_error = exc

            if fetch_error is not None:
                with status_slot.container():
                    st.warning(f"Backend unreachable – analytics unavailable. ({fetch_error})")
                st.toast(f"Analytics fetch failed: {fetch_error}", icon="⚠️")
                return

            # Success: forcibly replace any prior warning DOM with an
            # empty iframe so Streamlit Cloud cannot leave the banner
            # mounted as a ghost.
            with status_slot.container():
                components.html("", height=0)

            st.title("Customer Support AI · Admin Dashboard")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sessions", summary.get("total_sessions", 0))
            c2.metric("Queries", summary.get("total_queries", 0))
            c3.metric("Today", summary.get("queries_today", 0))
            c4.metric("Avg Session", summary.get("avg_session_length", 0))

            f1, f2, f3 = st.columns(3)
            f1.metric("P95 Latency", f"{latency.get('p95', 0)} ms")
            f2.metric("👍 Upvotes", feedback.get("up", 0))
            f3.metric("👎 Downvotes", feedback.get("down", 0))

            st.divider()

            import altair as alt
            # Equal 50/50 split gives the donut enough room for its
            # legend at the bottom without stealing area from the line
            # chart. Heights bumped to 420 for better visibility.
            left, right = st.columns([1, 1], gap="large")
            with left:
                st.subheader("Support Volume")
                if timeseries and any(r.get("count", 0) > 0 for r in timeseries):
                    line = (
                        alt.Chart(alt.Data(values=timeseries))
                        .mark_line(
                            point=alt.OverlayMarkDef(filled=True, size=70, color="#4F8BF9"),
                            color="#4F8BF9", strokeWidth=3, interpolate="monotone",
                        )
                        .encode(
                            x=alt.X("date:T", title=None, axis=alt.Axis(labelColor="#7a8190", labelFontSize=11, grid=False)),
                            y=alt.Y("count:Q", title=None, axis=alt.Axis(labelColor="#7a8190", labelFontSize=11, grid=True, gridOpacity=0.15)),
                            tooltip=["date:T", "count:Q"],
                        )
                        .properties(height=420)
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.info("No data yet.")
            with right:
                st.subheader("Category Breakdown")
                if intents and any(r.get("count", 0) > 0 for r in intents):
                    donut = (
                        alt.Chart(alt.Data(values=intents))
                        .mark_arc(innerRadius=70, outerRadius=140, stroke="#0f1218", strokeWidth=2)
                        .encode(
                            theta=alt.Theta("count:Q"),
                            color=alt.Color("intent:N",
                                legend=alt.Legend(title=None, orient="bottom", labelFontSize=12, columns=3, labelColor="#cdd5e0", symbolSize=120),
                                scale=alt.Scale(scheme="tableau10")),
                            tooltip=["intent:N", "count:Q"],
                        )
                        .properties(height=420)
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(donut, use_container_width=True)
                else:
                    st.info("No data yet.")

            st.divider()

            ctrl_l, ctrl_m, ctrl_r = st.columns([3, 1, 1], gap="medium")
            with ctrl_l:
                st.subheader("Activity")
            with ctrl_m:
                if st.button("🔄 Refresh", key="adm_refresh", use_container_width=True):
                    _fetch_admin.clear()
                    st.rerun()
                st.selectbox(
                    "Auto-refresh",
                    ["Off", "30s", "1min", "5min"],
                    key="adm_auto_refresh",
                    label_visibility="collapsed",
                )
            with ctrl_r:
                st.slider(
                    "Rows to show",
                    min_value=10, max_value=50, step=5,
                    value=row_limit,
                    key="adm_row_limit",
                    help="Applies to both Top Questions and Recent Sessions. Backend caps at 50.",
                )

            # Auto-refresh: client-side meta-refresh fallback. Streamlit
            # has no native interval primitive that survives across
            # reruns, but a tiny components.html iframe with setTimeout
            # → parent.click on the rerun button reliably triggers a
            # script re-execution at the chosen interval. Iframe is
            # 0-height so it adds no layout cost.
            _intervals = {"Off": 0, "30s": 30, "1min": 60, "5min": 300}
            _ar_seconds = _intervals.get(st.session_state.get("adm_auto_refresh", "Off"), 0)
            if _ar_seconds:
                components.html(
                    f"""<script>
                    setTimeout(() => {{
                      try {{
                        const root = window.parent.document;
                        // Streamlit's rerun is triggered by clicking the
                        // hamburger > Rerun, but a stable shortcut is to
                        // dispatch the F5/refresh on the streamlit app
                        // container. Simpler: post a message Streamlit's
                        // runtime listens for. Here we just reload the
                        // current location with the same query string.
                        root.location.reload();
                      }} catch (e) {{}}
                    }}, {_ar_seconds * 1000});
                    </script>""",
                    height=0,
                )

            tab1, tab2 = st.tabs(["Top Questions", "Recent Sessions"])
            table_height = max(300, min(680, 32 + 35 * row_limit))
            with tab1:
                top = summary.get("top_questions", [])
                if top:
                    st.dataframe(top[:row_limit], use_container_width=True, hide_index=True, height=table_height)
                else:
                    st.info("No questions logged.")
            with tab2:
                if sessions:
                    st.dataframe(sessions[:row_limit], use_container_width=True, hide_index=True, height=table_height)
                else:
                    st.info("No sessions.")
            st.caption(f"Last refreshed: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")

# ══════════════════════════════════════════════
# Dispatch
# ──────────────────────────────────────────────
# Single-script view dispatch with a force-remount strategy: the
# wrapping container's key includes a counter (`_view_idx`) that
# increments on every view switch, so each transition produces a
# brand-new container identity — Streamlit's reconciler must drop
# the prior view's tree (with all its iframes / charts / Lottie
# players) and mount a fresh one. This is the only in-process
# pattern that reliably defeats Streamlit Cloud's iframe-component
# DOM retention across reruns.
# ══════════════════════════════════════════════
with st.sidebar:
    view = st.radio(
        "View",
        [NAV_CHAT, NAV_ADMIN],
        key="nav_view",
        label_visibility="collapsed",
        on_change=_on_view_change,
    )
    st.divider()

view_tag = "chat" if view == NAV_CHAT else "admin"
inactive_tag = "admin" if view == NAV_CHAT else "chat"

# Tag the active view on Sentry's current scope so issues are easy
# to filter by where they occurred. No-op when Sentry isn't installed
# or DSN is not configured.
try:
    import sentry_sdk as _ss
    _ss.set_tag("view", view_tag)
    _ss.set_tag("theme", "light" if st.session_state.get("_theme_light") else "dark")
except Exception:
    pass

with st.sidebar:
    sidebar_view = st.container(key=f"view_sb_{view_tag}")
main_view = st.container(key=f"view_main_{view_tag}")

# Component-whitelist ghost cleanup. Each view knows the exact set
# of widget types that ONLY ever appear in the OTHER view — when
# this view is active, any DOM matching those selectors is by
# definition a ghost left by Streamlit Cloud's iframe-component
# retention. Hiding them via CSS is the only mechanism that
# reliably catches stale DOM regardless of which keyed container
# wrapper Streamlit decided to reparent it under.
#
# Chat-only widgets: category icon buttons, empty-state Lottie,
# drill block, chip buttons, chat-message bubbles, greeting,
# icon row, lottie-player wrap, chat input.
#
# Admin-only widgets: admin-grid wrapper, Vega/Altair charts,
# the analytics metric cards (sidebar metrics were removed
# earlier so st.metric is now admin-only), dataframes.
HIDE = "{display:none !important;visibility:hidden !important;height:0 !important;overflow:hidden !important;position:absolute !important;left:-99999px !important}"
if view == NAV_ADMIN:
    ghost_css = (
        "[class*='st-key-chat_root'],"
        "[class*='st-key-iconbtn_'],"
        "[class*='st-key-empty_state_block'],"
        "[class*='st-key-drill_block'],"
        "[class*='st-key-chipwrap_'],"
        "[class*='st-key-chatmsg-'],"
        "[class*='st-key-greeting_block'],"
        "[class*='st-key-icon_row'],"
        "[class*='st-key-lottie_wrap'],"
        "[class*='st-key-chat_history_box'],"
        ".drill-section,"
        "[data-testid='stChatInput'],"
        "[data-testid='stChatMessage'],"
        # Floating elements owned by the network-bridge / scroll iframes
        # live on the parent document (not in a Streamlit-keyed wrapper),
        # so they need explicit hiding when not on the chat view.
        "#cs-jump-btn,"
        "#cs-resume-card"
        f"{HIDE}"
    )
    # Admin needs the wider container so the metrics grid + dataframes
    # don't get pinched. Emitted conditionally so it doesn't leak when
    # the user switches back to chat.
    ghost_css += "[data-testid='stMain'] .block-container{max-width:1500px!important}"
else:  # NAV_CHAT
    ghost_css = (
        "[class*='st-key-admin_grid'],"
        "[data-testid='stVegaLiteChart'],"
        "[data-testid='stArrowVegaLiteChart'],"
        "[data-testid='stPlotlyChart'],"
        "[data-testid='stMetric'],"
        "[data-testid='stDataFrame'],"
        "[data-testid='stTabs']"
        f"{HIDE}"
    )
    # Force the chat view back to 1100px regardless of any prior
    # admin-width inheritance. Belt-and-braces with the base rule.
    ghost_css += "[data-testid='stMain'] .block-container{max-width:1100px!important}"
ghost_css += (
    f",[class*='st-key-view_main_{inactive_tag}'],"
    f"[class*='st-key-view_sb_{inactive_tag}']"
    f"{HIDE}"
)
st.markdown(f"<style>{ghost_css}</style>", unsafe_allow_html=True)

if view == NAV_CHAT:
    render_chat(sidebar_view, main_view)
else:
    render_admin(sidebar_view, main_view)
