"""
Cover: the first thing anyone sees. One idea, one button.

Deliberately quiet. Warm paper, sage and clay, a slow sunrise over layered
ground, two leaves that barely move. The navigation bar is hidden here so the
page reads as a cover rather than as a page of the app; every other page
brings it back.
"""

from __future__ import annotations

import textwrap

import streamlit as st

COVER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600&family=Inter:wght@400;500&display=swap');

/* hide the app chrome, on this page only: the styles vanish with the page */
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], footer { display: none !important; }
.stApp {
  background:
    radial-gradient(1100px 560px at 78% 8%, rgba(244, 205, 150, .55), rgba(244, 205, 150, 0) 60%),
    radial-gradient(900px 500px at 8% 30%, rgba(190, 214, 190, .45), rgba(190, 214, 190, 0) 62%),
    linear-gradient(180deg, #fbf7ef 0%, #f3ecdd 58%, #e8e0cb 100%);
}
.block-container { max-width: 100% !important; padding: 0 !important; }

.cv { position: relative; min-height: 100vh; overflow: hidden;
  font-family: 'Inter', system-ui, sans-serif; color: #2d3a2e; }

/* the ground: three soft layers, back to front */
.cv-ground { position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 46vh; }

.cv-leaf { position: absolute; transform-origin: 50% 100%; opacity: .9;
  animation: sway 9s ease-in-out infinite; }
.cv-leaf.l1 { left: 6%;  bottom: 22vh; width: 132px; }
.cv-leaf.l2 { right: 7%; bottom: 24vh; width: 104px; animation-delay: -4s; }
@keyframes sway { 0%, 100% { transform: rotate(-2.5deg) } 50% { transform: rotate(2.5deg) } }

.cv-sun { position: absolute; top: 7vh; right: 11%; width: 210px; height: 210px; border-radius: 50%;
  background: radial-gradient(circle at 40% 38%, #fbe3b4 0%, #f2c987 55%, rgba(242, 201, 135, 0) 72%);
  animation: rise 2.6s ease-out both; }
@keyframes rise { from { transform: translateY(46px); opacity: 0 } to { transform: none; opacity: 1 } }

.cv-inner { position: relative; z-index: 2; max-width: 780px; margin: 0 auto;
  padding: 17vh 28px 0; text-align: center; }
.cv-eyebrow { font-size: .74rem; letter-spacing: .28em; text-transform: uppercase;
  color: #5d7a5f; font-weight: 500; animation: fadeup 1.2s ease-out .1s both; }
.cv-title { font-family: 'Fraunces', Georgia, 'Times New Roman', serif; font-weight: 300;
  font-size: clamp(3.4rem, 9vw, 6.6rem); line-height: 1; letter-spacing: -.025em;
  margin: 1.1rem 0 1.3rem; color: #26332a; animation: fadeup 1.4s ease-out .25s both; }
.cv-title em { font-style: italic; font-weight: 400; color: #a5573a; }
.cv-line { font-family: 'Fraunces', Georgia, serif; font-weight: 300; font-size: clamp(1.15rem, 2.4vw, 1.5rem);
  line-height: 1.5; color: #465547; max-width: 560px; margin: 0 auto;
  animation: fadeup 1.4s ease-out .5s both; }
.cv-pills { margin-top: 2rem; animation: fadeup 1.4s ease-out .75s both; }
.cv-pill { display: inline-block; margin: .25rem .3rem; padding: .32rem .9rem; border-radius: 999px;
  font-size: .8rem; color: #3f5140; background: rgba(255, 253, 247, .7);
  border: 1px solid rgba(93, 122, 95, .28); backdrop-filter: blur(3px); }
.cv-foot { position: absolute; z-index: 2; bottom: 2.2vh; left: 0; right: 0; text-align: center;
  font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: rgba(251, 247, 239, .78); }
@keyframes fadeup { from { transform: translateY(16px); opacity: 0 } to { transform: none; opacity: 1 } }

/* the one button, lifted above the ground layers */
[data-testid="stPageLink"], [data-testid="stElementContainer"]:has([data-testid="stPageLink"]) { width: 100% !important; }
[data-testid="stPageLink"] { position: relative; z-index: 3; }
[data-testid="stPageLink"] > *, [data-testid="stPageLink"] a { display: flex; }
[data-testid="stPageLink"] a span, [data-testid="stPageLink"] a svg { color: #fbf7ef !important; fill: #fbf7ef !important; }
[data-testid="stPageLink"] a {
  width: fit-content !important; flex: 0 0 auto !important; gap: .55rem;
  justify-content: center; margin: -21vh auto 0; padding: .8rem 2.3rem; border-radius: 999px;
  background: #33513a; color: #fbf7ef !important; font-weight: 500; letter-spacing: .04em;
  box-shadow: 0 10px 28px rgba(51, 81, 58, .28); transition: transform .25s ease, box-shadow .25s ease, background .25s;
  animation: fadeup 1.4s ease-out 1s both; }
[data-testid="stPageLink"] a:hover { background: #a5573a; transform: translateY(-2px);
  box-shadow: 0 14px 32px rgba(165, 87, 58, .32); }
[data-testid="stPageLink"] a p { color: #fbf7ef !important; font-size: 1rem; }

@media (max-width: 640px) {
  .cv-sun { width: 140px; height: 140px; right: 6%; }
  .cv-leaf.l1 { width: 84px; left: 2%; } .cv-leaf.l2 { width: 70px; right: 2%; }
  .cv-inner { padding-top: 14vh; }
}
@media (prefers-reduced-motion: reduce) { .cv *, .cv { animation: none !important; }
  [data-testid="stPageLink"] a { animation: none !important; } }
</style>
"""

COVER_HTML = textwrap.dedent("""
<div class="cv">
  <div class="cv-sun"></div>

  <svg class="cv-leaf l1" viewBox="0 0 120 200" aria-hidden="true">
    <path d="M60 198 C58 140 60 90 60 24" stroke="#5d7a5f" stroke-width="2.4" fill="none" stroke-linecap="round"/>
    <path d="M60 150 C28 140 14 112 16 84 C46 90 62 116 60 150Z" fill="#8fae8b" opacity=".85"/>
    <path d="M60 118 C92 108 106 82 104 56 C76 62 60 86 60 118Z" fill="#a9c4a1" opacity=".9"/>
    <path d="M60 78 C40 70 32 52 34 34 C52 40 62 56 60 78Z" fill="#7c9d7a" opacity=".85"/>
  </svg>
  <svg class="cv-leaf l2" viewBox="0 0 120 200" aria-hidden="true">
    <path d="M60 198 C62 150 58 96 60 30" stroke="#a5573a" stroke-width="2.2" fill="none" stroke-linecap="round"/>
    <path d="M60 140 C90 132 104 108 102 82 C74 88 58 110 60 140Z" fill="#d49a7c" opacity=".8"/>
    <path d="M60 100 C32 92 20 70 22 46 C48 52 62 74 60 100Z" fill="#c98466" opacity=".8"/>
  </svg>

  <svg class="cv-ground" viewBox="0 0 1440 400" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 150 C220 90 420 190 700 140 C980 90 1200 170 1440 120 L1440 400 L0 400Z" fill="#c5d5bd" opacity=".55"/>
    <path d="M0 210 C260 150 480 250 760 200 C1040 150 1240 240 1440 190 L1440 400 L0 400Z" fill="#a9c1a2" opacity=".75"/>
    <path d="M0 280 C300 230 520 320 820 270 C1100 225 1280 300 1440 262 L1440 400 L0 400Z" fill="#7f9f7b"/>
  </svg>

  <div class="cv-inner">
    <div class="cv-eyebrow">iGEM IIT Delhi &middot; 2026</div>
    <div class="cv-title">Derma<em>Sense</em></div>
    <div class="cv-line">Skin that answers the air. A living moisturiser, and the mathematics
      that decides how much it should give.</div>
    <div class="cv-pills">
      <span class="cv-pill">Pulcherrimin</span>
      <span class="cv-pill">Xylolipid</span>
      <span class="cv-pill">Lyso-ornithine lipid</span>
    </div>
  </div>
  <div class="cv-foot">Mathematical model &middot; v3</div>
</div>
""").strip()

st.markdown(COVER_CSS, unsafe_allow_html=True)
st.markdown(COVER_HTML, unsafe_allow_html=True)

_, mid, _ = st.columns([1, 1, 1])
with mid:
    st.page_link("views/home.py", label="Enter", icon=":material/arrow_forward:")
