"""Every GameUI part on one page, in a given style: used to check the design system itself (each style must pass the
layout checker at desktop and phone size) and as the example the AI is shown."""
import os

KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kit")
THEMES = ["clean", "pixel", "neon", "cozy"]


def css():
    return open(os.path.join(KIT, "gameui.css"), encoding="utf-8").read()


def page(body, theme="clean", title="Game"):
    """a full page from GameUI parts (the stylesheet is inlined, so it works offline and as one file)"""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{title}</title><style>html,body{{margin:0}}{css()}</style></head><body>"
            f'<div class="gui" data-theme="{theme}">{body}</div></body></html>')


SHOWCASE = """
<div class="gui-hud"><div class="gui-stat"><span>Score</span><b>1,240</b></div><div class="gui-stat"><span>Lives</span><b>3</b></div>
 <div class="gui-bar" role="progressbar" aria-label="Health" aria-valuenow="70"><i style="width:70%"></i></div><div class="gui-stat"><span>Time</span><b>1:32</b></div></div>
<div class="gui-screen"><h1 class="gui-title">Gem Cave</h1><p class="gui-subtitle">Collect every gem, dodge the bats</p>
 <div class="gui-menu"><button class="gui-btn gui-primary">Play</button><button class="gui-btn">Settings</button><button class="gui-btn gui-danger">Quit</button></div></div>
<div class="gui-row">
 <div class="gui-panel"><h2>Settings</h2><ul class="gui-list">
  <li class="gui-item"><label for="s1">Sound</label><input id="s1" type="checkbox" class="gui-toggle" checked></li>
  <li class="gui-item"><label for="s2">Music volume</label><input id="s2" type="range" class="gui-range" value="60"></li>
  <li class="gui-item"><label for="s3">Screen shake</label><input id="s3" type="checkbox" class="gui-toggle"></li></ul></div>
 <div class="gui-panel"><h2>High scores</h2><table class="gui-table"><tr><th>Name</th><th>Score</th></tr>
  <tr><td>Ava</td><td>2,310</td></tr><tr><td>Leo</td><td>1,240</td></tr><tr><td>Mia</td><td>980</td></tr></table></div>
</div>
<div class="gui-row"><div class="gui-panel"><h2>Inventory</h2><div class="gui-grid">
 <div class="gui-slot gui-selected"><b>🗝</b>Key</div><div class="gui-slot"><b>💎</b>Gem</div><div class="gui-slot"><b>❤</b>Heart</div><div class="gui-slot"><b>🛡</b>Shield</div></div></div>
 <div class="gui-dialog"><span class="gui-speaker">Old Miner</span><p>The bats only come out when the torches go dark. Keep moving!</p>
  <div class="gui-row"><button class="gui-btn gui-primary">Got it</button><button class="gui-btn">Tell me more</button></div></div></div>
<div class="gui-toast">Checkpoint saved</div>
"""
