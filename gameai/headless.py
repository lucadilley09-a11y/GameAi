"""The game engine and its playtester bots, run without a browser (in the V8 engine that mini-racer bundles).

The page's own engine code is used unchanged (the part between the engine markers in play_template.html), with
the same small stand-ins the page's background workers use, so results match what the page does.

    from gameai.headless import Engine
    e = Engine(); e.judge(resolved_recipe)   # -> the same dict the page's playtesters return
"""
import json, os, random

HERE = os.path.dirname(__file__)
SHIM = ("var self=this;var window=this;var performance={now:()=>Date.now()};var addEventListener=()=>0;"
        "var localStorage={getItem:()=>null,setItem(){}};var requestAnimationFrame=()=>0;var navigator={hardwareConcurrency:1};"
        "const document={getElementById:()=>({getContext:()=>({}),addEventListener(){},focus(){},classList:{add(){},remove(){},toggle(){}},style:{}}),addEventListener(){}};"
        "const $=id=>document.getElementById(id),rnd=a=>a[Math.random()*a.length|0],ri=n=>Math.random()*n|0,"
        "shuffle=a=>{for(let i=a.length-1;i>0;i--){const j=ri(i+1);[a[i],a[j]]=[a[j],a[i]]}return a};"
        "let F={},FN=[],trained=true;"
        # V8 on its own has no structuredClone (browsers do): a faithful copy for the explorer bot's snapshots
        "var structuredClone=globalThis.structuredClone||function sc(v,seen){seen=seen||new Map();if(v===null||typeof v!=='object')return v;"
        "if(seen.has(v))return seen.get(v);let o;if(Array.isArray(v)){o=[];seen.set(v,o);for(let i=0;i<v.length;i++)o[i]=sc(v[i],seen);return o}"
        "if(ArrayBuffer.isView(v))return v.slice();if(v instanceof Date)return new Date(v.getTime());"
        "if(v instanceof Map){o=new Map();seen.set(v,o);v.forEach((x,k)=>o.set(sc(k,seen),sc(x,seen)));return o}"
        "if(v instanceof Set){o=new Set();seen.set(v,o);v.forEach(x=>o.add(sc(x,seen)));return o}"
        "o={};seen.set(v,o);for(const k of Object.keys(v))o[k]=sc(v[k],seen);return o};\n")


def engine_source():
    play = open(os.path.join(HERE, "play_template.html"), encoding="utf-8").read()
    return play[play.index("// ---------- engine ----------"):play.index("// ---------- prompt -> recipe -> game ----------")]


def random_recipe(rng, **fixed):
    """A resolved recipe like the page builds: random where nothing was asked."""
    from .recipe import resolve, FIELD_NAMES, OUT_FIELDS, rules_of, fill_defaults
    rec = {f: "auto" for f in OUT_FIELDS}; fill_defaults(rec)
    for f in OUT_FIELDS:
        if f.startswith("r") and f[1:2].isdigit(): rec[f] = "none"
    r = resolve({f: rec[f] for f in FIELD_NAMES}, rng)
    r["rules"] = rules_of(rec); r.update(fixed); r.setdefault("seed", rng.randrange(2**31))
    return r


class Engine:
    def __init__(self, timeout_s=120):
        from py_mini_racer import MiniRacer
        self.ctx, self.timeout = MiniRacer(), timeout_s
        self.ctx.eval(SHIM + engine_source())

    def judge(self, R):
        return json.loads(self.ctx.eval(f"JSON.stringify(judge({json.dumps(R)}))", timeout_sec=self.timeout))

    def close(self):
        try: self.ctx.close()
        except Exception: pass


def _worker_judge(args):
    """for process pools: each worker process keeps one engine"""
    global _ENG
    try: _ENG
    except NameError: _ENG = Engine()
    t_R = args
    return _ENG.judge(t_R)
