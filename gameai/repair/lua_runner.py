"""The Lua sandbox's test runner (plan: Phase 4, code sandboxes per language). Lua 5.4 runs inside Python (the lupa
package), in its own environment that has the language's safe parts only: no io, os.execute, debug, load, dofile or
outside files. `require` reaches the project's own .lua files only. Memory is capped inside Lua (512 MB) and each
test has a time limit; debugger.py adds an outer limit too.

    python -I lua_runner.py config.json   ->  one JSON object on stdout (the JavaScript runner's shape)

Tests: files named *_test.lua, *.test.lua or inside tests/, using
    test("name", function() expect(x).toBe(3) end)      -- also toEqual, toBeTruthy, toBeFalsy, toBeGreaterThan,
    assert(cond, "message")                              -- toBeLessThan, toContain, toThrow, and expect(x).never.toBe(y)
Lua is the plain language (5.4); Roblox's Luau extras (type annotations, +=, continue) aren't supported.
"""
import json, os, sys, time

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
ROOT = os.path.realpath(cfg["root"])
LIMIT_S = cfg.get("test_timeout_ms", 10000) / 1000
out = {"load_errors": [], "tests": [], "console": [], "passed": 0, "failed": 0}

from lupa import lua54  # noqa: E402

lua = lua54.LuaRuntime(unpack_returned_tuples=True, register_eval=False, register_builtins=False, max_memory=cfg.get("memory_mb", 512) * 2**20)

# The host side (this file) keeps load and debug; the project's environment never sees them.
HOST = lua.execute(r'''
local load, debug, os_clock, xpcall, traceback = load, debug, os.clock, xpcall, debug.traceback
local H = {}

local function fmt(v)
  local t = type(v)
  if t == "string" then return string.format("%q", #v > 80 and v:sub(1, 80) .. "..." or v) end
  if t ~= "table" then return tostring(v) end
  local parts, n = {}, 0
  for k, x in pairs(v) do
    n = n + 1; if n > 8 then parts[#parts + 1] = "..."; break end
    parts[#parts + 1] = (type(k) == "number" and "" or tostring(k) .. " = ") .. (type(x) == "table" and "{...}" or (type(x) == "string" and string.format("%q", x) or tostring(x)))
  end
  return "{" .. table.concat(parts, ", ") .. "}"
end
H.fmt = fmt

function H.env(root_reader, say)
  local E = {}
  for _, k in ipairs({"assert", "error", "ipairs", "pairs", "next", "pcall", "xpcall", "select", "tonumber", "tostring", "type",
                      "rawequal", "rawget", "rawset", "rawlen", "setmetatable", "getmetatable"}) do E[k] = _G[k] end
  local function copy(t) local c = {} for k, v in pairs(t) do c[k] = v end return c end
  E.string, E.table, E.math, E.utf8, E.coroutine = copy(string), copy(table), copy(math), copy(utf8), copy(coroutine)
  E.os = {time = os.time, clock = os.clock, date = os.date, difftime = os.difftime}
  E.unpack = table.unpack
  E.print = function(...) local p = {} for i = 1, select("#", ...) do p[i] = tostring((select(i, ...))) end say("print", table.concat(p, "\t")) end
  E._G = E
  local loaded = {}
  E.require = function(name)
    if loaded[name] ~= nil then return loaded[name] end
    local path = tostring(name):gsub("%.", "/") .. ".lua"
    local src = root_reader(path)
    if not src then error("module '" .. tostring(name) .. "' not found in the project (only the project's own files can be required)", 2) end
    local chunk, err = load(src, "@" .. path, "t", E)
    if not chunk then error(err, 0) end
    local r = chunk(name); if r == nil then r = true end
    loaded[name] = r
    return r
  end
  -- tests
  local tests = {}
  E.test = function(name, fn) tests[#tests + 1] = {name = tostring(name), fn = fn, file = H.current_file} end
  local function AssertionError(msg, actual, expected) return setmetatable({message = msg, actual = fmt(actual), expected = fmt(expected), assertion = true},
    {__tostring = function(s) return s.message end}) end
  local function deq(a, b) return fmt(a) == fmt(b) end
  E.expect = function(actual)
    local function fail(what, expected) error(AssertionError("expected " .. fmt(actual) .. " " .. what .. (expected == nil and "" or " " .. fmt(expected)), actual, expected), 3) end
    local m = {
      toBe = function(e) if actual ~= e then fail("to be", e) end return true end,
      toEqual = function(e) if not deq(actual, e) then fail("to equal", e) end return true end,
      toBeTruthy = function() if not actual then fail("to be truthy") end return true end,
      toBeFalsy = function() if actual then fail("to be falsy") end return true end,
      toBeGreaterThan = function(e) if not (actual > e) then fail("to be greater than", e) end return true end,
      toBeLessThan = function(e) if not (actual < e) then fail("to be less than", e) end return true end,
      toContain = function(e)
        if type(actual) == "string" then if not actual:find(e, 1, true) then fail("to contain", e) end return true end
        for _, v in pairs(actual or {}) do if v == e then return true end end
        fail("to contain", e)
      end,
      toThrow = function() if pcall(actual) then fail("to throw") end return true end,
    }
    m.never = {}
    for k, f in pairs(m) do if k ~= "never" then m.never[k] = function(...) if pcall(f, ...) then error(AssertionError("expected " .. fmt(actual) .. " not " .. k, actual, (...)), 2) end return true end end end
    return m
  end
  E.assert = function(c, msg, ...) if not c then error(AssertionError(msg or "assertion failed", c, true), 2) end return c, msg, ... end
  H.tests = tests
  return E
end

-- run fn with a time limit; on error collect the project's frames with their local variables
function H.run(fn, limit)
  local deadline = os_clock() + limit
  debug.sethook(function() if os_clock() > deadline then debug.sethook(); error("took longer than " .. limit .. "s: probably an endless loop", 2) end end, "", 10000)
  local frames
  local ok, err = xpcall(fn, function(e)
    frames = {}
    for level = 2, 40 do
      local info = debug.getinfo(level, "Sln")
      if not info then break end
      if info.source:sub(1, 1) == "@" and info.currentline > 0 then
        local f = {["function"] = info.name or (info.what == "main" and "(top level)" or "(function)"), file = info.source:sub(2), line = info.currentline}
        if #frames < 2 then
          local vars, i = {}, 1
          while i < 30 do
            local name, value = debug.getlocal(level, i)
            if not name then break end
            if name:sub(1, 1) ~= "(" then vars[name] = fmt(value) end
            i = i + 1
          end
          f.variables = vars
        end
        frames[#frames + 1] = f
        if #frames >= 6 then break end
      end
    end
    return e
  end)
  debug.sethook()
  return ok, err, frames
end

function H.load_file(src, name, env) return load(src, "@" .. name, "t", env) end
return H
''')


def read(path):
    """the project's own .lua files only"""
    p = os.path.realpath(os.path.join(ROOT, *str(path).split("/")))
    if not p.startswith(ROOT + os.sep) or not p.endswith(".lua") or not os.path.isfile(p): return None
    return open(p, encoding="utf-8").read()


current_test = [None]
def say(level, text):
    out["console"].append({"level": level, "message": str(text)[:300], "file": None, "line": None, "test": current_test[0]})


lines = {}
def source(file, n):
    if file not in lines:
        t = read(file); lines[file] = t.split("\n") if t else []
    return lines[file][n - 1].strip() if 0 < n <= len(lines[file]) else ""


def failure(err, frames):
    if lua.eval("type")(err) == "table" and err["assertion"]:
        r = {"error": "AssertionError: " + str(err["message"]), "assertion": {"actual": str(err["actual"]), "expected": str(err["expected"])}}
    else:
        r = {"error": str(HOST.fmt(err)).strip('"') if lua.eval("type")(err) != "string" else str(err)}
    st = []
    for f in (frames.values() if frames else []):
        fr = {"function": str(f["function"]), "file": str(f["file"]), "line": int(f["line"])}
        fr["source"] = source(fr["file"], fr["line"])
        if f["variables"]: fr["variables"] = {str(k): str(v) for k, v in f["variables"].items()}
        st.append(fr)
    r["stack"] = st
    return r


env = HOST.env(read, say)
for test_file in cfg["tests"]:
    src = read(test_file)
    HOST.current_file = test_file
    got = HOST.load_file(src or "", test_file, env)  # one value when it compiles, (nil, message) when it doesn't
    chunk, err = got if isinstance(got, tuple) else (got, None)
    if not chunk:
        m = __import__("re").match(r"(.+?):(\d+): (.*)", str(err))
        out["load_errors"].append({"file": test_file, "error": "SyntaxError: " + (m.group(3) if m else str(err)),
                                   "stack": [{"function": "(syntax)", "file": test_file, "line": int(m.group(2)) if m else 1, "source": source(test_file, int(m.group(2))) if m else ""}]})
        continue
    try:
        ok, err, frames = HOST.run(chunk, LIMIT_S)
    except lua54.LuaMemoryError:
        ok, err, frames = False, "used more than the sandbox's memory limit", None
    if not ok: out["load_errors"].append({"file": test_file, **failure(err, frames)})
for t in list(HOST.tests.values()):
    name = str(t["name"]); current_test[0] = name
    r = {"name": name, "file": str(t["file"])}
    sys.stderr.write(f"running: {name}\n"); sys.stderr.flush()
    t0 = time.time()
    try:
        ok, err, frames = HOST.run(t["fn"], LIMIT_S)
    except lua54.LuaMemoryError:
        ok, err, frames = False, "used more than the sandbox's memory limit", None
    r["passed"] = bool(ok)
    if not ok:
        r.update(failure(err, frames))
        for fr in r["stack"]:  # the test's own function has no name in Lua: call it by the test's name
            if fr["function"] == "(function)" and fr["file"] == r["file"]: fr["function"] = 'test "%s"' % name
    r["ms"] = int((time.time() - t0) * 1000)
    out["tests"].append(r); out["passed" if ok else "failed"] += 1
sys.stdout.write(json.dumps(out))
