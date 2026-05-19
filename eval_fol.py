#!/usr/bin/env python3

import argparse, json, re, sys
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

try:
    import z3
except ImportError:
    z3 = None

# =============================
# Tokenizer
# =============================
TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in [
    ("WS",      r"\s+"),
    ("FORALL",  r"\u2200|\\forall"),
    ("EXISTS",  r"\u2203|\\exists"),
    ("NOT",     r"\u00ac|~|!|\\neg"),
    ("AND",     r"\u2227|&|\\wedge"),
    ("OR",      r"\u2228|\||\\vee"),
    ("IMPLIES", r"\u2192|->|=>|\\to"),
    ("IFF",     r"\u2194|<->|<=>|\\leftrightarrow"),
    ("XOR",     r"\u2295|\\oplus"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("COMMA",   r","),
    ("DOT",     r"\."),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
]))

@dataclass(frozen=True)
class Token:
    typ: str; val: str; pos: int

def tokenize(s):
    out, i = [], 0
    while i < len(s):
        m = TOKEN_RE.match(s, i)
        if not m: raise ValueError(f"Tokenizer error at pos {i}")
        if m.lastgroup != "WS": out.append(Token(m.lastgroup, m.group(m.lastgroup), i))
        i = m.end()
    return out

# =============================
# AST
# =============================
@dataclass(frozen=True)
class Term:
    kind: str; name: str

@dataclass(frozen=True)
class Formula:
    kind: str
    name: Optional[str] = None
    args: Optional[Tuple[Term, ...]] = None
    a: Optional["Formula"] = None
    b: Optional["Formula"] = None
    var: Optional[str] = None
    body: Optional["Formula"] = None

# =============================
# Parser
# =============================
class Parser:
    def __init__(self, toks): self.toks = toks; self.i = 0
    def peek(self): return self.toks[self.i] if self.i < len(self.toks) else None
    def accept(self, typ):
        t = self.peek()
        if t and t.typ == typ: self.i += 1; return t
    def expect(self, typ):
        t = self.peek()
        if not t or t.typ != typ: raise ValueError(f"Expected {typ}")
        self.i += 1; return t

    def parse(self):
        f = self._iff()
        if self.peek(): raise ValueError("Unexpected token")
        return f

    def _iff(self):
        l = self._xor()
        while self.accept("IFF"): l = Formula("iff", a=l, b=self._xor())
        return l
    def _xor(self):
        l = self._implies()
        while self.accept("XOR"): l = Formula("xor", a=l, b=self._implies())
        return l
    def _implies(self):
        l = self._or()
        while self.accept("IMPLIES"): l = Formula("implies", a=l, b=self._or())
        return l
    def _or(self):
        l = self._and()
        while self.accept("OR"): l = Formula("or", a=l, b=self._and())
        return l
    def _and(self):
        l = self._unary()
        while self.accept("AND"): l = Formula("and", a=l, b=self._unary())
        return l

    def _unary(self):
        if self.accept("NOT"): return Formula("not", a=self._unary())
        t = self.peek()
        if t and t.typ in ("FORALL", "EXISTS"): return self._quant()
        if self.accept("LPAREN"):
            f = self._iff(); self.expect("RPAREN"); return f
        return self._atom()

    def _quant(self):
        q = self.peek(); self.i += 1
        qk = "forall" if q.typ == "FORALL" else "exists"
        vs = []
        while (t := self.peek()) and t.typ == "IDENT": vs.append(t.val); self.i += 1
        if not vs: raise ValueError("Quantifier missing variable")
        self.accept("DOT")
        body = self._unary()
        for v in reversed(vs): body = Formula(qk, var=v, body=body)
        return body

    def _atom(self):
        t = self.expect("IDENT"); name = t.val
        if self.accept("LPAREN"):
            args = []
            if not self.accept("RPAREN"):
                args.append(self._term())
                while self.accept("COMMA"): args.append(self._term())
                self.expect("RPAREN")
            return Formula("pred", name=name, args=tuple(args))
        m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)([a-z])", name)
        if m: return Formula("pred", name=m[1], args=(Term("var", m[2]),))
        return Formula("pred", name=name, args=())

    def _term(self):
        t = self.expect("IDENT")
        return Term("var" if re.fullmatch(r"[a-z]", t.val) else "const", t.val)


def parse_formula(s): return Parser(tokenize(s)).parse()

# =============================
# Normalization
# =============================
def expand_xor(f):
    if f.kind == "xor":
        A, B = expand_xor(f.a), expand_xor(f.b)
        return Formula("and", a=Formula("or", a=A, b=B),
                        b=Formula("not", a=Formula("and", a=A, b=B)))
    if f.kind == "not": return Formula("not", a=expand_xor(f.a))
    if f.kind in ("and","or","implies","iff"):
        return Formula(f.kind, a=expand_xor(f.a), b=expand_xor(f.b))
    if f.kind in ("forall","exists"):
        return Formula(f.kind, var=f.var, body=expand_xor(f.body))
    return f

def alpha_rename(f):
    ctr = {"n": 0}
    def go(node, env):
        if node.kind in ("forall","exists"):
            new = f"v{ctr['n']}"; ctr["n"] += 1
            env2 = {**env, node.var: new}
            return Formula(node.kind, var=new, body=go(node.body, env2))
        if node.kind == "pred":
            return Formula("pred", name=node.name,
                args=tuple(Term("var", env[t.name])
                           if t.kind == "var" and t.name in env else t
                           for t in (node.args or ())))
        if node.kind == "not": return Formula("not", a=go(node.a, env))
        if node.kind in ("and","or","implies","iff","xor"):
            return Formula(node.kind, a=go(node.a, env), b=go(node.b, env))
        raise ValueError(node.kind)
    return go(f, {})

def normalize(f): return alpha_rename(expand_xor(f))

# =============================
# Free-variable closure
# =============================
def free_vars(f, bound=None):
    bound = bound or set()
    if f.kind == "pred":
        return {t.name for t in (f.args or ())
                if t.kind == "var" and t.name not in bound}
    if f.kind in ("forall","exists"):
        return free_vars(f.body, bound | {f.var})
    if f.kind == "not": return free_vars(f.a, bound)
    if f.kind in ("and","or","implies","iff","xor"):
        return free_vars(f.a, bound) | free_vars(f.b, bound)
    return set()

def close_universally(f):
    for v in sorted(free_vars(f)):
        f = Formula("forall", var=v, body=f)
    return f

# =============================
# Signature collection 
# =============================
def collect_signature(f):
    preds, consts = {}, set()
    def go(n):
        if n.kind == "pred":
            preds.setdefault(n.name, len(n.args or ()))
            for t in (n.args or ()):
                if t.kind == "const": consts.add(t.name)
        elif n.kind in ("forall","exists"): go(n.body)
        elif n.kind == "not": go(n.a)
        elif n.kind in ("and","or","implies","iff","xor"): go(n.a); go(n.b)
    go(f); return preds, consts

# =============================
# Bounded semantic equivalence (Z3)
# =============================
def bounded_equiv_check(gold, pred, max_domain=4):
    if z3 is None: return "NO_Z3"
    gold, pred = close_universally(gold), close_universally(pred)
    g_p, g_c = collect_signature(gold)
    p_p, p_c = collect_signature(pred)
    sigs = {**g_p, **{k: v for k, v in p_p.items() if k not in g_p}}
    consts = sorted(set(g_c) | set(p_c))
    cmap = {c: i % max(1, max_domain) for i, c in enumerate(consts)}

    def build(f, dom_n, pf):
        def ev_term(t, env):
            if t.kind == "var": return env[t.name]
            return z3.IntVal(cmap.get(t.name, 0))
        def ev(node, env):
            if node.kind == "pred":
                nm = node.name or ""; ar = len(node.args or ())
                if nm not in pf:
                    pf[nm] = (z3.Bool(nm) if ar == 0
                              else z3.Function(nm, *([z3.IntSort()]*ar
                                                     + [z3.BoolSort()])))
                return (pf[nm] if ar == 0
                        else pf[nm](*[ev_term(t, env) for t in node.args]))
            if node.kind == "not": return z3.Not(ev(node.a, env))
            if node.kind == "and": return z3.And(ev(node.a, env), ev(node.b, env))
            if node.kind == "or":  return z3.Or(ev(node.a, env), ev(node.b, env))
            if node.kind == "implies":
                return z3.Implies(ev(node.a, env), ev(node.b, env))
            if node.kind == "iff": return ev(node.a, env) == ev(node.b, env)
            if node.kind == "xor": return z3.Xor(ev(node.a, env), ev(node.b, env))
            if node.kind in ("forall","exists"):
                inst = [ev(node.body, {**env, node.var: z3.IntVal(v)})
                        for v in range(dom_n)]
                return ((z3.And if node.kind == "forall" else z3.Or)(inst)
                        if inst else z3.BoolVal(node.kind == "forall"))
            raise ValueError(node.kind)
        return ev(f, {})

    for n in range(1, max_domain + 1):
        pf = {}
        for p, ar in sigs.items():
            pf[p] = (z3.Bool(p) if ar == 0
                     else z3.Function(p, *([z3.IntSort()]*ar + [z3.BoolSort()])))
        g_z = build(gold, n, pf)
        p_z = build(pred, n, pf)
        s = z3.Solver()
        s.add(z3.Or(z3.And(g_z, z3.Not(p_z)), z3.And(p_z, z3.Not(g_z))))
        if s.check() == z3.sat:
            return "SAT"
    return "UNSAT"

# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser(description="Evaluate NL->FOL: SynV + BSC")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--max-domain", type=int, default=4)
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f: gold_data = json.load(f)
    with open(args.pred, encoding="utf-8") as f: pred_data = json.load(f)
    if len(gold_data) != len(pred_data):
        sys.exit(f"Length mismatch: gold={len(gold_data)} pred={len(pred_data)}")

    n = len(gold_data)
    parsed, bsc = 0, 0

    for g, p in zip(gold_data, pred_data):
        try:
            g_f = parse_formula(g["FOL"])
            p_f = parse_formula(p["FOL"])
        except Exception:
            continue
        parsed += 1
        try:
            g_n, p_n = normalize(g_f), normalize(p_f)
            if bounded_equiv_check(g_n, p_n, args.max_domain) == "UNSAT":
                bsc += 1
        except Exception:
            pass

    print(f"Examples: {n}")
    print(f"SynV: {parsed}/{n} ({100*parsed/n:.1f}%)")
    print(f"BSC:  {bsc}/{parsed} ({100*bsc/max(1,parsed):.1f}%)")

if __name__ == "__main__":
    main()
