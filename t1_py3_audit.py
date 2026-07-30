#!/usr/bin/env python3
"""
T1 - Static Python-3 audit. Runs anywhere, needs no TensorFlow, no model, no video.

py_compile only PARSES. It cannot see semantic Py2->Py3 breaks. This script
looks for the four classes of break that compile cleanly but fail at runtime:

  A. removed builtins            long(), unicode(), basestring, raw_input, execfile
  B. moved builtins              bare reduce() without functools import
  C. iterator-vs-list            map()/filter()/zip() consumed as a sequence
  D. classic-vs-true division    int/int used as an index, shape, or count

Exit code 1 if any Class A/B/C finding, so you can wire it into CI.

Usage:  python3 t1_py3_audit.py <repo_root>
"""
import ast
import os
import sys


REMOVED_BUILTINS = {"long", "unicode", "basestring", "raw_input", "execfile", "xrange", "cmp"}

# Names that, when a bare `/` result flows into them, almost certainly wanted //
INT_CONTEXT_HINTS = (
    "shape", "size", "len", "dim", "idx", "index", "count", "stride",
    "frames", "samples", "rate", "pad", "crop", "offset", "start", "end",
)


class Auditor(ast.NodeVisitor):
    def __init__(self, path, src):
        self.path = path
        self.lines = src.splitlines()
        self.findings = []
        self.has_functools_reduce = False

    def line(self, node):
        try:
            return self.lines[node.lineno - 1].strip()
        except IndexError:
            return ""

    def add(self, cls, node, msg):
        self.findings.append((cls, self.path, node.lineno, msg, self.line(node)))

    # ---- imports -------------------------------------------------------
    def visit_ImportFrom(self, node):
        if node.module == "functools":
            for a in node.names:
                if a.name == "reduce":
                    self.has_functools_reduce = True
        # `from io import StringIO` then using `StringIO.StringIO(...)`
        if node.module == "io":
            for a in node.names:
                if a.name == "StringIO":
                    self._io_stringio = True
        self.generic_visit(node)

    # ---- calls ---------------------------------------------------------
    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name):
            if f.id in REMOVED_BUILTINS:
                self.add("A", node, "`%s()` was removed in Python 3" % f.id)
            if f.id == "reduce" and not self.has_functools_reduce:
                self.add("B", node, "bare `reduce()` - moved to functools in Py3")
        # StringIO.StringIO() when StringIO was imported from io
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "StringIO"
            and f.attr == "StringIO"
            and getattr(self, "_io_stringio", False)
        ):
            self.add("A", node, "`StringIO.StringIO()` but StringIO was imported from io")

        # map/filter/zip wrapped in something that needs a real sequence
        if isinstance(f, ast.Name) and f.id in ("np", "array"):
            pass
        for arg in node.args:
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                if arg.func.id in ("map", "filter", "zip"):
                    callee = _name_of(f)
                    if callee and not callee.startswith(("list", "tuple", "set", "dict", "sorted", "sum", "any", "all", "iter", "enumerate")):
                        self.add(
                            "C",
                            node,
                            "`%s()` returns an iterator in Py3, passed straight into `%s()`"
                            % (arg.func.id, callee),
                        )
        self.generic_visit(node)

    # ---- subscripts: x[a / b] -----------------------------------------
    def visit_Subscript(self, node):
        sl = node.slice
        for sub in ast.walk(sl):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div):
                self.add("D", node, "true-division result used as an INDEX (Py2 floored here)")
                break
        self.generic_visit(node)

    # ---- assignments whose target name smells integral -----------------
    def visit_Assign(self, node):
        if not isinstance(node.value, ast.BinOp) or not isinstance(node.value.op, ast.Div):
            self.generic_visit(node)
            return
        if _mentions_float(node.value):
            self.generic_visit(node)
            return
        for t in node.targets:
            n = _name_of(t) or ""
            if any(h in n.lower() for h in INT_CONTEXT_HINTS) or _mentions_int_context(node.value):
                self.add("D", node, "`/` assigned to integral-looking name `%s`" % n)
                break
        self.generic_visit(node)


def _name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return (base + "." + node.attr) if base else node.attr
    return None


def _mentions_float(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
            return True
        if isinstance(sub, ast.Call) and _name_of(sub.func) in ("float", "np.float32", "np.float64"):
            return True
    return False


def _mentions_int_context(node):
    for sub in ast.walk(node):
        n = _name_of(sub) or ""
        if any(h in n.lower() for h in INT_CONTEXT_HINTS):
            return True
    return False


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    all_findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                tree = ast.parse(src, filename=p)
            except SyntaxError as e:
                all_findings.append(("PARSE", p, e.lineno or 0, str(e), ""))
                continue
            a = Auditor(os.path.relpath(p, root), src)
            a.visit(tree)
            all_findings.extend(a.findings)

    order = {"PARSE": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    all_findings.sort(key=lambda f: (order.get(f[0], 9), f[1], f[2]))

    labels = {
        "PARSE": "SYNTAX ERROR",
        "A": "CLASS A - removed builtin (certain NameError)",
        "B": "CLASS B - moved builtin (certain NameError)",
        "C": "CLASS C - iterator used as sequence (silent wrong result)",
        "D": "CLASS D - division semantics (TypeError or silent wrong result)",
    }
    current = None
    counts = {}
    for cls, path, lineno, msg, text in all_findings:
        counts[cls] = counts.get(cls, 0) + 1
        if cls != current:
            print("\n" + "=" * 78)
            print(labels.get(cls, cls))
            print("=" * 78)
            current = cls
        print("%s:%d" % (path, lineno))
        print("    %s" % msg)
        if text:
            print("    | %s" % text)

    print("\n" + "-" * 78)
    print("SUMMARY")
    for c in ("PARSE", "A", "B", "C", "D"):
        if counts.get(c):
            print("  %-6s %d" % (c, counts[c]))
    if not counts:
        print("  clean")
    print("-" * 78)

    blocking = sum(counts.get(c, 0) for c in ("PARSE", "A", "B", "C"))
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
