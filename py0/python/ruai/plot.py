"""CPython stand-in for the py0 package `ruai.plot` (src/pyrt/lib/ruai_plot.rs):
matplotlib.pyplot itself; `savefig` also writes .html (the SVG in a page)."""

import io as _io

import matplotlib.pyplot as _plt
from matplotlib.figure import Figure as _Figure
from matplotlib.pyplot import *  # noqa: F401,F403

_savefig = _Figure.savefig


def _savefig_html(self, fname, *args, **kwargs):
    if isinstance(fname, str) and fname.lower().endswith((".html", ".htm")):
        buf = _io.StringIO()
        _savefig(self, buf, *args, format="svg", **kwargs)
        with open(fname, "w") as f:
            f.write("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>ruai.plot</title></head><body>\n")
            f.write(buf.getvalue())
            f.write("</body></html>\n")
        return
    _savefig(self, fname, *args, **kwargs)


_Figure.savefig = _savefig_html
savefig = _plt.savefig
