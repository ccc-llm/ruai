import html
from html import escape, unescape

for s in ["", "plain", "<a href=\"x\">'q' & r</a>", "&amp;", "é < ü"]:
    print(escape(s), html.escape(s, False), escape(s, quote=False))
refs: list[str] = [
    "&amp;", "&lt;&gt;", "&#62;", "&#x3e;", "&#X3E", "&#0;", "&#128;", "&#xD800;", "&#x110000;",
    "&#99999999999999999999;", "&#1;", "&#x0b;", "&ampx", "&amp", "&notit;", "&notin;", "&nosuch;",
    "&nos", "&#;", "&#x;", "&", "a & b", "&;", "&#65&#66", "&AElig&AElig;", "caf&eacute;",
    "&#x1F600;", "&lt3", "&ThisIsAVeryLongNameThatExceedsThirtyTwoCharacters;", "&#x9F;", "&#13;",
    "&gt\t&lt", "x&#y", "&&amp;", "&#00065;", "&quot;&apos;&nbsp;|", "&Eacuteric", "&#xFFFE;",
]
for r in refs:
    print(repr(r), repr(unescape(r)))
print(unescape(escape("<'round' & \"trip\">")))
