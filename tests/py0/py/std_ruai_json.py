"""ruai.json: a JSON value with CPython's json formatting and errors."""
from ruai.json import Json


def show(j: Json) -> None:
    print(j.kind(), j.dumps(), j.dumps(indent=2, ensure_ascii=False))


def main() -> None:
    src = '{"name": "agge", "n": 3, "pi": 3.25, "big": 1e16, "ok": true, "none": null, "xs": [1, "two", [], {}], "zh": "中文\\n", "esc": "\\ud83d\\ude00"}'
    j = Json.loads(src)
    show(j)
    print(j.get("name").str_or("?"), j.get("n").int_or(-1), j.get("pi").float_or(0.0), j.get("n").float_or(0.0))
    print(j.get("missing").is_null(), j.has("ok"), j.has("nope"), j.keys())
    print(j.get("xs").size(), j.get("xs").at(1).text(), j.get("xs").at(-1).kind(), j.get("xs").at(9).kind())
    print([x.text() for x in j.get("xs").items()], j.get("xs").strs())
    m = Json.new_dict()
    m.set("a", Json.of_int(1))
    m.set("b", Json.of_strs(["x", "y"]))
    m.set("a", Json.of_float(2.5))
    sub = m.get("b")
    sub.append(Json.of_str("z"))
    print(m.dumps(), sub.dumps())
    m.remove("b")
    print(m.dumps(indent=0), m.truthy(), Json.new_list().truthy())
    print(Json.of_bool(False).dumps(), Json.null().dumps(), Json.of_str("é\t").dumps())
    for bad in ["", "[1,", '{"a" 1}', "[1] x", '"abc', "{'a': 1}"]:
        try:
            Json.loads(bad)
            print("parsed?")
        except ValueError as e:
            print("error:", e)
    print(j.text())


main()
