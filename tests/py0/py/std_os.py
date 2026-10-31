"""More of os / os.path, sys and time, and subprocess.run (src/pyrt/lib)."""
import os
import sys
import time
import subprocess

def main() -> None:
    d = "py0_os_test_" + str(os.getpid())
    os.makedirs(d + "/a/b", exist_ok=True)
    try:
        os.makedirs(d)
    except OSError as e:
        print("exists", str(e).startswith("[Errno 17] File exists"))
    with open(d + "/a/x.txt", "w") as f:
        f.write("hi")
    print(sorted(os.listdir(d + "/a")))
    print(os.path.isdir(d), os.path.isfile(d + "/a/x.txt"), os.path.getsize(d + "/a/x.txt"))
    print(os.path.normpath("a//b/./c/../d"), os.path.dirname("/x/y/z.py"), os.path.basename("/x/y/z.py"))
    print(os.path.splitext("a/b.tar.gz"), os.path.splitext(".bashrc"), os.path.relpath("/a/b/c", "/a/x"))
    print(os.path.abspath(d) == os.path.join(os.getcwd(), d))
    print(os.environ.get("PY0_NOPE"), os.environ.get("PY0_NOPE", "dflt"))
    try:
        os.listdir("no/such/dir")
    except FileNotFoundError as e:
        print("missing:", e)
    r = subprocess.run("echo hello; echo err 1>&2; exit 3", shell=True, capture_output=True, text=True)
    print(r.returncode, repr(r.stdout), repr(r.stderr))
    r2 = subprocess.run("ls", shell=True, cwd=d, capture_output=True, text=True, timeout=10)
    print(r2.stdout.split())
    try:
        subprocess.run("sleep 2", shell=True, timeout=0.2)
    except Exception as e:
        print("timeout")
    print(len(time.strftime("%Y-%m-%d %H:%M:%S")))
    print(sys.stdin.isatty() == sys.stdin.isatty())
    os.remove(d + "/a/x.txt")
    os.rmdir(d + "/a/b")
    os.rmdir(d + "/a")
    os.rmdir(d)
    print(os.path.exists(d))

main()
