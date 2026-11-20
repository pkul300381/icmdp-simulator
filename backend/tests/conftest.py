import os, sys
os.environ.setdefault("ICMDP_DB", ":memory:")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
