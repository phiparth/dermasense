"""Headless render of one page, reporting any exception it raises.

A dev convenience, not part of the test suite. Pages that call st.page_link
have to be checked through app.py instead: page_link resolves against the
entrypoint's navigation, which does not exist when a page runs on its own.

    python smoke_check.py app.py               # the home page, via navigation
    python smoke_check.py views/controls.py
"""
import os, sys
from streamlit.testing.v1 import AppTest
page = os.path.abspath(sys.argv[1])
at = AppTest.from_file(page, default_timeout=300)
at.run()
if at.exception:
    for e in at.exception:
        print("EXC", sys.argv[1], (e.message or "")[:900])
    sys.exit(1)
print("OK", sys.argv[1], "| markdown", len(at.markdown), "| warnings", len(at.warning))
