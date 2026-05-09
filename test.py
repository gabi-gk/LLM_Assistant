import pygetwindow as gw

windows = gw.getAllWindows()
for w in windows:
    print(repr(w.title))