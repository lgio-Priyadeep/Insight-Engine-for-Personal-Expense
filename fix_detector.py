import re
path = "recurring_detector.py"
with open(path) as f: text = f.read()
text = text.replace("if T < 0.3 or A < 0.2:", "if T == 0.0 or A == 0.0:")
with open(path, "w") as f: f.write(text)
