from pathlib import Path
import py_compile

targets = [
    "app.py",
    "app/dashboard.py",
    "src/data.py",
    "src/segmentation.py",
    "src/churn_model.py",
]

for target in targets:
    py_compile.compile(target, doraise=True)
    print(f"OK: {target}")

print("\nPython syntax validation passed.")
print("Add data/Churn_Modelling.csv and run: streamlit run app.py")
