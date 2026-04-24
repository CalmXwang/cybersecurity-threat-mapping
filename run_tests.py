from sysml_to_cpe import parse_sysml_xmi
from sysml_cpe import SysMLThreatModelParser
import json

files = [
    "sample1.xml",
    "sample2.xml",
    "sample3.xml",
    "sample4.xml",
    "sample5.xml",
    "sample6.xml",
    "sample7_large_system.xml",
    "sample8_mixed_quality.xml"
]

results = {}

print("===== TESTING sysml_to_cpe.py =====")
for f in files:
    print(f"\n--- {f} ---")
    try:
        result = parse_sysml_xmi(f)
        print(result)
        results[f] = {"sysml_to_cpe": result}
    except Exception as e:
        print("Error:", e)
        results[f] = {"sysml_to_cpe": f"Error: {str(e)}"}

print("\n===== TESTING sysml_cpe.py =====")
for f in files:
    print(f"\n--- {f} ---")
    try:
        parser = SysMLThreatModelParser(f)
        result = parser.get_inventory()
        print(result)
        results[f]["sysml_cpe"] = result
    except Exception as e:
        print("Error:", e)
        results[f]["sysml_cpe"] = f"Error: {str(e)}"

with open("test_results.json", "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=4, ensure_ascii=False)

print("\nSaved to test_results.json")