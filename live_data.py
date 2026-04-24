import os
import datetime
import requests
import dload  
import zipfile
import csv
import json
import io
import re
import requests
import pandas as pd


def log_download(resource_dir, dataset_name, entry_count, notes=None, version_number=None):
    log_path = os.path.join(resource_dir, "download_log.txt")
    with open(log_path, "w", encoding="utf-8") as log_file:  # <-- "w" instead of "a"
        log_file.write(f"{datetime.datetime.now().isoformat()} | {dataset_name} | Entries: {entry_count} | Version #: {version_number} | Notes: {notes}\n")

def download_all_nvd_data(start_year=2002): # **hardcoded value** '2002'
    current_year = datetime.datetime.now().year
    file_paths_nvdjson_recent = ['modified']

    # Check all years from start_year up to current year
    for year in range(start_year, current_year + 1):
        url = f"https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{year}.json.zip"
        response = requests.head(url)
        if response.status_code == 200:
            file_paths_nvdjson_recent.append(str(year))
        else:
            print(f"Skipping year {year} (no feed found)")

    # Remove old JSON files if they exist
    for name in file_paths_nvdjson_recent:
        try:
            os.remove(f'./OSRs/NVD/nvdcve-2.0-{name}.json')
        except OSError:
            pass  # Ignore if file doesn't exist

    # Download and unzip the data
    nvd_dir = './OSRs/NVD'
    total_entries = 0
    for name in file_paths_nvdjson_recent:
        print(f"Downloading {name}...")
        dload.save_unzip(
            f"https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{name}.json.zip",
            nvd_dir,
            True
        )
        # After unzipping, count entries in the JSON file
        json_path = os.path.join(nvd_dir, f"nvdcve-2.0-{name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entries = len(data.get("vulnerabilities", []))
                    total_entries += entries
            except Exception:
                pass
    # log_download(resource_dir, dataset_name, entry_count, notes=None, version_number=None)
    log_download(nvd_dir, "NVD", total_entries, version_number=", ".join(file_paths_nvdjson_recent))
    print(f"Downloaded NVD data for: {', '.join(file_paths_nvdjson_recent)}")


def download_latest_attack_json():
    url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
    output_dir = "./OSRs/ATTACK"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "enterprise-attack.json")

    print("Downloading latest MITRE ATT&CK Enterprise JSON...")
    try:
        response = requests.get(url)
        response.raise_for_status() 
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        # Count entries in the JSON
        try:
            data = json.loads(response.text)
            entry_count = len(data.get("objects", []))
        except Exception:
            entry_count = 0
        log_download(output_dir, "MITRE ATT&CK Enterprise", entry_count, "Latest version as of download", version_number="Latest")
        print(f"Saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the file: {e}")

def download_attack_v16_1_json():
    url = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/refs/heads/master/enterprise-attack/enterprise-attack-16.1.json"
    output_dir = "./OSRs/ATTACK"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "enterprise-attack-v16.1.json")

    print("Downloading MITRE ATT&CK Enterprise v16.1 JSON...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        # Count entries in the JSON
        try:
            data = json.loads(response.text)
            entry_count = len(data.get("objects", []))
        except Exception:
            entry_count = 0
        log_download(output_dir, "MITRE ATT&CK Enterprise v16.1", entry_count, "Dowloaded v16.1 instead of latest version (17.1) to insure compatibility ewith " \
        "NIST 800-53 Layer Navigator", version_number="16.1")
        print(f"Saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the v16.1 JSON file: {e}")

def download_nist_csf_mapping():
    url = "https://csrc.nist.gov/files/pubs/sp/800/53/r5/upd1/final/docs/csf-pf-to-sp800-53r5-mappings.xlsx"
    output_dir = "./OSRs/NIST-CSF-to-800-53"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "csf-pf-to-sp800-53r5-mappings.xlsx")

    print("Downloading NIST CSF → SP800-53 R5 mapping Excel...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        # Excel file, can't count entries easily
        log_download(output_dir, "NIST CSF to SP800-53 R5 Mapping Excel", "N/A")
        print(f"Saved Excel to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the Excel file: {e}")

def download_nist_800_53_layer_navigator():
    url = "https://center-for-threat-informed-defense.github.io/mappings-explorer/data/nist_800_53/attack-16.1/nist_800_53-rev5/enterprise/nist_800_53-rev5_attack-16.1-enterprise_json.json"
    output_dir = "./OSRs/NIST800-53-layer-navigator"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "nist_800_53-rev5_attack-16.1-enterprise_json.json")

    print("Downloading NIST 800-53 Layer Navigator JSON...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        # Count entries in the JSON
        try:
            data = json.loads(response.content.decode('utf-8'))
            entry_count = len(data.get("mapping_objects", []))
        except Exception:
            entry_count = 0
        log_download(output_dir, "NIST 800-53 Layer Navigator", entry_count, "NIST 800-53 Rev5 to MITRE ATT&CK v16.1 mapping", version_number="Rev5 to ATT&CK v16.1")
        print(f"Saved JSON to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the JSON file: {e}")


# https://csrc.nist.gov/projects/risk-management/sp800-53-controls/downloads
def download_nist_800_53():
    """
    Downloads the NIST SP 800-53 Rev5 (OSCAL-derived) Excel from NIST,
    reads the controls from the correct worksheet (sheet index 1),
    converts to JSON, and stores the result in ./OSRs/NIST800_53.
    """

    url = "https://csrc.nist.gov/CSRC/media/Projects/risk-management/800-53%20Downloads/800-53r5/NIST_SP-800-53_rev5-derived-OSCAL.xlsx"

    output_dir = "./OSRs/NIST800_53"
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, "NIST_SP-800-53_rev5-derived-OSCAL.xlsx")
    output_path = os.path.join(output_dir, "nist_800_53.json")

    print("Downloading NIST SP 800-53 Rev5 OSCAL Excel...")

    # Download the Excel file from NIST
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(excel_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded Excel to: {excel_path}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download Excel: {e}")
        return

    # Load the SECOND worksheet (index 1)
    try:
        excel_stream = io.BytesIO(response.content)
        df = pd.read_excel(excel_stream, sheet_name=1, engine="openpyxl")

    except Exception as e:
        print(f"[ERROR] Could not read second worksheet (index=1): {e}")
        return

    # Expected column names (UPDATED)
    expected_cols = [
        "Control Identifier",
        "Control Name",
        "Control",
        "Discussion",
        "Related Controls"
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns in worksheet: {missing}")
        print("Columns present:", list(df.columns))
        return

    # Normalize to internal keys
    df = df[expected_cols].copy()
    df.rename(columns={
        "Control Identifier": "control_id",
        "Control Name": "name",
        "Control": "statement",         # <-- renamed logically
        "Discussion": "discussion",
        "Related Controls": "related_controls"
    }, inplace=True)

    # Related controls: split by comma/semicolon
    def split_related(val):
        if pd.isna(val):
            return []
        parts = re.split(r"[;,]", str(val))
        return [p.strip() for p in parts if p.strip()]

    df["related_controls"] = df["related_controls"].apply(split_related)

    # Build JSON OSR objects
    records = []
    for _, row in df.iterrows():
        records.append({
            "type": "CONTROL",
            "control_id":       str(row["control_id"]).strip(),
            "name":             str(row["name"]).strip(),
            "statement":        str(row["statement"]).strip() if not pd.isna(row["statement"]) else "",
            "discussion":       str(row["discussion"]).strip() if not pd.isna(row["discussion"]) else "",
            "related_controls": row["related_controls"],
        })

    # Save JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print(f"Saved JSON to {output_path}")

    except Exception as e:
        print(f"[ERROR] Failed to write JSON: {e}")
        return

    # Log in the same style as your other downloaders
    try:
        log_download(
            output_dir,
            "NIST 800-53 OSCAL Controls",
            len(records),
            "Downloaded OSCAL-derived Excel and converted to JSON",
            version_number="Rev5 (OSCAL-derived)"
        )
    except NameError:
        pass

    print(f"Total controls processed: {len(records)}")
    print("Done.")



download_latest_attack_json()
download_attack_v16_1_json()
download_all_nvd_data()
download_nist_800_53_layer_navigator()
download_nist_csf_mapping()
download_nist_800_53()