import xml.etree.ElementTree as ET
import json
import logging
import re

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PHASE 1 - %(levelname)s - %(message)s')

def strip_namespace(tag):
    """Removes the {url} namespace from XML tags for robust version handling."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def parse_sysml_xmi(file_path):
    """
    Parses XMI 2.1/2.5 export to find Blocks and their defined attributes.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logging.error(f"XML Parsing Error: {e}")
        return []

    assets = []

    # Traverse every element in the tree (depth-first)
    for elem in root.iter():
        tag = strip_namespace(elem.tag)

        # Look for SysML Blocks or UML Classes acting as components
        xmi_type = elem.get(f"{{{root.nsmap.get('xmi', '')}}}type") if hasattr(root, 'nsmap') else elem.get('{http://www.omg.org/spec/XMI/20131001}type')

        is_block = (tag == 'packagedElement' and 'Block' in (elem.get('type') or '')) or \
                   (tag == 'Block') or \
                   (elem.get('xmi:type') == 'sysml:Block')

        if not is_block:
            for k, v in elem.attrib.items():
                if 'type' in k and 'Block' in v:
                    is_block = True
                    break

        if is_block:
            asset = {
                'id': elem.get('id') or elem.get('{http://www.omg.org/spec/XMI/20131001}id'),
                'name': elem.get('name', 'Unknown'),
                'vendor': None,
                'product': None,
                'version': None,
                'cpe': None
            }

            for child in elem:
                child_tag = strip_namespace(child.tag)
                if child_tag == 'ownedAttribute':
                    attr_name = child.get('name', '').lower()
                    attr_val = child.get('default')

                    if not attr_val:
                        continue

                    if 'vendor' in attr_name:
                        asset['vendor'] = attr_val
                    elif 'product' in attr_name:
                        asset['product'] = attr_val
                    elif 'version' in attr_name:
                        asset['version'] = attr_val
                    elif 'cpe' in attr_name:
                        asset['cpe'] = attr_val

            if not asset['cpe'] and asset['vendor'] and asset['product']:
                ver = asset['version'] if asset['version'] else '*'
                asset['cpe'] = f"cpe:2.3:a:{asset['vendor']}:{asset['product']}:{ver}:*:*:*:*:*:*:*"
                logging.info(f"Generated CPE for {asset['name']}: {asset['cpe']}")

            if asset['vendor'] or asset['cpe']:
                assets.append(asset)

    logging.info(f"Extracted {len(assets)} assets.")
    return assets

if __name__ == "__main__":
    assets = parse_sysml_xmi("sysml.xml")

    with open("01_assets.json", "w") as f:
        json.dump(assets, f, indent=4)