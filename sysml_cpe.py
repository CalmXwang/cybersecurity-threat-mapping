import xml.etree.ElementTree as ET
import re
from typing import List, Dict

class SysMLThreatModelParser:
    def __init__(self, xml_data: str):
        if xml_data.strip().startswith('<'):
            self.root = ET.fromstring(xml_data)
        else:
            self.root = ET.parse(xml_data).getroot()

    def _get_attr_value(self, block_node, attr_name: str) -> str:
        for attr in block_node.findall(".//ownedAttribute"):
            if attr.get('name') == attr_name:
                return attr.get('default')
        return None

    def construct_cpe_23(self, vendor: str, product: str, version: str) -> str:
        def clean(s):
            if not s or s == "*":
                return "*"
            return re.sub(r'[^a-zA-Z0-9_\\.~]', '_', str(s).strip().lower())

        return f"cpe:2.3:a:{clean(vendor)}:{clean(product)}:{clean(version)}:*:*:*:*:*:*:*"

    def get_inventory(self) -> List[Dict]:
        inventory = []

        for block in self.root.iter():
            if 'Block' in block.get('{http://www.omg.org/spec/XMI/20131001}type', ''):
                block_name = block.get('name')
                xmi_id = block.get('{http://www.omg.org/spec/XMI/20131001}id')

                # 优先使用 cpeHint
                cpe = self._get_attr_value(block, 'cpeHint')

                # fallback
                if not cpe:
                    vendor = self._get_attr_value(block, 'vendor') or "unknown"
                    product = self._get_attr_value(block, 'product') or block_name
                    version = self._get_attr_value(block, 'version') or "*"
                    cpe = self.construct_cpe_23(vendor, product, version)

                inventory.append({
                    "id": xmi_id,
                    "name": block_name,
                    "cpe": cpe
                })

        return inventory


if __name__ == "__main__":
    parser = SysMLThreatModelParser("sysml.xml")
    inventory = parser.get_inventory()
    for item in inventory:
        print(f"Asset: {item['name']} -> CPE: {item['cpe']}")