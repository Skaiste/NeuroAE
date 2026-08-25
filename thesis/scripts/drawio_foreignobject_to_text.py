#!/usr/bin/env python3
"""Convert draw.io XHTML foreignObject labels into native SVG text labels.

Usage:
    python3 scripts/drawio_foreignobject_to_text.py \
        ../figures/linearAEarchitecture.svg \
        ../figures/linearAEarchitecture_latex.svg

The original SVG is never modified. The conversion targets the label layout
produced by draw.io SVG exports (position stored as CSS margin-left/padding-top).
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SVG = "http://www.w3.org/2000/svg"
XHTML = "http://www.w3.org/1999/xhtml"
ET.register_namespace("", SVG)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def css_number(style: str, property_name: str, default: float = 0) -> float:
    match = re.search(rf"(?:^|;)\s*{re.escape(property_name)}:\s*([-\d.]+)px", style)
    return float(match.group(1)) if match else default


def find_positioned_div(foreign_object: ET.Element) -> ET.Element | None:
    for element in foreign_object.iter():
        style = element.get("style", "")
        if "margin-left:" in style and "padding-top:" in style:
            return element
    return None


def find_text_div(foreign_object: ET.Element) -> ET.Element | None:
    candidates = []
    for element in foreign_object.iter():
        if local_name(element.tag) == "div" and "font-size:" in element.get("style", ""):
            content = "".join(element.itertext()).strip()
            if content:
                candidates.append(element)
    return candidates[-1] if candidates else None


def text_lines(element: ET.Element) -> list[str]:
    lines = [""]

    def append(value: str | None) -> None:
        if value:
            lines[-1] += value

    def walk(node: ET.Element) -> None:
        append(node.text)
        for child in node:
            if local_name(child.tag).lower() == "br":
                lines.append("")
            else:
                walk(child)
            append(child.tail)

    walk(element)
    return [line.strip() for line in lines if line.strip()]


def replacement_label(foreign_object: ET.Element) -> ET.Element | None:
    positioned = find_positioned_div(foreign_object)
    text_div = find_text_div(foreign_object)
    if positioned is None or text_div is None:
        return None

    position_style = positioned.get("style", "")
    text_style = text_div.get("style", "")
    x = css_number(position_style, "margin-left")
    y = css_number(position_style, "padding-top")
    width = css_number(position_style, "width", 1)
    font_size = css_number(text_style, "font-size", 12)
    lines = text_lines(text_div)
    if not lines:
        return None

    group = ET.Element(f"{{{SVG}}}g")
    text = ET.SubElement(
        group,
        f"{{{SVG}}}text",
        {
            "x": f"{x + width / 2:g}",
            "y": f"{y:g}",
            "fill": "rgb(0, 0, 0)",
            "font-family": "Helvetica, Arial, sans-serif",
            "font-size": f"{font_size:g}px",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
        },
    )
    line_height = font_size * 1.2
    first_offset = -line_height * (len(lines) - 1) / 2
    for index, line in enumerate(lines):
        tspan = ET.SubElement(
            text,
            f"{{{SVG}}}tspan",
            {"x": f"{x + width / 2:g}", "dy": f"{first_offset if index == 0 else line_height:g}"},
        )
        tspan.text = line
    return group


def convert(source: Path, destination: Path) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    converted = 0
    skipped = 0

    for element in list(root.iter()):
        if local_name(element.tag) != "foreignObject":
            continue
        replacement = replacement_label(element)
        parent = parents[element]
        index = list(parent).index(element)
        if replacement is None:
            skipped += 1
            continue
        parent.remove(element)
        parent.insert(index, replacement)
        converted += 1

    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    print(f"Converted {converted} labels; skipped {skipped}; wrote {destination}")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} SOURCE.svg DESTINATION.svg")
    raise SystemExit(convert(Path(sys.argv[1]), Path(sys.argv[2])))
