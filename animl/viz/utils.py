import xml.etree.cElementTree as ET

def inline_svg_images(svg) -> str:
    """
    Convert all image tag refs directly under g tags like:

    <g id="node1" class="node">
        <image xlink:href="/tmp/node4.svg" width="45px" height="76px" preserveAspectRatio="xMinYMin meet" x="76" y="-80"/>
    </g>

    to

    <g id="node1" class="node">
        <svg width="49.0px" height="80.8px" preserveAspectRatio="xMinYMin meet" x="76" y="-80">
            XYZ
        </svg>
    </g>

    where XYZ is taken from ref'd svg image file:

    <?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
      "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <!-- Created with matplotlib (http://matplotlib.org/) -->
    <svg height="80.826687pt" version="1.1" viewBox="0 0 49.008672 80.826687" width="49.008672pt"
         xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
        XYZ
    </svg>

    Note that width/height must be taken from ref'd svg image file. The <image/> tag
    values seem a bit off.

    :param svg: SVG string with <image/> tags.
    :return: svg with <image/> tags replaced with content of referenced svg image files.
    """
    ns = {"svg": "http://www.w3.org/2000/svg"}
    root = ET.fromstring(svg)
    tree = ET.ElementTree(root)
    parent_map = {c: p for p in tree.iter() for c in p}

    # Find all image tags in document (must use svg namespace)
    image_tags = tree.findall(".//svg:g/svg:image", ns)
    for img in image_tags:
        img_attrib = {kv[0]:kv[1] for kv in img.attrib.items() if kv[0] not in {"width","height"}}
        # print(img_attrib)
        filename = img.attrib["{http://www.w3.org/1999/xlink}href"]
        with open(filename) as f:
            imgsvg = f.read()
        imgroot = ET.fromstring(imgsvg)
        w = imgroot.attrib['width']
        h = imgroot.attrib['height']
        content = [child for child in imgroot]
        # print(ET.tostring(imgroot).decode())
        print(w,h)
        p = parent_map[img]
        p.remove(img)
    # xml_str = ET.ElementTree.tostring(tree).decode()
    # return xml_str


if __name__ == '__main__':
    with open("/tmp/foo.svg") as f:
        svg = f.read()

    svg2 = inline_svg_images(svg)
    print(svg2)