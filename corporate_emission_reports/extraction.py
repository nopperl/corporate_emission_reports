import fitz
import lxml.etree as ET


def text_is_relevant(text):
    text = text.lower().replace(" ", "").replace("one", "1").replace("two", "2").replace("three", "3").replace("scopes", "scope")
    return "scope1" in text or "scope2" in text or "scope3" in text


def xml_to_plaintext(xml: str) -> str:
    root = ET.fromstring(xml)
    return "".join(root.itertext())


def clean_xhtml(xhtml: str, page_number: int) -> str:
    parser = ET.XMLParser(huge_tree=True)  # TODO: implement iterparse instead to reduce memory footprint
    root = ET.fromstring(xhtml, parser=parser)
    ignore_elements = root.findall(".//img")
    for element in ignore_elements:
        element.getparent().remove(element)
    root.attrib["id"] = str(page_number)
    xml_str = ET.tostring(root, encoding="unicode", method="html")
    return xml_str
    

def extract_chunks_from_document(document, max_tokens=65536, characters_per_token=3.5, mode="xhtml"):
    page_counter = 1
    user_message = ""
    with fitz.open(document) as doc:
        for i, page in enumerate(doc):
            if len(user_message) > max_tokens * characters_per_token:
                print("Maximum tokens reached, implement splitting")
                exit() 
            page_text = page.get_text(mode)
            if mode == "xhtml":
                page_text = clean_xhtml(page_text, page_number=i) + "\n"
                is_relevant = text_is_relevant(xml_to_plaintext(page_text))
            elif mode == "text":
                is_relevant = text_is_relevant(page_text)
                page_text = f'<CHUNK id="{i}">\n{page_text}\n</CHUNK>\n'
            else:
                raise ValueError("mode must be xhtml or text")
            if is_relevant:
                user_message += page_text
                page_counter += 1
    return user_message
