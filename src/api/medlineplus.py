import requests
from xml.etree import ElementTree as ET
import re
import html
import logging
from typing import List, Dict, Optional

# URL of the MedlinePlus search API
base_url = "https://wsearch.nlm.nih.gov/ws/query"


def clean_html(text: str) -> str:
    """Cleaning HTML tags and decoding HTML entities"""
    if not text:
        return text

    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)

    return text.strip()


def search_medline(term: str, max_results: int = 5) -> List[Dict]:
    """Search for medical topics in MedlinePlus"""
    if not term or not term.strip():
        logging.warning("Empty search term")
        return []
    
    params = {
        "db": "healthTopics",
        "term": term.strip(),
        "retmax": max_results
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        results = []
        for doc in root.findall(".//document"):
            title_el = doc.find(".//content[@name='title']")
            url_attr = doc.get('url')

            # Looking for alternative fields for description
            summary_el = doc.find(".//content[@name='FullSummary']")
            if not summary_el:
                summary_el = doc.find(".//content[@name='summary']")
            if not summary_el:
                summary_el = doc.find(".//content[@name='description']")
            if not summary_el:
                summary_el = doc.find(".//content[@name='abstract']")
            if not summary_el:
                summary_el = doc.find(".//content[@name='content']")

            # If no summary-like field found, try snippet as a fallback
            snippet_el = None
            if not summary_el:
                snippet_el = doc.find(".//content[@name='snippet']")

            # We obtain the cleaned values
            title = clean_html(title_el.text) if title_el is not None and title_el.text else None
            url = url_attr if url_attr else None

            # Determine summary with proper fallbacks
            summary: str
            if summary_el is not None and summary_el.text:
                summary = clean_html(summary_el.text)
            elif snippet_el is not None and snippet_el.text:
                summary = clean_html(snippet_el.text)
            else:
                summary = "Опис недоступний"

            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "source": "MedlinePlus"
                })

        return results

    except requests.RequestException as e:
        logging.error(f"Network request error: {e}")
        return []
    except ET.ParseError as e:
        logging.error(f"XML parsing error: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return []


