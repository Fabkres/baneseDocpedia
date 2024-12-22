from api.keywords import extract_keywords
import re
from collections import defaultdict

def generate_mind_map_structure(text, num_keywords=5):
    keywords = extract_keywords(text, num_keywords)
    
    mind_map = {"root": "Main Topic", "branches": []}
    sentences = re.split(r'(?<=\.)\s+', text)
    
    keyword_map = defaultdict(list)
    for keyword in keywords:
        for sentence in sentences:
            if keyword in sentence.lower():
                keyword_map[keyword].append(sentence.strip())
    
    # Constrói os ramos do mapa mental
    for keyword, related_sentences in keyword_map.items():
        mind_map["branches"].append({
            "keyword": keyword,
            "details": related_sentences
        })
    
    return mind_map

def export_mind_map_to_json(mind_map, output_file="mind_map.json"):
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mind_map, f, ensure_ascii=False, indent=4)
