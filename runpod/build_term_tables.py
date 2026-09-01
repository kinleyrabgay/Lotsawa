"""Extract candidate bilingual term tables from the corpus, with confidence.

Nothing here is invented. Every Dzongkha form is one that already appears in
dataset.csv, scored by how reliably it co-occurs with the English word, so a
reviewer can see the evidence rather than trusting a guess.

Method: for each English word, take every pair whose English side contains it,
count the space-delimited Dzongkha chunks in those pairs, and score each chunk by

    coverage = (carriers containing the chunk) / (carriers)
    lift     = coverage / (chunk's overall frequency in the corpus)

High coverage and high lift together mean the chunk is the term. Grammatical
particles score high coverage but low lift, so lift is what separates them.

Confidence is coverage, banded:
    >=0.80 high     -- almost certainly right, spot-check only
    >=0.50 medium   -- probably right, confirm
    <0.50  low      -- ambiguous, a reviewer must decide (or the corpus lacks it)

    python build_term_tables.py --csv ../dataset.csv --out terms/
"""

import argparse
import collections
import csv
import os
import re

CATEGORIES = {
    "fruits": [
        "apple", "banana", "mango", "orange", "grape", "peach", "pear", "guava",
        "papaya", "lemon", "lime", "pineapple", "watermelon", "muskmelon",
        "plum", "apricot", "cherry", "strawberry", "raspberry", "blackberry",
        "coconut", "date", "fig", "pomegranate", "persimmon", "jackfruit",
        "litchi", "avocado", "kiwi", "sugarcane", "tamarind", "walnut",
        "almond", "cashew", "peanut", "chestnut", "hazelnut", "raisin",
        "mandarin", "grapefruit", "passion fruit", "custard apple", "mulberry",
        "gooseberry", "banana flower",
    ],
    "vegetables": [
        "potato", "onion", "chilli", "cabbage", "carrot", "spinach", "tomato",
        "garlic", "ginger", "radish", "turnip", "pumpkin", "gourd",
        "bitter gourd", "bottle gourd", "beans", "peas", "mushroom",
        "cauliflower", "broccoli", "cucumber", "eggplant", "brinjal", "lettuce",
        "asparagus", "beetroot", "celery", "leek", "shallot", "sweet potato",
        "yam", "taro", "bamboo shoot", "fern", "nettle", "mustard leaf",
        "coriander", "mint", "parsley", "fenugreek", "chayote", "okra",
        "capsicum", "corn", "maize", "squash", "turmeric",
    ],
    "animals": [
        "cow", "bull", "ox", "buffalo", "yak", "dog", "puppy", "cat", "kitten",
        "horse", "pony", "mule", "donkey", "goat", "sheep", "lamb", "pig",
        "chicken", "hen", "rooster", "duck", "goose", "turkey", "rabbit",
        "tiger", "leopard", "snow leopard", "bear", "monkey", "langur",
        "elephant", "rhinoceros", "deer", "musk deer", "takin", "blue sheep",
        "wolf", "fox", "jackal", "otter", "squirrel", "mouse", "rat", "bat",
        "snake", "lizard", "frog", "turtle", "crocodile", "fish", "trout",
        "bird", "crow", "eagle", "vulture", "owl", "sparrow", "pigeon", "dove",
        "crane", "black-necked crane", "raven", "peacock", "parrot", "cuckoo",
        "butterfly", "bee", "ant", "spider", "mosquito", "fly", "worm",
        "grasshopper", "cricket", "beetle", "snail", "leech",
    ],
    "trees": [
        "tree", "pine", "blue pine", "chir pine", "oak", "cypress", "fir",
        "hemlock", "spruce", "juniper", "rhododendron", "magnolia", "maple",
        "birch", "willow", "poplar", "alder", "walnut tree", "apple tree",
        "mango tree", "peach tree", "orange tree", "banana tree", "bamboo",
        "cane", "teak", "sal", "sandalwood", "banyan", "peepal", "fig tree",
        "cedar", "cherry tree", "chestnut tree", "eucalyptus", "acacia",
        "mulberry tree", "bush", "shrub", "grass", "flower", "leaf", "root",
        "branch", "trunk", "bark", "seed", "fruit",
    ],
    "mountains": [
        "mountain", "hill", "peak", "pass", "cliff", "valley", "glacier",
        "himalaya", "gangkhar puensum", "jomolhari", "chomolhari",
        "jichu drake", "kula kangri", "masanggang", "tsenda kang", "teri kang",
        "table mountain", "kangphu gang", "everest", "kanchenjunga",
        "dochula", "chelela", "pelela", "yotongla", "thrumshingla",
    ],
    "names": [
        "tom", "mary", "john", "alice", "david", "mike", "tony", "ken", "jane",
        "bob", "peter", "sarah", "anna", "james", "robert", "michael",
        "tashi", "dorji", "pema", "sonam", "ugyen", "kinley", "jigme", "karma",
        "chimi", "tshering", "dechen", "sangay", "namgay", "wangchuk",
        "choden", "yangchen", "deki", "kezang", "rinzin", "tenzin", "phuntsho",
        "nima", "gyeltshen", "lhamo", "zangmo", "wangmo", "dema", "kelzang",
        "thinley", "norbu", "yeshey", "chencho", "kuenzang", "jamyang",
        "lobzang", "passang", "sherab", "tandin", "yonten", "damcho", "tobgay",
        "gyem", "sithar", "leki", "pelden", "rigzin", "singye", "sangye",
    ],
    "kinship": [
        "mother", "father", "parents", "brother", "sister", "elder brother",
        "elder sister", "younger brother", "younger sister", "uncle", "aunt",
        "son", "daughter", "child", "children", "baby", "grandmother",
        "grandfather", "grandson", "granddaughter", "wife", "husband", "spouse",
        "cousin", "nephew", "niece", "in-law", "father-in-law",
        "mother-in-law", "family", "relative", "friend", "neighbour", "guest",
        "boy", "girl", "man", "woman", "elder", "twin", "orphan", "widow",
    ],
    "occupation": [
        "teacher", "doctor", "nurse", "farmer", "student", "driver", "monk",
        "nun", "police", "soldier", "shopkeeper", "engineer", "carpenter",
        "blacksmith", "weaver", "tailor", "painter", "cook", "baker", "butcher",
        "barber", "mason", "plumber", "electrician", "mechanic", "guide",
        "porter", "shepherd", "herder", "fisherman", "hunter", "merchant",
        "trader", "clerk", "officer", "minister", "judge", "lawyer",
        "accountant", "banker", "journalist", "writer", "singer", "dancer",
        "musician", "artist", "astrologer", "healer", "pilot", "postman",
        "gardener", "cleaner", "guard", "secretary", "manager", "principal",
    ],
    "food": [
        "rice", "red rice", "bread", "noodle", "flour", "milk", "butter",
        "cheese", "curd", "yoghurt", "tea", "butter tea", "water", "juice",
        "meat", "beef", "pork", "chicken meat", "mutton", "fish meat", "egg",
        "salt", "sugar", "oil", "ghee", "honey", "soup", "stew", "curry",
        "chilli cheese", "porridge", "dumpling", "momo", "pancake", "biscuit",
        "cake", "sweet", "pickle", "vinegar", "pepper", "spice", "yeast",
        "beer", "wine", "liquor", "breakfast", "lunch", "dinner", "snack",
        "meal", "food",
    ],
    "colours": [
        "red", "blue", "green", "yellow", "white", "black", "brown", "grey",
        "pink", "purple", "orange colour", "gold", "silver", "maroon",
        "violet", "turquoise", "dark", "light", "bright", "pale", "colour",
    ],
    "time": [
        "today", "tomorrow", "yesterday", "day after tomorrow",
        "day before yesterday", "morning", "afternoon", "evening", "night",
        "midnight", "noon", "dawn", "dusk", "week", "month", "year", "hour",
        "minute", "second", "day", "date", "time", "season", "spring",
        "summer", "autumn", "winter", "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday", "january", "february",
        "march", "april", "may", "june", "july", "august", "september",
        "october", "november", "december", "now", "later", "early", "late",
        "always", "never", "sometimes", "often", "daily", "weekly", "monthly",
        "yearly", "past", "present", "future",
    ],
    "places": [
        "thimphu", "paro", "punakha", "wangdue", "bumthang", "trongsa",
        "trashigang", "trashiyangtse", "mongar", "lhuentse", "pemagatshel",
        "samdrup jongkhar", "samtse", "chukha", "haa", "gasa", "dagana",
        "tsirang", "sarpang", "zhemgang", "gelephu", "phuentsholing",
        "bhutan", "india", "nepal", "china", "tibet", "bangladesh", "japan",
        "korea", "thailand", "singapore", "america", "england", "australia",
        "canada", "germany", "france", "delhi", "kolkata", "kathmandu",
        "country", "city", "town", "village", "district", "capital",
    ],
    "places_local": [
        "school", "college", "university", "hospital", "clinic", "market",
        "shop", "office", "bank", "post office", "temple", "monastery",
        "dzong", "stupa", "chorten", "prayer wheel", "house", "home", "room",
        "kitchen", "toilet", "garden", "farm", "field", "road", "path",
        "bridge", "airport", "bus station", "river", "lake", "stream",
        "waterfall", "spring", "forest", "meadow", "pasture", "border",
        "hotel", "restaurant", "library", "museum", "stadium", "police station",
        "court", "prison", "factory", "workshop", "warehouse", "toll gate",
    ],
    "body": [
        "body", "head", "hair", "face", "forehead", "eye", "eyebrow", "ear",
        "nose", "mouth", "lip", "tooth", "tongue", "chin", "cheek", "neck",
        "throat", "shoulder", "arm", "elbow", "hand", "finger", "thumb",
        "nail", "chest", "breast", "back", "waist", "stomach", "belly", "hip",
        "leg", "knee", "foot", "toe", "heel", "skin", "bone", "blood",
        "heart", "lung", "liver", "kidney", "brain", "nerve", "muscle", "vein",
    ],
}

BAND = [(0.80, "high"), (0.50, "medium"), (0.0, "low")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../dataset.csv")
    ap.add_argument("--out", default="terms")
    ap.add_argument("--min-carriers", type=int, default=4)
    ap.add_argument("--top", type=int, default=3, help="candidates to report")
    args = ap.parse_args()

    csv.field_size_limit(10 ** 9)
    with open(args.csv, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        rows = [(a.strip(), b.strip()) for a, b in reader]
    print(f"Corpus: {len(rows)} pairs")

    overall = collections.Counter()
    for dz, _ in rows:
        overall.update(set(dz.split()))
    n = len(rows)

    os.makedirs(args.out, exist_ok=True)
    summary = collections.Counter()

    for category, words in CATEGORIES.items():
        out_rows = []
        for word in words:
            pat = re.compile(rf"\b{word}s?\b", re.I)
            carriers = [dz for dz, en in rows if pat.search(en)]
            if len(carriers) < args.min_carriers:
                out_rows.append({
                    "en": word, "dz_preferred": "", "confidence": "MISSING",
                    "coverage": "", "carriers": len(carriers),
                    "dz_candidates": "", "review": "YES - not in corpus",
                })
                summary["missing"] += 1
                continue

            local = collections.Counter()
            for dz in carriers:
                local.update(set(dz.split()))

            scored = []
            for chunk, c in local.items():
                cov = c / len(carriers)
                if cov < 0.15:
                    continue
                lift = cov / max(overall[chunk] / n, 1e-9)
                scored.append((lift, cov, chunk))
            scored.sort(reverse=True)
            if not scored:
                out_rows.append({
                    "en": word, "dz_preferred": "", "confidence": "MISSING",
                    "coverage": "", "carriers": len(carriers),
                    "dz_candidates": "", "review": "YES - no clear candidate",
                })
                summary["missing"] += 1
                continue

            lift, cov, best = scored[0]
            band = next(b for t, b in BAND if cov >= t)
            summary[band] += 1
            out_rows.append({
                "en": word,
                "dz_preferred": best,
                "confidence": band,
                "coverage": f"{cov:.2f}",
                "carriers": len(carriers),
                "dz_candidates": " | ".join(
                    f"{c}({cv:.2f})" for _, cv, c in scored[:args.top]),
                "review": "" if band == "high" else "YES",
            })

        path = os.path.join(args.out, f"{category}.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "en", "dz_preferred", "confidence", "coverage", "carriers",
                "dz_candidates", "review", "dz_variants", "notes"])
            w.writeheader()
            for r in out_rows:
                r.setdefault("dz_variants", "")
                r.setdefault("notes", "")
                w.writerow(r)
        hi = sum(1 for r in out_rows if r["confidence"] == "high")
        print(f"  {category:15s} {len(out_rows):3d} terms, {hi:3d} high-confidence"
              f"  -> {path}")

    print(f"\nTotals: {summary['high']} high, {summary['medium']} medium, "
          f"{summary['low']} low, {summary['missing']} missing from corpus")
    print("\nHigh-confidence rows still deserve a spot-check: the corpus stores "
          "some terms as\ntransliterations of the English word rather than the "
          "native Dzongkha term.\nEverything else needs a reviewer or the DDC "
          "dictionary.")


if __name__ == "__main__":
    main()
