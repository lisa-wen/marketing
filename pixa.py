import os, csv, json, argparse, pathlib, requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Konfiguriere API-Schlüssel und LLM-Modell
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "sk-REPLACE_WITH_YOUR_KEY"
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY") or ""
MODEL = "gpt-4o-mini"

# Erlaubte Kategorien und Bildtypen bei Pixabay
ALLOWED_PIXABAY_CATEGORIES = {
    "llm", "all", "backgrounds", "fashion", "nature", "science", "education", "feelings", "health", "people",
    "religion", "places", "animals", "industry", "computer", "food", "sports", "transportation",
    "travel", "buildings", "business", "music",
}
ALLOWED_IMAGE_TYPES = {"all", "photo", "illustration", "vector"}

# ====================================================
# Prompt-Parameter
# ====================================================

# Systemnachricht für die Social-Post-Generierung
SYSTEM = (
    "Du schreibst Social-Posts für die Hochschule Merseburg. Deine Posts sind seriös, gehen aber "
    "häufig viral. Antworte sachlich und ansprechend."
)

# Arbeitsaufforderung für die Social-Post-Generierung für jedes Titel-/Text-Paar
ASK = (
    "Erzeuge JSON: {\"linkedin\": \"...\", \"instagram\": \"...\"}. "
    "LinkedIn: professionell, 600–900 Zeichen, 5–8 Hashtags. "
    "Instagram: freundlich mit passenden Emojis, 150–300 Wörter, 6–10 Hashtags am Ende."
)

# Systemnachricht für die Pixabay-Kategorienzuordnung
CATEGORY_SYSTEM = (
    "Ordne eine News-Meldung einer einzigen Pixabay-Kategorie zu. Antworte NUR mit einem Wort aus der Liste."
)

# Arbeitsaufforderung für die Pixabay-Kategorienzuordnung anhand des News-Artikels
CATEGORY_USER_TEMPLATE = (
    "Text: \n{body}\n\nErlaubte Kategorien: backgrounds, fashion, nature, science, education, feelings, health, people, "
    "religion, places, animals, industry, computer, food, sports, transportation, travel, buildings, business, music\n" 
    "Antwort:"
)

# Systemnachricht für die Erzeugung von Pixabay-Suchparametern
QUERY_SYSTEM = (
    "Erzeuge ausschließlich prägnante englische Suchbegriffe für die Bildsuche (Pixabay). "
    "Gib *2-3 kurze und einfache Begriffe*"
)

# Arbeitsaufforderung für die Erzeugung von Pixabay-Suchparametern
QUERY_USER_TEMPLATE = (
    "News-Titel: {title}\nNews-Text: {body}\n\nGib *2-3 kurze und einfache Suchbegriffe* (Deutsch):"
)

# ====================================================
# Funktionen für die Textgenerierung
# ====================================================


# Erzeuge Pixabay-Kategorie
def llm_category_from_text(client: OpenAI, text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CATEGORY_SYSTEM},
                {"role": "user", "content": CATEGORY_USER_TEMPLATE.format(body=text[:4000])},
            ],
            temperature=0.2,
        )
        choice = (resp.choices[0].message.content or "").strip().lower()
        if not choice:
            return ""
        choice = choice.split()[0]
        return choice if choice in ALLOWED_PIXABAY_CATEGORIES else ""
    except Exception:
        return ""


# Erzeuge Pixabay-Suchwörter
def llm_query_terms(client: OpenAI, title: str, text: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM},
                {"role": "user", "content": QUERY_USER_TEMPLATE.format(title=title[:200], body=text[:1200])},
            ],
            temperature=0.2,
        )
        terms = (resp.choices[0].message.content or "").strip()
        terms = ", ".join([t.strip() for t in terms.replace("\n", ",").split(",") if t.strip()])
        return terms[:120] if terms else title
    except Exception:
        return title


# Erzeuge Social Posts
def generate_posts(rows: pd.DataFrame, category_mode: str) -> pd.DataFrame:
    if not API_KEY or API_KEY.startswith("sk-REPLACE"):
        raise SystemExit("Please set OPENAI_API_KEY (or API_KEY) or edit API_KEY in the script.")
    client = OpenAI(api_key=API_KEY)

    out = []
    for r in rows.itertuples(index=False):
        title, date, text = str(r.Titel), str(r.Datum), str(r.Text)
        prompt = f"Titel: {title}\nDatum: {date}\n{text}\n\n{ASK}"
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            data = json.loads(raw)
        except Exception:
            data = {"linkedin": raw, "instagram": ""}

        if category_mode == "llm":
            category = llm_category_from_text(client, text)
        else:
            category = category_mode  # can be "all" or a fixed category

        terms = llm_query_terms(client, title, text)
        out.append({
            "id": str(r.ID),
            "title": title,
            "date": date,
            "linkedin": data.get("linkedin", "").strip(),
            "instagram": data.get("instagram", "").strip(),
            "category": category,
            "query_terms": terms,
        })
    return pd.DataFrame(out)


# Stelle Anfrage an Pixabay
def pixabay_search_images(query: str, category: str, image_type: str, lang: str = "de", per_page: int = 3) -> list:
    if not PIXABAY_API_KEY:
        return []
    if image_type not in ALLOWED_IMAGE_TYPES:
        image_type = "all"
    params = {
        "key": PIXABAY_API_KEY,
        "q": query,
        "lang": lang,
        "per_page": per_page,
        "safesearch": "true",
        "orientation": "horizontal",
        "image_type": image_type,
        "editors_choice": "true"
    }
    if category in ALLOWED_PIXABAY_CATEGORIES:
        params["category"] = category
    try:
        r = requests.get("https://pixabay.com/api/", params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        return js.get("hits") or []
    except Exception:
        return []


# Lade Pixabay-Bilder anhand der URL herunter
def download_image(url: str, dest_path: pathlib.Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


# Starte Bildersuche und Bilddownload
def search_and_download_images(df: pd.DataFrame, images_out: str, image_type: str, num_images: int = 10, lang: str = "de") -> pd.DataFrame:
    num_images = max(1, min(int(num_images), 10))
    images_dir = pathlib.Path(images_out)
    images_dir.mkdir(parents=True, exist_ok=True)

    cols_urls = {i: [] for i in range(1, num_images + 1)}
    cols_paths = {i: [] for i in range(1, num_images + 1)}
    cols_creds = {i: [] for i in range(1, num_images + 1)}

    for idx, row in enumerate(df.itertuples(index=False)):
        q = getattr(row, "query_terms", None) or getattr(row, "instagram", None) or getattr(row, "title", "")
        rid = getattr(row, "id", None) or getattr(row, "ID", None) or f"row{idx}"
        cat = getattr(row, "category", None) or None

        hits = []
        if q:
            hits = pixabay_search_images(q, cat if cat else None, image_type=image_type, lang=lang, per_page=num_images) or []

        for i in range(1, num_images + 1):
            hit = hits[i - 1] if i - 1 < len(hits) else {}
            url = hit.get("largeImageURL") or hit.get("webformatURL") or ""
            page = hit.get("pageURL", "") or ""
            user = hit.get("user", "") or ""
            tags = hit.get("tags", "") or ""
            credit = f"Pixabay: {user} — {page} — {tags}" if url else ""
            fpath_str = ""
            if url:
                fpath = images_dir / f"{rid}_{i}.jpg"
                if download_image(url, fpath):
                    fpath_str = str(fpath)
            cols_urls[i].append(url)
            cols_paths[i].append(fpath_str)
            cols_creds[i].append(credit)

    out = df.copy()
    for i in range(1, num_images + 1):
        out[f"image{i}_url"] = cols_urls[i]
        out[f"image{i}_path"] = cols_paths[i]
        out[f"image{i}_credit"] = cols_creds[i]
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate social posts + Pixabay image search")
    parser.add_argument("--in", dest="in_path", required=True, help="Input CSV with ID;Titel;Datum;Text (semicolon-separated)")
    parser.add_argument("--out", dest="out_path", default="social_posts.csv", help="Output CSV path")
    parser.add_argument("--search-images", action="store_true", help="Search images on Pixabay and download")
    parser.add_argument("--images-out", default="images", help="Directory to save images")
    parser.add_argument("--image-type", default="", choices=sorted(ALLOWED_IMAGE_TYPES), help="Pixabay image_type filter")
    parser.add_argument("--category", default="all", choices=sorted(ALLOWED_PIXABAY_CATEGORIES), help="Pixabay category filter: 'all' (no filter), 'llm' (let LLM decide), or fixed category")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    df = pd.read_csv(args.in_path, sep=";")
    required = {"ID", "Titel", "Datum", "Text"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in input CSV: {sorted(missing)}")

    result = generate_posts(df[["ID", "Titel", "Datum", "Text"]], args.category)

    if args.search_images:
        if not PIXABAY_API_KEY:
            print("[warn] PIXABAY_API_KEY not set — skipping image search.")
        else:
            result = search_and_download_images(result, args.images_out, args.image_type)

    result.to_csv(args.out_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(result)} rows to {args.out_path}")
    if args.search_images:
        print(f"Images (if found) saved to {args.images_out}")


if __name__ == "__main__":
    main()
