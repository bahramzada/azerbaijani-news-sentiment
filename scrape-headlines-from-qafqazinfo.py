import requests
from bs4 import BeautifulSoup
import time
import re

HEADLINE_SELECTOR = "h4.hemcinin"

REMOVE_WORDS = ["Yenilənib", "foto", "video"]

def clean_headline(text):
    pattern = r'\b(?:' + '|'.join(REMOVE_WORDS) + r')\b'
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned.strip()

all_headlines = set()

for page in range(1, 2):  
    url = f"https://qafqazinfo.az/news/category/xeber-1?page={page}"
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        headlines = [tag.get_text(strip=True) for tag in soup.select(HEADLINE_SELECTOR)]
        # Hər başlığı təmizlə və boş olmayanları əlavə et
        for headline in headlines:
            clean = clean_headline(headline)
            if clean:
                all_headlines.add(clean)
        print(f"{page}-ci səhifə: {len(headlines)} başlıq tapıldı")
        time.sleep(1) 
        if len(all_headlines) >= 1000:
            break
    except Exception as e:
        print(f"{page}-ci səhifədə xəta: {e}")

with open("az_headlines.txt", "w", encoding="utf-8") as f:
    for headline in all_headlines:
        f.write(headline + "\n")

print(f"Toplam {len(all_headlines)} unikal başlıq yazıldı.")
