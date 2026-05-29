import requests
import random
import json
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

def get_random_scifact_claims(n=10):
    print(f"⏳ Scaricando {n} claim da SciFact (tramite MTEB)...")
    url = "https://huggingface.co/datasets/mteb/scifact/resolve/main/queries.jsonl"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        claims = [json.loads(line) for line in response.text.strip().split("\n") if line.strip()]
        return [c.get('text', c.get('claim', '')) for c in random.sample(claims, min(n, len(claims)))]
    except Exception as e:
        print(f"❌ Errore API Claim: {e}")
        return [f"Offline generated clinical claim #{i} for testing purposes." for i in range(1, n+1)]

def generate_social_mockups(output_dir):
    """Genera finti screenshot social in INGLESE per test OCR."""
    scenari = [
        ("social_1_tweet_diabetes.png", (21, 32, 43), (255, 255, 255), "Tweet by @HealthGuru:\n\nIntermittent fasting completely cures\ntype 2 diabetes in just 4 weeks.\n\nStop taking insulin today!"),
        ("social_2_facebook_covid.png", (240, 242, 245), (28, 30, 33), "Group Post 'Informed Moms':\n\nHigh doses of Vitamin C completely\nprevent and cure Covid-19 infection.\n\nDon't trust Big Pharma drugs!"),
        ("social_3_infographic_vaccines.png", (180, 40, 40), (255, 255, 255), "!!! VACCINE ALERT !!!\n\nmRNA vaccines permanently and\nirreversibly alter human DNA.\n\nThey cause fulminant myocarditis in 100%\nof healthy professional athletes."),
        ("social_4_poster_aspirin.png", (220, 240, 220), (10, 60, 20), "--- PHARMACOLOGY NOTES ---\n\nAcetylsalicylic acid (Aspirin)\ninhibits prostaglandin synthesis by\nblocking the cyclooxygenase (COX) enzyme.\n\nThis process reduces fever and inflammation."),
        ("social_5_article_asthma.png", (255, 255, 255), (0, 0, 0), "TITLE: Natural Secrets\n\nStudies show that taking 10mg of\nmelatonin nightly provides a permanent\ncure for chronic pediatric asthma.")
    ]
    
    generated_paths = []
    
    for filename, bg_color, text_color, text in scenari:
        img = Image.new('RGB', (900, 450), color=bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 35)
        except IOError:
            font = ImageFont.load_default()
                
        draw.text((50, 60), text, fill=text_color, font=font, spacing=15)
        path = os.path.join(output_dir, filename)
        img.save(path)
        generated_paths.append(os.path.abspath(path))
        
    return generated_paths

def generate_hybrid_mockup(claim_text, index, output_dir):
    """Genera un finto grafico di un paper scientifico coerente col claim testuale."""
    img = Image.new('RGB', (800, 500), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_text = ImageFont.truetype("arial.ttf", 18)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        
    # Titolo derivato dal claim (prime 6 parole)
    words = claim_text.split()
    title = "Research Fig: " + " ".join(words[:6]) + "..."
    
    draw.text((40, 30), title, fill=(0, 0, 0), font=font_title)
    draw.text((40, 80), "Figure 1. Observational data summary.", fill=(100, 100, 100), font=font_text)
    
    # Disegna un finto grafico a barre
    colors = [(74, 144, 226), (223, 104, 104), (80, 227, 194)]
    for i in range(3):
        x0 = 150 + i*180
        y0 = random.randint(150, 320)
        x1 = 250 + i*180
        y1 = 380
        draw.rectangle([x0, y0, x1, y1], fill=colors[i])
        draw.text((x0 + 20, y1 + 10), f"Cohort {chr(65+i)}", fill=(0,0,0), font=font_text)
        
    # Inserisce il claim come didascalia (Caption) in basso
    wrapped_claim = textwrap.fill(f"Caption: {claim_text}", width=85)
    draw.text((40, 420), wrapped_claim, fill=(50, 50, 50), font=font_text)
    
    filename = f"hybrid_figure_{index}.png"
    path = os.path.join(output_dir, filename)
    img.save(path)
    return os.path.abspath(path)

def generate_test_cases(n=5):
    os.makedirs("test_images", exist_ok=True)
    
    # 1. Recupero 10 Testi SciFact (5 pure text, 5 per ibridi)
    claims = get_random_scifact_claims(n * 2)
    claims_alone = claims[:n]
    claims_hybrid = claims[n:2*n] if len(claims) >= 2*n else claims[:n]
    
    # 2. Creazione URL Pool
    urls_pool = [
        "https://en.wikipedia.org/wiki/Paracetamol", "https://en.wikipedia.org/wiki/Ibuprofen",
        "https://en.wikipedia.org/wiki/Vitamin_D", "https://en.wikipedia.org/wiki/Vaccine",
        "https://en.wikipedia.org/wiki/Aspirin", "https://en.wikipedia.org/wiki/Antibiotic"
    ]
    selected_urls = random.sample(urls_pool, min(n, len(urls_pool)))
    
    # 3. Generazione Immagini Social (Nessun download!)
    print(f"🎨 Generazione di {n} finti screenshot Social in Inglese (Test OCR)...")
    social_images = generate_social_mockups("test_images")
    
    # 4. Generazione Immagini Ibride Dinamiche
    print(f"📊 Generazione di {n} finte figure scientifiche coerenti coi claim (Test Ibridi)...")
    hybrid_images = []
    for i, claim in enumerate(claims_hybrid, 1):
        hybrid_images.append(generate_hybrid_mockup(claim, i, "test_images"))
            
    # --- STAMPA DEL REPORT FINALE ---
    print(f"\n" + "="*50)
    print(f"🚀 GENERATORE DI TEST COMPLETATO CON SUCCESSO")
    print(f"="*50)
    
    print(f"\n✅ [1] {len(claims_alone)} CLAIM TESTUALI DA SCIFACT:")
    for i, c in enumerate(claims_alone, 1): print(f"   {i}. {c}")
        
    print(f"\n✅ [2] {len(selected_urls)} URL DA TESTARE (Routing):")
    for i, u in enumerate(selected_urls, 1): print(f"   {i}. {u}")
        
    print(f"\n✅ [3] {len(social_images)} IMMAGINI SOCIAL/INFOGRAFICHE INGLESI (Test OCR):")
    for i, img in enumerate(social_images, 1): print(f"   {i}. {img}")
        
    print(f"\n✅ [4] {len(claims_hybrid)} COMBINAZIONI IBRIDE COERENTI (Testo SciFact + Finto Grafico Paper):")
    for i in range(len(claims_hybrid)):
        print(f"   {i+1}. Testo: {claims_hybrid[i]}\n      Immagine: {hybrid_images[i]}\n")

if __name__ == "__main__":
    generate_test_cases(5)