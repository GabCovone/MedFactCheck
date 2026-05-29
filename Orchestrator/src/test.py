import requests
import random
import json
import os
from PIL import Image, ImageDraw, ImageFont

# Cartella dove l'utente ha salvato le immagini cliniche e dove verranno create le social
IMG_DIR = "test_image" 

def get_random_scifact_claims(n=5):
    print(f"⏳ Scaricando {n} claim da SciFact (tramite MTEB) per i test puramente testuali...")
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
    """Genera finti screenshot social in INGLESE per test OCR e Fact-Checking."""
    scenari = [
        ("social_1_tweet_diabetes.png", (21, 32, 43), (255, 255, 255), "Tweet by @HealthGuru:\n\nIntermittent fasting completely cures\ntype 2 diabetes in just 4 weeks.\n\nStop taking insulin today!"),
        ("social_2_facebook_covid.png", (240, 242, 245), (28, 30, 33), "Group Post 'Informed Moms':\n\nHigh doses of Vitamin C completely\nprevent and cure Covid-19 infection.\n\nDon't trust Big Pharma drugs!"),
        ("social_3_infographic_vaccines.png", (180, 40, 40), (255, 255, 255), "!!! VACCINE ALERT !!!\n\nmRNA vaccines permanently and\nirreversibly alter human DNA.\n\nThey cause fulminant myocarditis in 100%\nof healthy professional athletes."),
        ("social_4_poster_aspirin.png", (220, 240, 220), (10, 60, 20), "--- PHARMACOLOGY NOTES ---\n\nAcetylsalicylic acid (Aspirin)\ninhibits prostaglandin synthesis by\nblocking the cyclooxygenase (COX) enzyme.\n\nThis process reduces fever and inflammation."),
        ("social_5_article_asthma.png", (255, 255, 255), (0, 0, 0), "TITLE: Natural Secrets\n\nStudies show that taking 10mg of\nmelatonin nightly provides a permanent\ncure for chronic pediatric asthma.")
    ]
    
    generated_paths = []
    os.makedirs(output_dir, exist_ok=True)
    
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

def get_hybrid_test_cases(img_dir):
    """Associa le immagini cliniche salvate dall'utente ai Claim in Inglese corretti."""
    
    hybrid_cases = [
        {
            "img": "COVID-19_Chest_X-ray.jpg",
            "claim": "The pulmonary infection shown in this X-ray (Covid-19) can be permanently and quickly cured by taking broad-spectrum antibiotics such as amoxicillin."
        },
        {
            "img": "Melanoma.jpg",
            "claim": "This skin lesion, despite being asymmetrical with irregular borders, is a harmless cherry angioma that has no chance of evolving into a malignant tumor."
        },
        {
            "img": "Normal_ECG_-_12_lead.jpg",
            "claim": "This 12-lead ECG tracing shows obvious signs of an acute myocardial infarction (STEMI) with severe ongoing ischemia."
        },
        {
            "img": "Ibuprofen-3D-vdW.jpg",
            "claim": "This molecule (Ibuprofen) generates its strong analgesic effect by binding directly to the opioid receptors of the central nervous system, similar to morphine."
        },
        {
            "img": "Syringe.jpg",
            "claim": "mRNA vaccine injections contain graphene micro-particles that accumulate in the blood and cause irreversible thrombosis in 100% of patients."
        }
    ]
    
    formatted_cases = []
    for idx, case in enumerate(hybrid_cases, 1):
        full_path = os.path.abspath(os.path.join(img_dir, case["img"]))
        status = "✅ TROVATA" if os.path.exists(full_path) else "❌ MANCANTE"
        formatted_cases.append(
            f"   {idx}. Testo: {case['claim']}\n      Immagine ({status}): {full_path}\n"
        )
    return formatted_cases

def generate_test_cases():
    print(f"Inizializzazione ambiente di test in: {os.path.abspath(IMG_DIR)}\n")
    
    # 1. Recupero Testi SciFact (Solo 5 per il test testuale puro)
    claims_alone = get_random_scifact_claims(5)
    
    # 2. Creazione URL Pool per test Routing
    urls_pool = [
        "https://en.wikipedia.org/wiki/Paracetamol", "https://en.wikipedia.org/wiki/Ibuprofen",
        "https://en.wikipedia.org/wiki/Vitamin_D", "https://en.wikipedia.org/wiki/Vaccine",
        "https://en.wikipedia.org/wiki/Aspirin"
    ]
    
    # 3. Generazione Immagini Social
    print(f"🎨 Generazione di 5 finti screenshot Social in Inglese (Test OCR)...")
    social_images = generate_social_mockups(IMG_DIR)
    
    # 4. Associazione Ibrida
    print(f"🔗 Mappatura dei claim medici sulle immagini cliniche locali...")
    hybrid_cases_formatted = get_hybrid_test_cases(IMG_DIR)
            
    # --- STAMPA DEL REPORT FINALE ---
    print(f"\n" + "="*50)
    print(f"🚀 GENERATORE DI TEST COMPLETATO CON SUCCESSO")
    print(f"="*50)
    
    print(f"\n✅ [1] 5 CLAIM TESTUALI PURI (Da incollare in Streamlit senza immagini):")
    for i, c in enumerate(claims_alone, 1): print(f"   {i}. {c}")
        
    print(f"\n✅ [2] 5 URL DA TESTARE (Per verificare il Knowledge Base Routing):")
    for i, u in enumerate(urls_pool, 1): print(f"   {i}. {u}")
        
    print(f"\n✅ [3] 5 IMMAGINI SOCIAL/INFOGRAFICHE (Carica l'immagine senza testo per testare l'OCR di Qwen):")
    for i, img in enumerate(social_images, 1): print(f"   {i}. {img}")
        
    print(f"\n✅ [4] 5 COMBINAZIONI IBRIDE (Incolla il testo E carica l'immagine associata):")
    for case in hybrid_cases_formatted:
        print(case)

if __name__ == "__main__":
    generate_test_cases()