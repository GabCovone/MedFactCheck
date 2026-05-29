from PIL import Image, ImageDraw, ImageFont
import os

def genera_immagini_social():
    os.makedirs("test_social_claims", exist_ok=True)

    # Definizione degli scenari (Nome file, Colore Sfondo, Colore Testo, Testo del Post)
    scenari = [
        ("1_tweet_diabete.png", (21, 32, 43), (255, 255, 255), 
         "Tweet da @GuruSalute:\n\nIl digiuno intermittente cura definitivamente\nil diabete di tipo 2 in sole 4 settimane.\n\nSmetti di prendere l'insulina oggi stesso!"),
        
        ("2_facebook_covid.png", (240, 242, 245), (28, 30, 33), 
         "Post nel gruppo 'Mamme Informate':\n\nLa vitamina C ad alte dosi previene\ne cura totalmente l'infezione da Covid-19.\n\nNon fidatevi dei farmaci delle multinazionali!"),
        
        ("3_infografica_vaccini.png", (180, 40, 40), (255, 255, 255), 
         "!!! ALLERTA VACCINI !!!\n\nI vaccini a mRNA alterano il DNA umano\nin modo permanente e irreversibile.\n\nCausano miocardite fulminante nel 100%\ndegli atleti professionisti sani."),
        
        ("4_poster_clinico_aspirina.png", (220, 240, 220), (10, 60, 20), 
         "--- APPUNTI FARMACOLOGIA ---\n\nL'acido acetilsalicilico (Aspirina)\ninibisce la sintesi delle prostaglandine\nbloccando l'enzima ciclossigenasi (COX).\n\nQuesto processo riduce febbre e infiammazione."),
        
        ("5_articolo_asma.png", (255, 255, 255), (0, 0, 0), 
         "TITOLO: I segreti naturali\n\nLo studio dimostra che assumere 10mg\ndi melatonina a notte garantisce una cura\ndefinitiva contro l'asma pediatrico cronico.")
    ]

    print("🎨 Generazione delle fittizie immagini Social/Infografiche in corso...\n")

    for filename, bg_color, text_color, text in scenari:
        # Crea un'immagine vuota (tipo screenshot)
        img = Image.new('RGB', (900, 450), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Cerca un font di sistema leggibile per Windows (Arial)
        try:
            font = ImageFont.truetype("arial.ttf", 35)
        except IOError:
            # Fallback se arial non viene trovato
            try:
                font = ImageFont.load_default(size=35)
            except TypeError:
                font = ImageFont.load_default()
                
        # Scrive il claim medico sull'immagine
        draw.text((50, 60), text, fill=text_color, font=font, spacing=15)
        
        # Salva
        path = os.path.join("test_social_claims", filename)
        img.save(path)
        print(f"✅ Creata immagine test: {path}")

if __name__ == "__main__":
    genera_immagini_social()
    print("\nFinito! Le tue immagini con claim medici estratti da 'social' sono nella cartella test_social_claims.")