# 📑 Manuale d'Architettura: Modulo di Ingestion & Claim Decomposition

Il modulo di **Ingestion & Claim Decomposition** costituisce la prima fase di elaborazione attiva dell'architettura Multi-Agente di **MedFactCheck**. Il suo scopo è ricevere i dati non strutturati ed eterogenei (testo libero, URL, immagini mediche) delegati dal Supervisore, normalizzarli, ed estrarne proposizioni atomiche scientificamente verificabili, instradandole verso i corretti rami di Retrieval.

Il motore cognitivo alla base di questa fase è **Qwen2.5-7B-VL-Instruct**, un Large Language Model con architettura nativamente Vision-Language (VL). Il modello opera in regime di *few-shot prompting*, eliminando la necessità di fine-tuning o dell'impiego di deboli e proni ad errore software OCR (Optical Character Recognition) di terze parti.

---

## 🏗️ Diagramma del Flusso di Ingestion

```text
                     [Input Grezzo: Testo, URL, Immagine]
                                   │
                                   ▼
                   [Agent: Autonomous Tool-Calling]
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
       (Link Web)            (File Immagine)        (Testo Puro)
    [Web Scraper]          [Vision Validator]     [Pass-through]
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                   │
                                   ▼
                         [Input Normalizzato]
                                   │
                                   ▼
                     [Qwen2.5-VL: Decomposer]
                    (Greedy Decoding, SDPA, NF4)
                                   │
                                   ▼
                     [Decomposizione in JSON]
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          [Ramo "kb"]                   [Ramo "kb", "lit"]
    (Assiomi statici, chimica)    (Trial clinici, cause, effetti)
```

---

## 📥 1. Ingestion Autonoma e Pattern di Tool-Calling

Per gestire l'eterogeneità degli input, il modulo implementa un pattern di **Autonomous Tool-Calling**. Prima di iniziare la vera e propria scomposizione, un prompt dedicato interroga l'LLM passando l'input grezzo dell'utente. L'agente valuta la semantica della stringa e decide in autonomia quale strumento software (`Tool`) innescare:

* **Scraping Web (`scrape_text_from_url`)**: Se l'input nasconde l'intenzione di verificare una pagina web (es. un link a Wikipedia o a un blog), il tool estrae l'albero DOM, esegue lo stripping del markup HTML tramite `BeautifulSoup` e inietta il testo pulito nel contesto.
* **Vision Analysis (`validate_image`)**: Se viene riconosciuto il percorso locale a un file immagine o un URL multimediale, il tool ne valida gli header e l'integrità del buffer (es. tramite `Pillow`), istruendo la pipeline a elaborare i tensori visivi al momento della generazione.
* **Testo Puro (`nessuno`)**: Se l'input è una stringa dichiarativa standard, l'LLM restituisce il token di bypass, procedendo all'elaborazione diretta senza latenze I/O.

---

## 🧩 2. Claim Decomposition e Routing Dinamico

Il nucleo metodologico della fase risiede nella funzione `decompose`. Tramite *Few-Shot Prompting*, l'LLM è addestrato a comportarsi come un estrattore sintattico inflessibile, restituendo esclusivamente una stringa formattata come **JSON**. 

Il processo applica regole di filtraggio logico (*Entity Resolution* per pronomi sottintesi, ed eliminazione del "rumore aneddotico", preservando però proposizioni mediche estreme che richiedono confutazione) ed esegue l'esplosione della frase in **Sub-Claim atomici**.

Per ogni sub-claim, l'LLM esegue un'analisi semantica per assegnare la **Topologia di Routing** ottimale:

* **Ramo `["kb"]` (Knowledge Base Locale):** Assegnato a sub-claim che esprimono assiomi chimici, biologici, definizioni statiche o tassonomie (es. *"L'ibuprofene è un antinfiammatorio non steroideo"*). Evita di interrogare l'intera letteratura medica (LIT) per fatti puramente manualistici, risparmiando API calls e cicli di Tensor Core.
* **Ramo `["kb", "lit"]` (Ibrido Intensivo):** Assegnato ad affermazioni che postulano azioni, cause, trial clinici, effetti o alterazioni sistemiche (es. *"L'ibuprofene riduce drasticamente l'infiammazione articolare"*). Innesca la ricerca incrociata profonda.

---

## ⚙️ 3. Pipeline di Ottimizzazione Hardware (Memory Management)

Per far coesistere modelli da svariati miliardi di parametri all'interno di ambienti con VRAM fortemente limitata (es. GPU da 15/16GB), l'istanziazione di Qwen è stata sottoposta a tecniche avanzate di ingegneria dei tensor.

### 3.1 Quantizzazione Pesi Statici (NF4)
Il modello viene caricato sfruttando la libreria `BitsAndBytesConfig` con tipo di dato `nf4` (NormalFloat 4-bit) e computazione in `float16`. Questa tecnica matematica comprime l'impronta statica della rete neurale da ~14 GB a circa **5.5 GB**. Il formato NormalFloat, rispetto alla quantizzazione intera tradizionale (INT4), preserva una distribuzione normale dei pesi, garantendo che le capacità analitiche dell'LLM rimangano pressoché identiche all'FP16 nativo.

### 3.2 Ottimizzazione KV Cache: SDPA (Scaled Dot-Product Attention)
La memoria dinamica allocata durante l'inferenza (KV Cache) rischia di causare Out-Of-Memory. L'architettura di Qwen2.5-VL si affida all'M-RoPE (*Multimodal Rotary Positional Embedding*) per mappare simultaneamente le dimensioni del testo (1D) e i pixel dell'immagine (2D). Poiché algoritmi generici di *Cache Quantization* rischierebbero di troncare asimmetricamente i tensori visivi corrompendo l'estrazione dati dall'immagine, il sistema demanda l'ottimizzazione in modo sicuro a PyTorch abilitando nativamente l'**SDPA**. Ciò accelera la generazione e comprime i picchi di memoria dal 30% al 50%.

### 3.3 Generazione Deterministica (Greedy Decoding)
Il campionamento probabilistico (*sampling*) è controproducente quando l'obiettivo è estrarre codice JSON macchina. Nei metodi `decompose`, `reason` e `decide_tool` il parametro di generazione `do_sample` è stato forzato esplicitamente a `False`. Disattivando le fluttuazioni probabilistiche, l'LLM esegue il cosiddetto *Greedy Decoding*: sceglie sempre e solo il token con la probabilità più alta. Questo annulla totalmente le allucinazioni sintattiche e massimizza la velocità di propagazione dei tensori.

### 3.4 Aggressive Garbage Collection
In un sistema Multi-Agente orchestrato (LangGraph), la deallocazione deve essere istantanea. Al termine del costrutto `with torch.no_grad():`, il codice Python cancella esplicitamente le variabili che puntano ai tensori temporanei (`del inputs`, `del generated_ids`), forzando l'intervento simultaneo di `torch.cuda.empty_cache()` e del Garbage Collector nativo `gc.collect()`. Questo riporta rigorosamente la VRAM al suo livello di riposo.