# 🧩 Manuale d'Architettura: Claim Ingestion & Decomposition

Il modulo di **Claim Ingestion & Decomposition** è il punto di ingresso dell'architettura Multi-Agente di MedFactCheck. La sua responsabilità è accogliere l'input dell'utente (di qualsiasi natura) e trasformarlo in una serie di *sub-claims* atomici e interrogabili.

---

## 🤖 1. L'Agente IngestionAndDecomposer

L'agente è mosso dal LLM **Qwen2.5-VL-7B-Instruct**, configurato per operare nativamente in modalità Multimodale tramite quantizzazione NF4 a 4-bit (ottimizzando l'occupazione in VRAM).

### Fasi Operative:

#### Fase 1: Tool Selection (Routing Dinamico dell'Input)
L'agente ispeziona l'input grezzo (un testo, un link web, o il percorso di un'immagine temporanea). Sfruttando le sue capacità decisionali e il pattern di tool-calling, seleziona autonomamente lo strumento più appropriato:
- `scrape_text_from_url`: Se rileva un link, scarica e pulisce il contenuto HTML della pagina web.
- `validate_image`: Se rileva il path di un'immagine, la valida per l'analisi visiva.
- `nessuno`: Se l'utente ha inserito testo libero, procede direttamente all'analisi logica.

#### Fase 2: Elaborazione Multimodale (Vision-Language)
Se l'input è (o contiene) un'immagine medica (es. radiografie, grafici di paper, screenshot di social media), il modello utilizza la sua architettura *Vision* integrata per "guardare" l'immagine ed estrarne il contesto testuale, medico o visivo senza bisogno di moduli OCR esterni, minimizzando i tassi di errore sui referti.

#### Fase 3: Decomposizione (Atomic Claim Extraction)
I testi complessi o ambigui vengono scomposti in **proposizioni dichiarative indipendenti** (Soggetto + Verbo + Oggetto).
*Esempio:*
> Input: "La radice magica contiene vitamina C e cura il cancro al polmone."
> Output:
> 1. "La radice magica contiene vitamina C."
> 2. "La radice magica cura il cancro al polmone."

---

## 🚦 2. Routing Predittivo (KB vs LIT)

Durante la scomposizione, l'LLM analizza la natura semantica di ogni singolo sub-claim e applica regole di classificazione severe per decidere a quale modulo di ricerca affidarlo:
- **Rotta `["kb"]`**: Assegnata a definizioni biologiche, tassonomiche o di composizione statica (es. "L'aspirina è un antinfiammatorio"). Il Retriever interrogherà solo la Knowledge Base dogmatica (DisGeNET).
- **Rotta `["kb", "lit"]`**: Assegnata ad azioni cliniche, correlazioni o cause-effetto (es. "L'aspirina riduce il rischio di infarto"). Verrà interrogata sia la KB che la letteratura mondiale (Europe PMC).

L'output finale dell'agente è un JSON strutturato e pronto per essere passato al Retriever Agent.