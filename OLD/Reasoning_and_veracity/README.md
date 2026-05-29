# 📑 Manuale d'Architettura: Modulo di Reasoning & Veracity Assessment

Il modulo di **Reasoning and Veracity** rappresenta il motore decisionale conclusivo dell'architettura Multi-Agente di **MedFactCheck**. Il suo obiettivo è processare i sub-claim generati dal Decomposer e le relative evidenze fornite dal Retriever per formulare un verdetto scientifico definitivo e semanticamente motivato.

Per superare i limiti tipici delle architetture *single-model*, questo modulo adotta una **Pipeline Ibrida a Cascata** che unisce le elevate capacità generative e logiche di un LLM (per la costruzione argomentativa) all'intransigenza matematica di un classificatore specializzato in *Natural Language Inference* (NLI).

---

## 🏗️ Diagramma dell'Architettura Ibrida

```text
                    [Input: Sub-Claim + Evidenze]
                                  │
                 (Controllo Disponibilità Evidenze)
                                  │
                  ┌───────────────┴────────────────┐
                  ▼                                ▼
            (Assenti)                         (Presenti)
     [Short-Circuit Logic]           [Qwen2.5-VL: Reasoner Agent]
    (Bypass dei Tensori IA)          (Pattern "Detective Neutrale")
                  │                                │
                  ▼                                ▼
          [Verdetto: NEI]              [Chain-of-Thought Logico]
        (Confidence: 1.0)                          │
                  │                                ▼
                  │                   [DeBERTa-v3: Veracity Agent]
                  │                (Cross-Encoder NLI Classification)
                  │                                │
                  └───────────────┬────────────────┘
                                  ▼
                     [Verdetto Finale + Confidence Score]
```

---

## 🧠 1. Generazione del Ragionamento (Chain-of-Thought)

Per la prima fase valutativa, il sistema delega il task cognitivo a **Qwen2.5-7B-VL-Instruct**. In questo strato architetturale, l'agente non funge da giudice, bensì da **"Detective Neutrale"**: il suo compito esclusivo è analizzare l'intersezione semantica tra il sub-claim e i paper scientifici, emettendo una stringa di ragionamento logico (*Chain-of-Thought*).

Sfruttando un *Few-Shot Prompting* rigoroso, l'agente viene vincolato a mantenere una totale imparzialità. La rete neurale si limita a esporre le discrepanze, le iperboli o le conferme cliniche (es. *"Il claim postula l'effetto X assoluto, mentre lo Studio A descrive un rischio raro e autolimitante"*), senza mai anticipare una label definitiva. Questa *Context Enrichment* è essenziale per fornire al classificatore a valle una premessa chiara ed esplosa.

### 1.1 Inferenza Gerarchica (Map-Reduce) e Greedy Decoding
Al fine di non saturare la *Context Window* limitata dell'LLM, qualora il Retriever estragga un pool massivo di documenti (es. multipli testi ad alta densità informativa), il Reasoner instanzia dinamicamente un pattern algoritmico **Map-Reduce**. Le evidenze vengono frazionate in micro-batch sequenziali, di cui il modello distilla i concetti chiave in sintesi intermedie. Solo in un secondo momento, le sintesi vengono compattate in un CoT globale.
Inoltre, l'intera fase di generazione avviene imponendo `do_sample=False`. Il **Greedy Decoding** disabilita le diramazioni probabilistiche, garantendo l'output del ragionamento alla massima velocità e neutralizzando il rischio di allucinazioni interpretative.

---

## ⚖️ 2. Classificazione NLI tramite Cross-Encoder (DeBERTa-v3)

Il verdetto matematico conclusivo è delegato all'agente di Veracity, basato su **DeBERTa-v3-base**, un modello a base Transformer fine-tunato specificamente per la deduzione logica (*Natural Language Inference*).

### 2.1 Short-Circuit Evaluation (Fail-Fast)
Prima di inizializzare l'onerosa allocazione matriciale dell'inferenza, il software effettua un controllo di blocco (*Gatekeeping*). Qualora il Retriever non abbia trovato alcuna documentazione per il sub-claim analizzato, il sistema esegue uno **Short-Circuit logico**: salta totalmente l'invocazione dei modelli IA e assegna in automatico il verdetto hardware *Not Enough Information* con confidenza assoluta ($1.0$), risparmiando ingenti risorse di calcolo.

### 2.2 Architettura Cross-Encoder e Mappatura Label
Rispetto ai classici Bi-Encoder (che calcolano la similarità coseno tra i due vettori isolati), DeBERTa opera come **Cross-Encoder**. Elabora congiuntamente l'Ipotesi (il sub-claim) e la Premessa (il CoT + Evidenze) all'interno della stessa enorme matrice di *Self-Attention*, valutando le dipendenze logiche di ogni singolo token incrociato.
L'output finale attraversa uno strato *Softmax*, generando una distribuzione di probabilità su tre classi NLI, che MedFactCheck mappa nel seguente modo:
* **Entailment $\rightarrow$ Supported**: La letteratura conferma esplicitamente il claim.
* **Contradiction $\rightarrow$ Refuted**: La letteratura smentisce o corregge clinicamente il claim.
* **Neutral $\rightarrow$ Not Enough Information**: La letteratura discussa non contiene dati sufficienti a formulare un verdetto.

Il valore di probabilità assoluto (logit Softmax) assegnato alla classe vincente viene estratto e propagato come **Confidence Score** dell'intero sistema.

---

## ⚙️ 3. Memory Management e Mitigazione degli OOM

Dato che questo modulo esige la convivenza di due pesanti architetture neurali (Qwen da 7B parametri e DeBERTa) nello stesso ambiente VRAM (15GB target), le strategie di ottimizzazione hardware sono critiche.

### 3.1 Condivisione Dinamica del Dispositivo (PCIe Alignment)
Il classificatore NLI viene inizializzato vincolandolo esplicitamente allo stesso puntatore hardware dell'LLM (`device=self.reasoning_model.device`). In configurazioni hardware complesse, questo impedisce la frammentazione della memoria o il passaggio latente di tensori attraverso il bus della motherboard, obbligando i due modelli a sfruttare gli stessi blocchi fisici in maniera perfettamente sequenziale.

### 3.2 OOM-Proofing sulle Matrici NLI
L'architettura Self-Attention di un Cross-Encoder presenta una complessità spaziale che scala quadraticamente ($O(n^2)$) in base alla lunghezza del testo. Per neutralizzare la minaccia dei crash per *Out-of-Memory* in caso di paper anomali, l'inferenza di DeBERTa è plafonata a livello tensoriale imponendo `truncation=True` e `max_length=512`. In questo modo la matrice di attenzione interna non eccede mai le dimensioni fisiche massime allocabili dalla GPU.

### 3.3 Aggressive Garbage Collection
A chiusura di ogni task di classificazione su un singolo sub-claim, un costrutto di deallocazione entra in azione. Oltre a distruggere manualmente i puntatori ai tensori Pytorch (`del model_inputs`, `del generated_ids`), il flusso esegue una chiamata in successione a `torch.cuda.empty_cache()` e al *Garbage Collector* di sistema (`gc.collect()`). Questo approccio "svuota e spazza" elimina le tracce degli *activation tensors* di DeBERTa, assicurando la VRAM intatta per quando l'orchestratore dovrà risvegliare Qwen per il sub-claim successivo.
