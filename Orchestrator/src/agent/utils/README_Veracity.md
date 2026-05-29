# ⚖️ Manuale d'Architettura: Veracity Assessment

Il modulo di **Veracity** rappresenta l'ultimo stadio logico della pipeline di MedFactCheck. Il suo compito è emettere una sentenza definitiva (un'etichetta di classificazione) e un punteggio di confidenza per ogni sub-claim e, infine, calcolare in maniera deterministica il verdetto globale per il claim originale dell'utente.

---

## 🤖 1. Il Modello di Classificazione (Cross-Encoder NLI)

Per questo compito discriminativo, il sistema si appoggia ad architetture specializzate in **Natural Language Inference (NLI)** (ad es. DeBERTa-v3 o varianti fine-tunizzate in ambito biomedico come PubMedBERT), come fortemente suggerito dalle specifiche del progetto.

I modelli Cross-Encoder prendono in input coppie di frasi `[Premessa, Ipotesi]` e calcolano l'allineamento semantico restituendo logit probabilistici per tre classi fisse. Nel contesto di MedFactCheck:
- **Premessa:** L'evidenza scientifica aggregata + il Ragionamento Logico (CoT) generato dal Reasoner.
- **Ipotesi:** Il sub-claim da verificare.

---

## 📊 2. Le Classi di Output e la Confidenza

L'agente di Veracity classifica la relazione in una di queste tre categorie:
- **Supported (Entailment):** Le evidenze e il ragionamento logico confermano pienamente le affermazioni del sub-claim.
- **Refuted (Contradiction):** Le evidenze smentiscono il sub-claim, oppure dimostrano che contiene esagerazioni critiche o palesi falsità mediche.
- **Not Enough Information / NEI (Neutral):** I documenti recuperati non trattano l'argomento, non sono conclusivi, o non offrono prove statistiche sufficienti per confermare o smentire il claim in modo oggettivo.

### Confidence Score
Oltre all'etichetta categorica, il modello restituisce una probabilità matematica generata dalla funzione Softmax (es. 0.98, ovvero 98%). Questo valore indica quanto il modello è "sicuro" della classificazione assegnata, permettendo alla Dashboard di filtrare o evidenziare risultati incerti.

---

## 🔄 3. Aggregazione del Verdetto Finale

Poiché un claim utente complesso (es. *"L'aspirina cura il cancro e l'ibuprofene causa infarti"*) viene frammentato in molteplici sub-claims dal Decomposer, l'agente di Veracity deve sintetizzare un **verdetto globale** per la dashboard.

La logica di aggregazione segue regole cliniche conservative a tolleranza zero:
1. Se *almeno un* sub-claim critico è **Refuted**, l'intero claim globale viene etichettato come **Refuted** (una notizia che contiene parzialmente falsità in campo medico deve essere scartata in toto).
2. Se tutti i sub-claims sono **Supported**, il claim globale è **Supported**.
3. In caso di mancanza di evidenze per parti cruciali del testo, il sistema propende per **Not Enough Information**.

Al termine dell'aggregazione, i risultati finali vengono passati al Supervisore che dichiara la fine del processo (`FINISH`), consentendo all'Orchestratore di chiudere il ciclo e salvare il documento finale in MongoDB.