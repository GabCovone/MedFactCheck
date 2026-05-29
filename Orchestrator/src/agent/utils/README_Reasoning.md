# 🧠 Manuale d'Architettura: Reasoning Agent

Il modulo di **Reasoning** (Ragionamento) interviene nella pipeline di MedFactCheck subito dopo il recupero delle evidenze scientifiche. Il suo ruolo è colmare il divario semantico tra i documenti grezzi recuperati e il claim originale da verificare.

---

## 🎯 1. L'Obiettivo: Interpretability & Chain-of-Thought (CoT)

In molte pipeline standard di Retrieval-Augmented Generation (RAG), le evidenze vengono passate direttamente a un classificatore o mostrate grezze all'utente. Questo approccio a "scatola nera" non fornisce all'utente alcuna spiegazione sul *perché* una certa evidenza smentisca o supporti un claim.

Il Reasoning Agent di MedFactCheck risolve questo problema generando una **Chain-of-Thought (CoT)**, ovvero un paragrafo logico e discorsivo che emula il ragionamento deduttivo di un analista biomedico esperto.

---

## ⚙️ 2. Funzionamento dell'Agente

L'agente (basato su un prompt specifico inviato a **Qwen2.5-VL-7B-Instruct**) riceve in input due elementi:
1. Il singolo **sub-claim** atomico prodotto dal Decomposer.
2. I **top-k paragrafi di evidenza** (chunk) recuperati e rerankati dal Retriever.

Attraverso un prompt strutturato *Few-Shot*, al modello viene richiesto di sintetizzare i risultati clinici riportati nell'evidenza, relazionarli esplicitamente con il claim e mettere in risalto discrepanze di magnitudo, iperboli, correlazioni errate o conferme empiriche.

### Esempio Operativo
> **Claim:** "Il farmaco X causa un'impennata di morti per infarto in tutti i pazienti."  
> **Evidenza:** "Il farmaco X presenta un rischio raro (0.01%) di miocardite autolimitante."  
> **Ragionamento Generato:** "Le evidenze indicano che il farmaco X comporta un rischio raro e autolimitante di miocardite. Il claim altera e ingigantisce enormemente questo dato clinico, trasformando un rischio minimo in una 'impennata di morti' e generalizzandolo a 'tutti i pazienti'. A causa di questa grave iperbole, le evidenze smentiscono la portata estrema del claim."

---

## 🔗 3. Sinergia con la Veracity

L'output discorsivo prodotto dal Reasoner viene salvato in MongoDB per la trasparenza lato Dashboard, e soprattutto viene **passato come contesto arricchito** al modulo di Veracity. 
Il classificatore NLI finale non deve più capire da solo testi medici complessi, ma può basarsi sull'analisi logica già pre-masticata dal Reasoner, aumentando drasticamente l'accuratezza predittiva.