# NLP-Research-Projects
Details projects as I work on them. Click the link to find more details on these projects
1. [Legal Clause Extraction as a Generative task](https://github.com/v-acha/Statistical-Analysis/tree/main/voting_dfficulty_testing)
   - **Description:** This research project explores how modern sequence‑to‑sequence models extract legally meaningful clauses from long, unstructured contract text. The task is framed as question‑conditioned clause extraction, where each model must determine whether a clause exists and generate the correct span across chunked documents.
   - **Models & Methodology** We evaluated three architectures: T5‑Small, FLAN‑T5‑Base, and a custom BART Fusion model that incorporates cross‑chunk contextual fusion and a clause‑presence classification layer. All models were trained and tested on chunked contract data paired with clause‑specific questions, with performance measured at both chunk and document levels using EM, F1, Jaccard, BLEU, and ROUGE.
   - **Results:** 
        - FLAN‑T5‑Base achieved the strongest document‑level accuracy and most reliable clause generation, producing complete and well‑aligned spans across diverse clause types. 
        - BART Fusion contributed higher clause‑presence detection and better recovery of fragmented spans, making it a promising complementary model for hybrid setups.