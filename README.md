
# NLP

 📐 1. Similarity Computation

We compute similarities between each pair of product descriptions using two techniques:

 🔹 TF-IDF + Cosine Similarity
 🔹 Jaccard Similarity


📊 2. Sample Results


| Product 1                            | Product 2                           | Cosine Similarity | Jaccard Similarity |
| ------------------------------------ | ----------------------------------- | ----------------- | ------------------ |
| Nike SB Ishod Premium                | Nike SB Ishod Premium               | 1.000             | 1.000              |
| Nike Dri-FIT Victory                 | Nike Dri-FIT Victory                | 0.986             | 0.909              |
| Nike React Infinity 3                | Nike React Infinity 3 Premium       | 0.973             | 0.977              |
| Nike SB Ishod Wair Premium           | Nike SB Ishod Wair                  | 0.973             | 0.951              |
| Liverpool F.C. 2022/23 Stadium Third | F.C. Barcelona 2022/23 Stadium Home | 0.969             | 0.926              |
| Nike SB Ishod Wair                   | Nike SB Ishod Wair PRM              | 0.964             | 0.905              |
| Nike Air Max 90 SE                   | Nike Air Max 97                     | 0.961             | 0.947              |
| Nike SB Nyjah 3                      | Nike SB Zoom Nyjah 3                | 0.957             | 0.870              |
| Nike SB Force 58                     | Nike SB Force 58                    | 0.951             | 0.889              |
| Nike Culture of Basketball           | Nike Culture of Basketball          | 0.948             | 0.833              |

          
📌 3. Observations

- Cosine similarity highlights semantic closeness even if exact words differ.
- Jaccard is stricter, and better at exact duplicates
- TF-IDF + Cosine is more effective for identifying related product styles


