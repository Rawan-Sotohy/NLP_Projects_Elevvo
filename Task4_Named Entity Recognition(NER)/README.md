# 🏷️ Named Entity Recognition (NER)

A complete **Named Entity Recognition** system built using **Python**, **Transformers (HuggingFace)**, **PyTorch**, and **Streamlit**.
The project extracts **entities** such as **Person (PER)**, **Organization (ORG)**, **Location (LOC)**, and **Miscellaneous (MISC)** from any text input.

---

## 🧠 Dataset

Dataset used: [CoNLL-2003 NER Dataset](https://www.kaggle.com/datasets/juliangarratt/conll2003-dataset)

The dataset contains text with annotated named entities:

* **eng.train** → Training data
* **eng.testa** → Validation data
* **eng.testb** → Test data

---

## 🚀 Features

* Clean and preprocess text data
* Train **BERT-based NER model**
* Extract multiple entity types:

  * **PER** → Person names
  * **ORG** → Organizations
  * **LOC** → Locations
  * **MISC** → Miscellaneous entities (events, products, etc.)
* Evaluate model using **Accuracy**, **F1-Score**, and entity-level metrics
* Deploy using **Streamlit Web App**
* Interactive visualization with **highlighted entities**, **statistics**, and **entity tables**

---

## 📝 How to Use

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Enter your own text or select an example to extract named entities.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **HuggingFace Transformers**
* **PyTorch**
* **Pandas**
* **Streamlit**

 ---

## 📸 Screenshots

**Home Page**

![Home Page](https://raw.githubusercontent.com/Rawan-Sotohy/NLP_Projects_Elevvo/main/Task4_Named%20Entity%20Recognition%28NER%29/images/home.jpg)


![Home Page](https://raw.githubusercontent.com/Rawan-Sotohy/NLP_Projects_Elevvo/main/Task4_Named%20Entity%20Recognition%28NER%29/images/home2.jpg)


---

**Example Entity Extraction**

![Home Page](https://raw.githubusercontent.com/Rawan-Sotohy/NLP_Projects_Elevvo/main/Task4_Named%20Entity%20Recognition%28NER%29/images/ex.jpg)


![Home Page](https://raw.githubusercontent.com/Rawan-Sotohy/NLP_Projects_Elevvo/main/Task4_Named%20Entity%20Recognition%28NER%29/images/exx.jpg)


---

## 💡 Notes

* Large model files are **not included** in the repository. Use the `.gitignore` to avoid uploading heavy files.

* To use the full trained model, you can either:

  1. Train it locally using `train.py`
  2. Or download pretrained weights and place them in `models/bert-ner/`

* Confidence threshold for entity extraction can be adjusted in the **Streamlit sidebar**.

