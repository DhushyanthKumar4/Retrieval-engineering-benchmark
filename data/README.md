#  Dataset: TyDi QA Multilingual Retrieval Corpus

This folder contains documentation for the dataset used in the **Multilingual Retrieval-Augmented QA Benchmarking System**.

⚠️ NOTE:
Raw dataset is NOT stored in this repository due to size constraints.  
Only preprocessing scripts and schema definitions are included.

---

## 📊 Dataset Overview

**Dataset Name:** TyDi QA (Typologically Diverse QA Dataset)  
**Type:** Multilingual Question Answering Dataset  
**Languages:** 11+ (including Indonesian, Japanese, Bengali, Telugu, Swahili, Arabic, Korean, Russian, Finnish, English, etc.)

**Purpose:**
Used to benchmark multilingual retrieval systems with sparse, dense, and hybrid search architectures.

---

## 📁 Data Format

Each sample contains the following structured fields:

### 🔹 Core Fields

- `question_text` → Natural language question
- `document_title` → Title of source document
- `document_plaintext` → Full document text
- `language` → Language of the sample
- `document_url` → Source Wikipedia URL

---

### 🔹 Retrieval Fields

- `passage_answer_candidates` → Candidate passages extracted from document
- `annotations` → Ground-truth answer annotations
- `passage_answer_candidate_index` → Index of correct passage (if available)

---

### 🔹 Byte-Level Alignment Fields

Used for precise span extraction:

- `plaintext_start_byte`
- `plaintext_end_byte`
- `minimal_answers_start_byte`
- `minimal_answers_end_byte`

---

##  Example Sample Structure

```json
{
  "question_text": "berapakah jenis ras yang ada didunia?",
  "document_title": "Ras manusia",
  "language": "indonesian",
  "document_plaintext": "Ras manusia adalah ...",
  "document_url": "https://id.wikipedia.org/wiki/Ras_manusia",
  "passage_answer_candidates": [
    {
      "text": "Ras manusia adalah suatu sistem klasifikasi...",
      "start_byte": 1,
      "end_byte": 659
    }
  ],
  "annotations": {
    "yes_no_answer": "NONE",
    "minimal_answers": []
  }
}


https://huggingface.co/datasets/google-research-datasets/tydiqa
