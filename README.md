# SAIL Training Recommendation System

> **Semantic matching of employee development needs with relevant training courses using Sentence Transformers and cosine similarity.**

A machine-learning-based **Training Recommendation System** developed during a SAIL internship to automate the matching of employee development needs with available training/course documents.

The system accepts an employee training-needs dataset and a collection of course documents, converts their textual content into semantic embeddings, calculates similarity scores, identifies the best-matching course for each employee requirement, and generates both a **CSV similarity matrix** and a **visual PDF recommendation report**.

---

## Overview

Organizations may have a large number of employee development requirements alongside numerous training programs. Manually mapping each requirement to the most relevant course can become time-consuming and difficult to scale.

This project addresses that problem using **Natural Language Processing (NLP)** and **semantic similarity**.

Instead of relying on exact keyword matching, the system represents both:

* Employee **Development Needs**
* Course-document **content**

as dense vector embeddings and compares them using **cosine similarity**.

### Workflow

```text
Employee Development Needs
            │
            ▼
     Text Preprocessing
            │
            ▼
 SentenceTransformer Embeddings
            │
            │
            ├──────────────┐
            │              │
            ▼              ▼
 Course Documents     Employee Needs
            │              │
            ▼              ▼
   Document Extraction   Embeddings
            │              │
            └──────┬───────┘
                   ▼
          Cosine Similarity
                   │
                   ▼
       Best Course Recommendation
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   Similarity Matrix      PDF Report
        (.csv)             (.pdf)
```

---

## Key Features

* **Semantic text matching** using Sentence Transformers
* Supports employee requirement data in **Excel/CSV workflows**
* Supports course documents in **PDF and DOCX formats**
* Automatically identifies the **best-matching course** for every training need
* Calculates similarity against **all available courses**
* Generates a complete **course similarity matrix**
* Produces a visual **PDF recommendation report**
* Provides:

  * Top 5 most relevant courses
  * Bottom 5 least relevant courses
  * Most frequently recommended courses
  * Similarity-score distribution
  * Correlation heatmap
  * Courses with low recommendation scores for review

---

## Dataset & Results

In the demonstrated run, the system processed:

| Metric                        |     Result |
| ----------------------------- | ---------: |
| Training-need entries         |  **8,032** |
| Course documents              |     **32** |
| Average best-match similarity | **0.4170** |
| Highest similarity            | **0.7138** |
| Lowest similarity             | **0.0451** |
| Standard deviation            | **0.1043** |

The generated similarity matrix contains **8,032 rows and 35 columns**, including the SAIL PNO, best-matching course, highest similarity score, and similarity scores for the available course documents.

The generated report shows that the highest observed similarity was approximately **0.71**, while the overall distribution of best-match scores was concentrated around the **0.4–0.5 range**.

---

## Recommendation Output

For every employee development requirement, the system calculates:

```text
SAIL PNO
Best Match Course
Highest Similarity Score
Similarity Score → Course 1
Similarity Score → Course 2
...
Similarity Score → Course N
```

Example conceptually:

```text
SAIL PNO       : Employee_001
Best Match     : HSPM.pdf
Similarity     : 0.7138
```

This allows users to see not only the recommended course but also how strongly the requirement matched the available alternatives.

---

## Generated Analytics

### 1. Top Relevant Courses

The system identifies the highest-scoring recommendations and visualizes their similarity scores.

The generated report's top-course analysis shows **HSPM.pdf** among the highest-scoring recommendations, with scores reaching approximately **0.71**.

### 2. Least Relevant Courses

The system also identifies low-scoring matches. In the generated report, examples include:

* `LYM.docx`
* `PRGMM.docx`
* `CRM.docx`

with scores around **0.05–0.06** in the lowest-match analysis.

### 3. Recommendation Frequency

The system counts how often each course is selected as the **best match**.

The generated analysis shows `TOTO.docx` as the most frequently selected course, followed by courses including `HSPM.pdf`, `BFIP.docx`, `RCF.docx`, and `HSPM1.docx`.

### 4. Similarity Distribution

A histogram is generated to visualize the distribution of the highest similarity scores across all training requirements.

The observed scores range from approximately **0.045 to 0.714**, with a substantial concentration around the **0.4–0.5** range.

### 5. Low-Similarity Review

The system flags courses associated with recommendations below a similarity threshold of **0.15**.

For example, the generated report identified:

| Course     | Low-score recommendations |
| ---------- | ------------------------: |
| CRM.docx   |                        21 |
| WOMEN.docx |                        16 |
| ADCCO.docx |                        11 |
| PRGMM.docx |                         8 |
| LYM.docx   |                         7 |

This provides a useful mechanism for identifying areas where the available training catalogue may not strongly address employee development requirements.

---

## Technology Stack

### Machine Learning

* **Python**
* **Sentence Transformers**
* `all-MiniLM-L6-v2`
* Cosine Similarity
* Scikit-learn

### Document Processing

* **PyMuPDF (`fitz`)** — PDF text extraction
* **python-docx** — DOCX text extraction
* **Pandas** — dataset processing

### Visualization & Reporting

* Matplotlib
* Seaborn
* FPDF

---

## Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts employee development needs and course-document text into semantic embeddings.

Similarity is then calculated using cosine similarity:

```python
scores = cosine_similarity(
    [employee_embedding],
    course_embeddings
)[0]
```

The course with the highest similarity score is selected as the recommended course:

```python
best_idx = scores.argmax()
```

---

## Input

### Employee Training-Needs File

The application expects a CSV/XLSX-style dataset containing:

```text
SAIL PNO
DEVELOPMENT NEEDS
```

The implementation searches for the appropriate header row and normalizes column names before processing the data.

### Course Documents

The system accepts:

```text
.pdf
.docx
```

course documents.

Each document is converted to text before generating its semantic embedding.

---

## Output

The system produces:

### CSV Similarity Matrix

```text
all_course_similarity_matrix.csv
```

Contains:

* SAIL PNO
* Best Match Course
* Highest Similarity Score
* Similarity score for every available course

### PDF Recommendation Report

```text
recommendation_report.pdf
```

Contains:

* Overall matching statistics
* Top-course analysis
* Bottom-course analysis
* Course recommendation frequency
* Similarity distribution
* Correlation heatmap
* Low-similarity course review
* 
## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/sail-training-recommendation.git
cd sail-training-recommendation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask application:

```bash
python app_v2.py
```

The application will start locally at:

```text
http://127.0.0.1:5000
```

Open the URL in a browser and upload:

1. Employee training-needs dataset
2. Course PDF/DOCX documents

The application then performs the semantic matching and generates the output files.

---

## Core Processing Logic

The recommendation pipeline follows these steps:

### 1. Extract training requirements

The system reads the employee dataset and extracts:

```text
SAIL PNO
DEVELOPMENT NEEDS
```

### 2. Generate embeddings

Employee requirements are encoded using:

```python
model.encode(needs)
```

Course documents are similarly encoded after extracting their text.

### 3. Calculate similarity

Each employee requirement is compared against every course embedding using cosine similarity.

### 4. Select the best match

The course with the highest similarity score becomes the recommended course.

### 5. Generate analytics

The resulting recommendations are aggregated to calculate:

* Average score
* Minimum score
* Maximum score
* Standard deviation
* Course recommendation frequency
* Low-score recommendations

### 6. Generate reports

The results are exported as both:

```text
CSV → detailed similarity matrix
PDF → management-oriented analytical report
```

---

## Important Design Consideration

The system is designed as a **recommendation/decision-support tool**, not as a replacement for human judgment.

A high cosine-similarity score indicates stronger semantic similarity between a development need and course content, but it does not by itself establish that a course is definitively appropriate for an employee.

The low-score analysis can therefore be used to identify cases that may benefit from **human review or additional training content**.

---

## Future Improvements

Potential extensions include:

* Top-*K* course recommendations instead of only the top match
* Configurable similarity thresholds
* Improved document chunking for long course documents
* Metadata-aware recommendations using department, role, level, or competency
* Course descriptions and competency tags
* Interactive dashboard for HR/training teams
* Human-feedback loop for recommendation validation
* Evaluation against manually verified recommendations
* More advanced embedding/reranking models
* Authentication and role-based access control
* Database-backed storage instead of file-based processing

---

## Project Outcome

The project demonstrates how **Semantic similarity can be applied to employee development and corporate training workflows**, transforming unstructured development requirements and course documents into structured, searchable recommendations and analytical reports.

The demonstrated pipeline successfully processed **8,032 employee development-need records across 32 course documents** and generated a complete similarity matrix together with a multi-section recommendation report.

---

## Author

**Mayank Raj**

B.Tech — Artificial Intelligence & Machine Learning
Birla Institute of Technology, Mesra

