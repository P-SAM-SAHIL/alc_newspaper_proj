
# English A La Carte (ALC) Embedding Pipeline Methods Documentation  
**Project NU historical newspaper analysis**

## Purpose of this document
This document describes the data construction and embedding procedure used to
produce state-year bias scores from historical newspaper text. It is written as
a methods note.
The pipeline combines article text from the American Stories dataset with
newspaper-level geographic metadata from Chronicling America. It then creates
localized A La Carte embeddings for dictionary concepts and computes cosine
similarity scores between social group concepts and attribute concepts.

**Primary output:**  
`state_year_bias_table_1963.csv`

**Primary columns:**
- State
- Year
- Diff bias score
- Bias score_group1
- Bias score_group2
- Bias concept

## 1. Data access

### 1.1 Article text source: American Stories
The article text came from the American Stories dataset hosted on Hugging Face:  
<https://huggingface.co/datasets/dell-research-harvard/AmericanStories>

American Stories is a structured historical U.S. newspaper corpus built from
public-domain Chronicling America scans. The Hugging Face dataset card describes
it as article-level text extracted from historical newspaper images using a
deep-learning pipeline for layout detection, legibility classification, OCR, and
association of article text across multiple bounding boxes. The dataset is
organized by year rather than by a standard train/test split.

The year-specific article records used in this pilot included fields such as:
- article_id
- newspaper_name
- edition
- date
- page
- headline
- byline
- article

Example raw American Stories record:
- article_id: `1_1870-01-01_p1_sn82014899_00211105483_1870010101_0773`
- newspaper_name: `The weekly Arizona miner.`
- edition: `01`
- date: `1870-01-01`
- page: `p1`
- headline: `[blank]`
- article: `PREyors 10 leaving San Francisco for Wash ington City...`

For the pilot work, year slices were pulled for 1963 and 1964. The 1963 file was
used as the main ALC embedding run. The 1964 file was retained as a smaller
pilot slice.

**Access pattern:**  
The intended reproducible access method is Hugging Face Datasets, using the
American Stories "subset_years" configuration and specifying the year list.
For example, in Python:

```python
from datasets import load_dataset

dataset = load_dataset(
    "dell-research-harvard/AmericanStories",
    "subset_years",
    year_list=["1963"]
)
```

**Practical access issues:**  
The full dataset is very large, so year-specific access was preferred over
downloading all years. The Hugging Face dataset viewer is disabled because the
dataset uses a loading script, so programmatic access through the Datasets
library is the practical route. The article-level data did not directly
provide all geographic fields needed for state, county, and city aggregation,
so it had to be merged with Chronicling America title metadata.

### 1.2 Newspaper metadata source: Chronicling America master list
The geographic metadata came from the Chronicling America digitized newspaper
title list, maintained by the Library of Congress:  
<https://www.loc.gov/collections/chronicling-america/titles/>

Chronicling America is a Library of Congress/NEH newspaper digitization
collection. The Library of Congress describes the collection as providing access
to selected digitized newspaper pages from the National Digital Newspaper
Program, with coverage through 1963:  
<https://www.loc.gov/collections/chronicling-america/about-this-collection/>

The local master metadata file used in this project is:  
`chronicling-america.csv`

**Master metadata columns:**
- Newspapers
- LCCN
- OCLC
- ISSN
- State
- County
- City
- Geo Location
- Browse Digitized Issues
- Number of Issues
- First Issue
- Last Issue
- Essay Available
- Languages
- Ethnicity

The key field used for merging was **LCCN**, the Library of Congress Control Number.
LCCN identifies the newspaper title and is embedded inside American Stories
`article_id` strings.

## 2. Data preprocessing

### 2.1 Extracting the LCCN from American Stories article_id
The American Stories `article_id` contains several underscore-separated
components. In the records used here, the LCCN appears as the fourth component.

**Example:**
- `article_id`: `1_1870-01-01_p1_sn82014899_00211105483_1870010101_0773`

**Parsed components:**
1. `1`
2. `1870-01-01`
3. `p1`
4. `sn82014899`
5. `00211105483`
6. `1870010101`
7. `0773`

**Extracted LCCN:** `sn82014899`

The extracted LCCN was saved as a standalone `LCCN` column in the article data.

### 2.2 Standardizing merge keys
Before merging, LCCN values in both datasets were converted to strings and
stripped of leading/trailing whitespace. This was necessary because a join key
with hidden whitespace or mixed type handling can silently reduce match rates.

**Standardization rule:**  
`LCCN = str(LCCN).strip()`

### 2.3 Merging article text with geographic metadata

**Source A:** American Stories article data for the target year.

**Source B:** Chronicling America master title metadata.

**Join key:** LCCN

**Join type:** Inner join.

**Reason for inner join:** The analysis requires a geographic grouping variable. Articles whose LCCN
could not be matched to a Chronicling America title with state/county/city
metadata were excluded from the analytical dataset. This means the final
merged file is the geographically matched sample, not necessarily every
article present in the original American Stories year slice.

**Final merged 1963 file:** `american_stories_1963_merged.csv`

**Final merged columns:**
- article_id
- date
- newspaper_name
- headline
- article
- LCCN
- State
- County
- City

### 2.4 Text normalization for embedding construction
The ALC script normalizes text before building context windows.

**Tokenizer:** NLTK `RegexpTokenizer` with pattern `[A-Za-z0-9]+`

**Main normalization decisions:**
1. Convert text to lowercase.
2. Tokenize into alphanumeric tokens.
3. Repair common OCR digit substitutions only when a token contains both
   alphabetic characters and digits.
4. Drop tokens that are not alphabetic after repair.
5. Remove English stopwords from NLTK.
6. Remove single-letter alphabetic tokens.
7. Preserve dictionary terms even when they overlap with stopwords.
8. Optionally apply fuzzy repair for OCR-damaged dictionary words. (For future work if text quality is bad)

**OCR digit repair map:**
- `0 -> o`
- `1 -> l`
- `3 -> e`
- `5 -> s`
- `7 -> t`

**Example:** If a token mixes letters and digits, `"g00d"` can be repaired toward `"good"`.
Pure numbers are not retained as analytic tokens.

**Optional fuzzy repair:**  
The script can fuzzy-repair OCR variants of dictionary terms when
`--enable-fuzzy` is passed. It uses NLTK edit distance and defaults to a maximum
edit distance of 1. Fuzzy repair is restricted to dictionary terms, which
limits the risk of globally rewriting historically meaningful spellings.

### 2.5 Dictionary construction
If no external dictionary JSON is supplied, the script uses built-in concept
dictionaries. The concepts are:
- BLACK
- WHITE
- RICH
- POOR
- MEN
- WOMEN
- POSITIVE
- NEGATIVE

The dictionaries include plural and gendered variants where relevant. Masked
racial slurs in the project prompt are expanded to their unmasked forms before
normalization so that historical newspaper occurrences can be counted. This is
important methodologically because the corpus contains historical language that
is offensive today but substantively relevant for measuring racialized semantic
associations in the period.

**Local cleaned dictionary sizes in the 1963 run:**
- BLACK: 17 terms
- WHITE: 14 terms
- RICH: 25 terms
- POOR: 11 terms
- MEN: 30 terms
- WOMEN: 33 terms
- POSITIVE: 12 terms
- NEGATIVE: 12 terms

The cleaned dictionary used by the run is stored in:  
`1963_csvdata/dictionaries_used.json`

## 3. Data size and coverage

### 3.1 Chronicling America master list coverage
The local master file `chronicling-america.csv` contains:
- 4,709 title rows
- 4,619 unique LCCNs
- 54 states/territories or state-like values
- 1,050 counties
- 1,581 cities
- 3,294,460 digitized issues summed across title rows

**Temporal coverage in the local master file:**
- Earliest first issue: 1736-09-03
- Latest first issue: 1963-03-14
- Earliest last issue: 1763-03-04
- Latest last issue: 1963-12-31

**Language coverage in the local master file:**
- English: 4,300 title rows
- German: 77 title rows
- English, Spanish: 53 title rows
- English, French: 36 title rows
- English, German: 33 title rows
- Spanish: 32 title rows
- English, Japanese: 25 title rows
- English, Italian: 14 title rows
- Polish: 13 title rows
- French: 13 title rows
- English, Polish: 11 title rows
- Italian: 8 title rows

**Ethnicity metadata in the local master file:**
- Blank/no ethnicity label: 3,992 title rows
- African American: 348 title rows
- German: 87 title rows
- Indians of North America: 39 title rows
- Latin American: 34 title rows
- Japanese: 33 title rows
- Jewish: 22 title rows
- Polish: 21 title rows
- Italian: 17 title rows
- French: 14 title rows
- Czech: 13 title rows
- Norwegian: 9 title rows

### 3.2 Merged 1963 analytical sample
The main analysis file `american_stories_1963_merged.csv` contains:
- 547,783 article rows
- 15 states
- 31 counties
- 38 cities
- 42 unique LCCNs
- 61 newspaper-name strings
- Date range: 1963-01-01 to 1963-12-31
- Empty article fields: 0
- Approximate whitespace-token count: 69,012,773
- Average article length: 125.99 whitespace tokens

**Top states by article count:**
- District of Columbia: 317,669
- Mississippi: 40,477
- Maryland: 33,040
- Alaska: 32,728
- Delaware: 27,286
- Virginia: 20,358
- Minnesota: 19,244
- Montana: 17,620
- North Carolina: 12,200
- Arizona: 11,567

**Top counties by article count:**
- Sussex: 20,603
- Prince Edward: 20,358
- Montgomery: 18,814
- Holmes: 17,700
- Garrett: 14,226
- Hinds: 12,734
- Nome: 10,962
- Columbus: 10,951
- Richland: 10,162
- Ramsey: 10,098

**Top cities by article count:**
- Washington: 317,669
- Milford: 20,603
- Farmville: 20,358
- Rockville, Gaithersburg: 18,814
- Lexington: 17,700
- Oakland: 14,226
- Jackson: 12,734
- Nome: 10,962
- Tabor City: 10,951
- Sidney: 10,162

**Monthly article counts in 1963:**
- January: 47,957
- February: 44,348
- March: 48,400
- April: 45,038
- May: 50,204
- June: 44,525
- July: 40,722
- August: 44,052
- September: 43,308
- October: 46,926
- November: 46,662
- December: 45,641

The final 1963 state-year bias table contains:
- 120 rows
- 15 states
- 8 bias contrasts per state

### 3.3 Merged 1964 pilot sample
The 1964 pilot file `american_stories_1964_merged (1).csv` contains:
- 2,678 article rows
- 1 state: Arizona
- 1 county: Maricopa
- 1 city: Phoenix
- 1 unique LCCN
- Date range: 1964-01-03 to 1964-12-31
- Approximate whitespace-token count: 252,360
- Average article length: 94.23 whitespace tokens

Because the 1964 pilot sample is geographically narrow, it should be treated as
a test of the data and code path rather than as a broad state-comparative sample.

## 4. ALC embedding creation approach

### 4.1 Conceptual overview
The A La Carte method creates embeddings for words as they are used in a
specific corpus or subcorpus. Instead of relying only on a static pretrained
embedding for a word, the method estimates a word vector from the contexts in
which that word appears.

In this project, the pretrained base space is English fastText. The historical
newspaper corpus supplies the contexts. The ALC transformation matrix maps an
average context vector into the pretrained embedding space.

This is useful for historical newspaper analysis because it allows the same
dictionary term, such as "black", "white", "men", "women", "rich", or "poor",
to be represented by how it appears in a particular year and geographic
subcorpus, rather than only by its pretrained meaning (Training data of model).

### 4.2 Base embedding model
The script accepts a fastText model path through:
`--fasttext`

**Supported formats:**
- `.bin`
- `.vec`
- `.txt`
- `.kv`

**Preferred format:** `.bin`

**Reason:** The `.bin` fastText model supports subword inference for out-of-vocabulary
forms, which is valuable for OCR-damaged and historically variant text. Plain
`.vec` or `.txt` vectors can be used, but they do not provide true fastText
subword OOV behavior.

The local 1963 ALC artifacts indicate a 300-dimensional embedding space:
- `A_1963_news.npy` shape: 300 × 300
- `global_alc_vectors_1963.npz` vectors shape: 57,729 × 300

### 4.3 Context window construction
**Default context window size:** 5 tokens on each side of the target word

For every target occurrence in a tokenized article, the script collects up to
five tokens to the left and five tokens to the right, excluding the target token
itself.

Example with window size 5:
```
context(target_i) =
  tokens from i-5 through i-1 plus tokens from i+1 through i+5
```

The context embedding for one occurrence is the average of all available
fastText vectors for the surrounding context tokens. Tokens that cannot be
mapped to a base vector are skipped. If no usable context token remains, that
target occurrence is skipped.

**Rationale for window size 5:** A five-token window is a standard local semantic window. It is wide enough to
capture nearby adjectives, predicates, and noun modifiers but narrow enough to
keep the representation tied to the immediate phrase/sentence context. For
bias measurement, this helps target local evaluative associations rather than
general article topic.

### 4.4 Training the global ALC transformation matrix
The transformation matrix is trained once for the full year corpus.

**Main parameters:**
- `--window-size`: 5
- `--min-count`: 20
- `--alpha-k`: 100.0
- `--max-regression-words`: 100,000
- `--max-global-alc-words`: 200,000

**Step 1:** Count normalized tokens globally.  
The script scans the full merged year CSV and counts all normalized tokens.

**Step 2:** Select eligible words.  
A word is eligible if it appears at least `--min-count` times. For the 1963 run,
the default threshold was 20. This removes rare words whose context averages
would be unstable.

**Step 3:** Build global context sums.  
For each eligible word, the script averages the context vectors around each
occurrence and accumulates those averages across the full year corpus.

**Step 4:** Select regression vocabulary.  
The regression vocabulary includes high-frequency eligible words that:
  a. have observed context vectors in the year corpus, and
  b. exist in the base fastText vocabulary.

**Step 5:** Fit the weighted linear map.  
Let **X** be the matrix of observed average context vectors for regression words.
Let **Y** be the matrix of pretrained fastText vectors for the same words.

The script solves:
```
minimize_A sum_i w_i || y_i - A x_i ||^2
```

In implementation, it solves the row-major least-squares problem:
```
Y ≈ X B
```
Then it stores:
```
A = B^T
```

A transformed context vector is computed as:
```
transformed_vector = context_vector A^T
```

**Step 6:** Frequency weighting.  
Each regression word receives a weight:
```
w_i = min(1.0, context_count_i / alpha_k)
```
With `alpha_k = 100.0`, words with at least 100 observed contexts receive full
weight, while lower-frequency words receive proportionally smaller weight.
This reduces the influence of unstable low-frequency estimates without
removing all moderately frequent terms.

**Step 7:** Normalize vectors.  
Transformed vectors are L2-normalized. This makes cosine similarity equivalent
to the dot product of normalized vectors and keeps all concept vectors on a
comparable scale.

### 4.5 Global and local ALC artifacts
The 1963 output folder is: `1963_csvdata`

**Available local artifacts:**
- `A_1963_news.npy`  
  300 × 300 float32 ALC transformation matrix.

- `global_alc_vectors_1963.npz`  
  Contains 57,729 global ALC vectors of dimension 300, plus vocabulary and
  counts. The local copy examined here contains `vocab`, `counts`, and `vectors`.
  It does not contain the `matrix` field expected by the newest version of the
  script, so the separate `A_1963_news.npy` file is the reusable matrix artifact
  for this run.

- `dictionaries_used.json`  
  Cleaned concept dictionaries used by the 1963 pipeline.

- `state_year_bias_table_1963.csv`  
  Final state-level bias table.

### 4.6 Localized state-level embeddings
After the global matrix **A** is trained or loaded, the script estimates localized
embeddings for dictionary words by state.

For each state:
1. Read articles whose `State` value equals that state.
2. Tokenize and normalize each article.
3. For each occurrence of a dictionary target word, collect its context window.
4. Average the fastText vectors of the context tokens.
5. Average all occurrence-level context vectors for the same target word in the same state.
6. Apply the global ALC matrix **A** to the state-specific context vector.
7. L2-normalize the resulting localized word vector.

This produces state-specific embeddings for dictionary words. For example, the
word "women" receives a vector based on the contexts in which "women" appears
in Alabama newspapers in 1963, and a separate vector based on the contexts in
which "women" appears in Mississippi newspapers in 1963.

### 4.7 Concept vectors
The pipeline groups dictionary words into concepts. Within each state, a concept
vector is the average of the available localized word vectors for that concept.

**Example:**
BLACK concept vector for a state =
mean of localized vectors for *black, african, africans, blacks, colored,
negro, negros*, and other retained BLACK dictionary terms that appeared with
usable contexts in that state.

After averaging, each concept vector is L2-normalized.

This produces one vector per available concept per state:
- `BLACK_state`
- `WHITE_state`
- `RICH_state`
- `POOR_state`
- `MEN_state`
- `WOMEN_state`
- `POSITIVE_state`
- `NEGATIVE_state`

## 5. Cosine similarity and bias-score computation

### 5.1 Cosine similarity
Cosine similarity measures the angle between two vectors. It is commonly used in
embedding analysis because it measures semantic proximity independent of vector
magnitude.

For vectors **a** and **b**:
```
cosine(a, b) = (a · b) / (||a|| ||b||)
```

Because the ALC vectors are L2-normalized, cosine similarity is primarily
interpretable as directional similarity in the embedding space.

### 5.2 Bias contrasts
The script computes eight contrasts:
- Black-negative / White-negative
- Black-positive / White-positive
- Black-rich / White-rich
- Black-poor / White-poor
- Men-positive / Women-positive
- Men-negative / Women-negative
- Rich-positive / Poor-positive
- Rich-negative / Poor-negative

Each label has:
- group 1 concept
- group 2 concept
- attribute concept

**Example:**
- Bias concept: `Black-negative/White-negative`
- group 1: BLACK
- group 2: WHITE
- attribute: NEGATIVE

### 5.3 Bias score formula
For each state, year, and contrast:
```
Bias score_group1 = cosine(group1 concept vector, attribute concept vector)
Bias score_group2 = cosine(group2 concept vector, attribute concept vector)
Diff bias score   = Bias score_group1 - Bias score_group2
```

**Example for Black-negative/White-negative:**
```
Bias score_group1 = cosine(BLACK_state, NEGATIVE_state)
Bias score_group2 = cosine(WHITE_state, NEGATIVE_state)
Diff bias score   = cosine(BLACK_state, NEGATIVE_state) - cosine(WHITE_state, NEGATIVE_state)
```

**Interpretation:** A positive difference means the group 1 concept is more semantically similar
to the attribute concept than the group 2 concept is, within that state's 1963
newspaper language. A negative difference means the reverse.

**Important interpretive caution:** These scores are distributional semantic associations, not direct measures of
individual attitudes. They reflect patterns in published newspaper language,
shaped by editorial choices, OCR quality, newspaper availability, geographic
coverage, and the historical language of the period.

## 6. Reproducible pipeline sequence
A typical 1963 run follows this order:

1. Pull American Stories article-level data for 1963 from Hugging Face.
2. Extract LCCN from `article_id`.
3. Load Chronicling America master metadata from `chronicling-america.csv`.
4. Standardize LCCN fields in both datasets.
5. Inner join article records to master metadata on LCCN.
6. Save the merged file with article text and geographic fields.
7. Load the merged file into the ALC yearly pipeline.
8. Load the English fastText base embedding model.
9. Load or build the concept dictionaries.
10. Normalize article text and count global tokens.
11. Select eligible global words with count ≥ 20.
12. Build global context vectors using a five-token window.
13. Fit the weighted ALC transformation matrix **A**.
14. Save the matrix and global ALC vectors.
15. Build state-specific local context vectors for dictionary terms.
16. Transform local context vectors with **A**.
17. Average localized word vectors into concept vectors.
18. Compute cosine similarities for all bias contrasts.
19. Save the final state-year bias table.

**Example command for fitting a new yearly ALC model:**

```bash
python alc_year_pipeline.py ^
  --csv american_stories_1963_merged.csv ^
  --year 1963 ^
  --fasttext cc.en.300.bin ^
  --out-dir 1963_csvdata ^
  --window-size 5 ^
  --min-count 20
```

**Example command for reusing an existing matrix:**

```bash
python alc_year_pipeline.py ^
  --csv american_stories_1963_merged.csv ^
  --year 1963 ^
  --fasttext cc.en.300.bin ^
  --matrix 1963_csvdata/A_1963_news.npy ^
  --out-dir 1963_csvdata
```

## 7. Limitations and reporting notes

1. **Geographic coverage is uneven.**  
   The 1963 merged sample is heavily weighted toward the District of Columbia.
   State comparisons should report article counts and should avoid interpreting
   states with very different corpus sizes as equally measured.

2. **The merged file is an inner-joined sample.**  
   Articles without an LCCN match in the Chronicling America master metadata are
   excluded. The analytical sample is therefore the subset of American Stories
   article records with usable newspaper-title geography.

3. **OCR noise remains substantively important.**  
   American Stories improves on raw OCR by using a structured extraction
   pipeline, but historical newspaper OCR remains noisy. The pipeline includes
   digit repair and optional fuzzy repair for dictionary words, but it does not
   fully correct all OCR errors.

4. **Dictionary methods are transparent but bounded.**  
   The results depend on the selected dictionary terms. The dictionaries should
   be reported in an appendix or supplement, and robustness checks can test
   whether results change when controversial or ambiguous terms are removed.

5. **Newspaper language is not equivalent to public opinion.**  
   The scores capture semantic associations in available newspaper text. They
   should be interpreted as media-language associations, not direct individual-
   level prejudice or endorsement.

6. **Historical language contains offensive terms.**  
   Some dictionary terms are offensive today. They are included because they
   occur in historical sources and are relevant to measuring racialized language.
   Any presentation of these terms should contextualize them carefully.

## 8. Suggested citation and source links

**American Stories dataset:**  
Dell, Melissa, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora,
Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin, and Leander
Heldring. 2023. *American Stories: A Large-Scale Structured Text Dataset of
Historical U.S. Newspapers.* arXiv:2308.12477.  
<https://huggingface.co/datasets/dell-research-harvard/AmericanStories>

**Chronicling America:**  
Library of Congress. *Chronicling America: Historic American Newspapers.*  
<https://www.loc.gov/collections/chronicling-america/about-this-collection/>  
<https://www.loc.gov/collections/chronicling-america/titles/>

**FastText:**  
The pipeline uses English fastText vectors as the base embedding space. The
exact file path and version should be reported from the run configuration,
for example `cc.en.300.bin` if that was the model used.
