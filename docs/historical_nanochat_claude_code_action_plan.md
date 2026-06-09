# Historical Nanochat: Immediately Actionable Claude Code Implementation Plan

Created: 2026-04-19

This file is intended to be pasted directly into Claude Code as an implementation brief. It focuses on concrete repo changes, open data ingestion, metadata preservation, OCR triage, and proof-of-concept training readiness. The companion files are:

- `historical_nanochat_access_emails.md` — email/outreach templates and request-access notes.
- `historical_nanochat_next_steps.md` — later roadmap after the first proof of concept.

---

## 0. Mission

Modify `AshitaOrbis/historical-nanochat` into a reproducible, provenance-preserving corpus and training pipeline for time-locked historical language models.

Immediate goal:

1. Build a corpus ingestion framework for free/open public-domain or research-accessible historical sources.
2. Create a rights/date/metadata audit layer that fails closed.
3. Add an OCR triage and GPT-5.4 OCR-correction queue, but do not require OpenAI access for baseline ingestion.
4. Produce historical parquet shards usable by nanochat training on a single RTX 3090 and, later, rented GPUs.
5. Preserve enough source metadata to support source-grounded synthetic data generation and downstream evaluation.

Do **not** scrape sources whose terms prohibit automated access. For request-access sources, create registry entries, documentation stubs, and email templates only.

---

## 1. High-level repo changes

Add these directories/files:

```text
data/
  sources/
    registry.yaml
    README.md
  harvesters/
    __init__.py
    base.py
    gutenberg.py
    internet_archive.py
    loc_selected_books.py
    chronicling_america.py
    american_stories.py
    british_library_books.py
    biodiversity_heritage_library.py
    old_bailey.py
    caselaw_access_project.py
    tcp.py
    ncse_v2.py
    papers_past.py
    delpher.py
    common_corpus_stub.py
  process/
    normalize_text.py
    metadata_normalize.py
    rights_audit.py
    date_audit.py
    dedupe.py
    ocr_quality.py
    ocr_triage.py
    ocr_batch_export.py
    ocr_batch_import.py
    pack_to_parquet.py
    manifests.py
  prompts/
    ocr_gpt54_contract.md
    source_grounded_synthetic_data_contract.md
  schemas/
    document.schema.json
    page.schema.json
    source.schema.json
    ocr_queue.schema.json
    training_record.schema.json
  scripts/
    harvest_source.py
    build_poc_corpus.py
    audit_corpus.py
    build_training_shards.py
    sample_corpus.py
    estimate_tokens.py
    make_ocr_queue.py
  tests/
    test_registry.py
    test_rights_audit.py
    test_date_audit.py
    test_metadata_schema.py
    test_dedupe.py
    test_pack_to_parquet.py
    fixtures/
```

Also update:

```text
README.md
nanochat/README.md
data/README.md
pyproject.toml or requirements.txt
```

Minimum new dependencies:

```text
pydantic>=2
pyyaml
orjson
pyarrow
pandas
polars
beautifulsoup4
lxml
requests
httpx
tqdm
rich
rapidfuzz
xxhash
ftfy
langdetect or fasttext language-id wrapper
python-dateutil
```

Optional dependencies guarded behind extras:

```text
internetarchive
warcio
datasets
huggingface_hub
boto3
pymupdf
pillow
imagehash
openai
```

---

## 2. Source registry schema

Create `data/sources/registry.yaml`. Every source must be registered before a harvester is allowed to write data.

Required fields:

```yaml
- source_id: string                 # stable snake_case id
  name: string
  homepage: string
  access_type: open | api_key | request_access | restricted | paid_skip | stub
  rights_summary: string
  license_or_rights_url: string
  date_range: string
  languages: [string]
  genres: [book | newspaper | periodical | legal | government | letter | science | early_modern | mixed]
  geographic_scope: [string]
  bulk_access_method: string
  expected_formats: [txt | xml | alto | hocr | pdf | image | parquet | json | rdf | marc]
  priority: p0 | p1 | p2 | later | skip
  needs_contact: boolean
  harvester: string | null
  notes: string
  verified_reference_urls: [string]
```

Add validation in `data/tests/test_registry.py`:

- all required fields exist;
- `source_id` is unique;
- `access_type` is one of the allowed values;
- no `request_access`, `restricted`, or `paid_skip` source has an active downloader;
- all `p0` open sources have either a working harvester or a documented stub with TODOs.

---

## 3. P0 sources to implement now

These are immediately actionable and should be implemented first. They are either open bulk sources or Hugging Face/open-data packages where automated access is normal.

### 3.1 Project Gutenberg

Registry entry:

```yaml
- source_id: gutenberg
  name: Project Gutenberg
  homepage: https://www.gutenberg.org/
  access_type: open
  rights_summary: Public-domain and permissioned ebooks; for this project, ingest only U.S. public-domain works and preserve PG rights notes.
  license_or_rights_url: https://www.gutenberg.org/policy/license.html
  date_range: mostly pre-1929 source works, but metadata does not reliably include original print date
  languages: [multi]
  genres: [book, mixed]
  geographic_scope: [global]
  bulk_access_method: Use Project Gutenberg mirrors and machine-readable RDF/CSV catalog; do not crawl the website.
  expected_formats: [txt, html, epub, rdf, marc, csv]
  priority: p0
  needs_contact: false
  harvester: gutenberg.py
  notes: Metadata does not reliably include original publication date. Treat publication date as weak unless matched to external metadata or title-page evidence.
  verified_reference_urls:
    - https://www.gutenberg.org/ebooks/offline_catalogs.html
    - https://www.gutenberg.org/cache/epub/feeds/
    - https://www.gutenberg.org/MIRRORS.ALL
```

Implementation notes:

- Use `pg_catalog.csv.gz` or `rdf-files.tar.bz2` as catalog input.
- Prefer bulk text tar if available; otherwise use a configured mirror, not the interactive website.
- Extract canonical plain text when present.
- Strip PG header/footer only with tested regexes; store raw and cleaned text paths separately.
- Do not trust PG release date as original publication date. Set:

```json
"publication_date": null,
"date_confidence": "unknown_or_external_required",
"source_publication_date": "Project Gutenberg release date only"
```

- Include `needs_external_date_enrichment=true`.

CLI target:

```bash
python -m data.scripts.harvest_source --source gutenberg --out data/raw/gutenberg --limit 1000
```

---

### 3.2 Internet Archive public-domain texts

Registry entry:

```yaml
- source_id: internet_archive_pd
  name: Internet Archive public-domain texts
  homepage: https://archive.org/details/texts
  access_type: open
  rights_summary: Mixed rights at collection scale; ingest only items with explicit public-domain/open-rights metadata and fail closed on unknown rights.
  license_or_rights_url: https://archive.org/about/terms.php
  date_range: multi-century, metadata varies
  languages: [multi]
  genres: [book, periodical, government, mixed]
  geographic_scope: [global]
  bulk_access_method: Metadata/API plus downloadable OCR/hOCR/PDF/images per item; use rate limits and cached manifests.
  expected_formats: [txt, hocr, pdf, image, json, xml]
  priority: p0
  needs_contact: false
  harvester: internet_archive.py
  notes: High volume and noisy metadata. Must filter rights, dates, mediatype, collection, language, and dedupe aggressively.
  verified_reference_urls:
    - https://blog.openlibrary.org/2008/11/24/bulk-access-to-ocr-for-1-million-books/
    - https://archive.org/developers/ocr.html
```

Implementation notes:

- Use the `internetarchive` Python package if installed, else HTTP API fallback.
- Search queries should be configurable in YAML, e.g.:

```yaml
internet_archive_pd_queries:
  - 'mediatype:texts AND rights:("Public Domain" OR "publicdomain") AND year:[1800 TO 1913]'
  - 'mediatype:texts AND collection:(americana) AND year:[1800 TO 1913]'
```

- Rights filter must fail closed. Accept only explicit public-domain/open signals in item metadata. Keep all rights fields in metadata.
- Download metadata first, then selected OCR files. Do not download PDFs/images unless OCR triage selects them.
- Preserve `_meta.xml`, `_files.xml`, OCR text, hOCR/ALTO if available, and item identifier.
- Add a `--manifest-only` mode to inspect candidate volume counts before download.

CLI target:

```bash
python -m data.scripts.harvest_source --source internet_archive_pd --out data/raw/internet_archive_pd --manifest-only
python -m data.scripts.harvest_source --source internet_archive_pd --out data/raw/internet_archive_pd --limit 5000
```

---

### 3.3 Library of Congress Selected Digitized Books

Registry entry:

```yaml
- source_id: loc_selected_books
  name: Library of Congress Selected Digitized Books Data Package
  homepage: https://data.labs.loc.gov/digitized-books/
  access_type: open
  rights_summary: LOC data package of OCR text and metadata; preserve LOC rights fields and fail closed for unclear rights.
  license_or_rights_url: https://data.labs.loc.gov/digitized-books/
  date_range: broad, filter to project cutoff
  languages: [multi]
  genres: [book]
  geographic_scope: [mostly_us]
  bulk_access_method: Download LOC data package metadata and full-text files.
  expected_formats: [txt, json, csv]
  priority: p0
  needs_contact: false
  harvester: loc_selected_books.py
  notes: Dataset contains 84,058 full-text files from 90,414 books; OCR generated as part of LOC digitization workflows.
  verified_reference_urls:
    - https://data.labs.loc.gov/digitized-books/
    - https://data.labs.loc.gov/digitized-books/README.html
```

Implementation notes:

- Implement direct data-package download.
- Parse LOC metadata.
- Keep LOC item URL, LCCN, title, contributors, date, subjects, language, locations, rights.
- Filter by date after date normalization.

CLI target:

```bash
python -m data.scripts.harvest_source --source loc_selected_books --out data/raw/loc_selected_books
```

---

### 3.4 Chronicling America bulk OCR

Registry entry:

```yaml
- source_id: chronicling_america
  name: Chronicling America bulk OCR
  homepage: https://chroniclingamerica.loc.gov/
  access_type: open
  rights_summary: Public-domain U.S. historical newspapers via Library of Congress/NEH program; preserve issue/page identifiers and rights statements.
  license_or_rights_url: https://chroniclingamerica.loc.gov/about/
  date_range: 1770s-1963, but project should filter by cutoff
  languages: [en, multi]
  genres: [newspaper]
  geographic_scope: [us]
  bulk_access_method: Bulk OCR files and LOC API; obey LOC rate limits.
  expected_formats: [txt, alto, json, pdf, image]
  priority: p0
  needs_contact: false
  harvester: chronicling_america.py
  notes: Raw page OCR has layout scrambling; prefer American Stories for article-level text when possible, but keep raw pages for OCR improvement and grounding.
  verified_reference_urls:
    - https://chroniclingamerica.loc.gov/ocr/
    - https://www.loc.gov/ndnp/migration/
    - https://libraryofcongress.github.io/data-exploration/loc.gov%20JSON%20API/Chronicling_America/README.html
```

Implementation notes:

- Implement two modes:
  - `bulk_ocr`: download/decompress listed OCR batches.
  - `api_manifest`: use LOC API for metadata enrichment and issue/page URLs.
- Obey LOC rate limits. The LOC API guide warns about burst/crawl limits; default to conservative rate limiting.
- Map OCR paths like `lccn/YYYY/MM/DD/ed-N/seq-N/ocr.txt` to canonical page URL.
- Keep date and newspaper title at page level.
- Mark as `granularity=page`.

CLI target:

```bash
python -m data.scripts.harvest_source --source chronicling_america --out data/raw/chronicling_america --states "CA,NY,MA" --years 1850:1913
```

---

### 3.5 American Stories

Registry entry:

```yaml
- source_id: american_stories
  name: American Stories historical U.S. newspaper article dataset
  homepage: https://dell-research-harvard.github.io/resources/americanstories
  access_type: open
  rights_summary: Derived from public-domain Chronicling America scans; preserve dataset license and LOC page provenance.
  license_or_rights_url: https://huggingface.co/datasets/dell-research-harvard/AmericanStories
  date_range: concentrated pre-1920
  languages: [en]
  genres: [newspaper]
  geographic_scope: [us]
  bulk_access_method: Hugging Face dataset / dataset package.
  expected_formats: [parquet, json]
  priority: p0
  needs_contact: false
  harvester: american_stories.py
  notes: Strong default newspaper source because it extracts structured article text from about 20M Chronicling America scans.
  verified_reference_urls:
    - https://dell-research-harvard.github.io/resources/americanstories
    - https://huggingface.co/datasets/dell-research-harvard/AmericanStories
    - https://arxiv.org/abs/2308.12477
```

Implementation notes:

- Use `datasets` or `huggingface_hub` if installed.
- Keep original fields; do not collapse article metadata.
- Filter date <= cutoff.
- Deduplicate against raw Chronicling America where needed; American Stories should normally supersede raw page OCR for article text.
- Mark `granularity=article` and `derived_from=chronicling_america`.

CLI target:

```bash
python -m data.scripts.harvest_source --source american_stories --out data/raw/american_stories --years 1850:1913
```

---

### 3.6 British Library 19th Century Digitised Books

Registry entry:

```yaml
- source_id: british_library_19c_books
  name: British Library 19th Century Digitised Books / BL Books
  homepage: https://huggingface.co/datasets/TheBritishLibrary/blbooks
  access_type: open
  rights_summary: Out-of-copyright texts, public-domain marked; preserve BL dataset citation and item metadata.
  license_or_rights_url: https://huggingface.co/datasets/TheBritishLibrary/blbooks
  date_range: mostly 18th and 19th century, with some earlier material
  languages: [multi]
  genres: [book]
  geographic_scope: [uk, global]
  bulk_access_method: Hugging Face dataset; ALTO/OCR-derived JSONL variants also exist.
  expected_formats: [parquet, json, alto, txt]
  priority: p0
  needs_contact: false
  harvester: british_library_books.py
  notes: Direct competitor substrate for Victorian book models such as Mr. Chatterbox.
  verified_reference_urls:
    - https://huggingface.co/datasets/TheBritishLibrary/blbooks
    - https://github.com/davanstrien/digitised-books-ocr-and-metadata
```

Implementation notes:

- Ingest HF dataset first; optionally support `davanstrien/digitised-books-ocr-and-metadata` JSONL variant.
- Preserve BL identifier, title, author, date, language, publication place, subjects.
- Filter date <= cutoff and language according to target corpus.
- Run OCR quality scoring; BL OCR may have old typography and multi-column issues.

CLI target:

```bash
python -m data.scripts.harvest_source --source british_library_19c_books --out data/raw/bl_19c_books --years 1700:1913
```

---

### 3.7 Biodiversity Heritage Library

Registry entry:

```yaml
- source_id: biodiversity_heritage_library
  name: Biodiversity Heritage Library
  homepage: https://www.biodiversitylibrary.org/
  access_type: open
  rights_summary: Open access biodiversity literature; preserve BHL license/rights fields per item/page.
  license_or_rights_url: https://about.biodiversitylibrary.org/tools-and-services/developer-and-data-tools/
  date_range: 15th-21st centuries, filter by cutoff
  languages: [multi]
  genres: [science, book, periodical]
  geographic_scope: [global]
  bulk_access_method: AWS Open Data, APIs, OCR/full-text exports.
  expected_formats: [txt, json, xml, image]
  priority: p0
  needs_contact: false
  harvester: biodiversity_heritage_library.py
  notes: High-value scientific/natural-history language source; strong for period scientific vocabulary.
  verified_reference_urls:
    - https://about.biodiversitylibrary.org/tools-and-services/developer-and-data-tools/
    - https://registry.opendata.aws/bhl-open-data/
```

Implementation notes:

- Implement AWS Open Data manifest support first; no need to download images by default.
- Ingest title/item/page metadata and OCR text.
- Preserve page IDs for source grounding.
- Filter by publication date <= cutoff.
- Tag `genre=science` and domain taxonomy if available.

CLI target:

```bash
python -m data.scripts.harvest_source --source biodiversity_heritage_library --out data/raw/bhl --years 1700:1913 --text-only
```

---

### 3.8 Old Bailey Online

Registry entry:

```yaml
- source_id: old_bailey
  name: Old Bailey Online Proceedings
  homepage: https://www.oldbaileyonline.org/
  access_type: open
  rights_summary: XML data available for research use; preserve Old Bailey citation and terms.
  license_or_rights_url: https://www.oldbaileyonline.org/about/data
  date_range: 1674-1913
  languages: [en]
  genres: [legal]
  geographic_scope: [uk, london]
  bulk_access_method: XML data download/API/R package-compatible endpoint.
  expected_formats: [xml, json]
  priority: p0
  needs_contact: false
  harvester: old_bailey.py
  notes: Excellent structured source for ordinary lives, law, crime, testimony, gender/class, and social history. Treat sensitive content carefully.
  verified_reference_urls:
    - https://www.oldbaileyonline.org/about/data
    - https://www.oldbaileyonline.org/about/homepage
```

Implementation notes:

- Parse XML sessions/trials.
- Preserve fields: trial date, offence, verdict, punishment, defendant/victim names when present, gender, age, occupation, session id, trial id.
- Emit both trial-level text and structured metadata.
- Add `sensitive_content=true` and genre tags.

CLI target:

```bash
python -m data.scripts.harvest_source --source old_bailey --out data/raw/old_bailey
```

---

### 3.9 Caselaw Access Project

Registry entry:

```yaml
- source_id: caselaw_access_project
  name: Caselaw Access Project
  homepage: https://case.law/
  access_type: open
  rights_summary: Free public access and bulk/API routes; preserve CAP license/metadata and jurisdiction details.
  license_or_rights_url: https://case.law/
  date_range: ~360 years of U.S. case law
  languages: [en]
  genres: [legal]
  geographic_scope: [us]
  bulk_access_method: CAP API/bulk download and mirrored datasets.
  expected_formats: [json, txt]
  priority: p0
  needs_contact: false
  harvester: caselaw_access_project.py
  notes: Large, structured, date-rich legal corpus. Useful for source-grounded QA and long-form formal prose.
  verified_reference_urls:
    - https://case.law/
    - https://lil.law.harvard.edu/our-work/caselaw-access-project/
    - https://huggingface.co/datasets/free-law/Caselaw_Access_Project
```

Implementation notes:

- Start with the Hugging Face/free-law mirror or CAP API if bulk is available.
- Preserve citation, decision date, court, jurisdiction, reporter, casebody data, frontend URL, CAP ID.
- Filter to date <= cutoff.
- Add jurisdiction tags.

CLI target:

```bash
python -m data.scripts.harvest_source --source caselaw_access_project --out data/raw/cap --years 1750:1913
```

---

### 3.10 Text Creation Partnership / early modern corpora

Registry entry:

```yaml
- source_id: tcp_public_domain
  name: EEBO/ECCO/Evans Text Creation Partnership public-domain corpora
  homepage: https://ota.bodleian.ox.ac.uk/
  access_type: open
  rights_summary: Public-domain TCP subsets via Oxford Text Archive / University of Michigan; preserve attribution.
  license_or_rights_url: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/5
  date_range: mostly 1475-1800 depending on subset
  languages: [en]
  genres: [early_modern, book, pamphlet, mixed]
  geographic_scope: [uk, us]
  bulk_access_method: OTA/UMich XML downloads.
  expected_formats: [xml, html, txt]
  priority: p0
  needs_contact: false
  harvester: tcp.py
  notes: High-quality keyed text, not OCR. Use lower weight for 1900/1913 models unless early modern depth is desired.
  verified_reference_urls:
    - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/5
    - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/7
    - https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/8
    - https://quod.lib.umich.edu/e/evans
    - https://earlyprint.org/download/
```

Implementation notes:

- Support EEBO-TCP Phase I first.
- Then ECCO-TCP and Evans-TCP.
- Normalize TEI/XML to plain text while preserving document structure.
- Keep publication year/date and short-title metadata.
- Tag as `source_quality=keyed_text`.

CLI target:

```bash
python -m data.scripts.harvest_source --source tcp_public_domain --out data/raw/tcp --subsets eebo_phase1,ecco,evans
```

---

### 3.11 NCSE v2.0

Registry entry:

```yaml
- source_id: ncse_v2
  name: NCSE v2.0 19th-century English newspapers/periodicals
  homepage: https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610
  access_type: open
  rights_summary: Freely available research dataset; preserve dataset citation and rights statement.
  license_or_rights_url: https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610
  date_range: 19th century
  languages: [en]
  genres: [newspaper, periodical]
  geographic_scope: [uk]
  bulk_access_method: Download dataset zip/parquet from UCL repository.
  expected_formats: [parquet, json]
  priority: p0
  needs_contact: false
  harvester: ncse_v2.py
  notes: Important precedent for image-to-text LLM OCR; contains 82,690 pages, 1.4M entries, and 321M words.
  verified_reference_urls:
    - https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610
    - https://arxiv.org/abs/2502.14901
```

Implementation notes:

- Download parquet files, preserve periodical, issue, page, block type/topic fields.
- Use as high-quality periodical corpus and OCR benchmark reference.
- Tag `ocr_engine=image_to_text_llm_pixtral_source_dataset` if described in dataset metadata.

CLI target:

```bash
python -m data.scripts.harvest_source --source ncse_v2 --out data/raw/ncse_v2
```

---

## 4. P1 sources: registry stubs now, implementation later or after access

Add these to the registry with no active downloader unless explicitly open and terms permit automated access.

### HathiTrust public-domain full text

```yaml
- source_id: hathitrust_pd_requested
  name: HathiTrust public-domain full-view text
  homepage: https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/
  access_type: request_access
  rights_summary: Public-domain full-view text can be downloaded in bulk for non-commercial research after approval.
  license_or_rights_url: https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/
  date_range: broad
  languages: [multi]
  genres: [book, periodical, government, mixed]
  geographic_scope: [global]
  bulk_access_method: Research dataset approval process.
  expected_formats: [txt, json, xml]
  priority: p1
  needs_contact: true
  harvester: null
  notes: Add stub only. Email support@hathitrust.org after first PoC.
  verified_reference_urls:
    - https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/
    - https://www.hathitrust.org/contact/
```

### Harvard Library Public Domain Corpus

```yaml
- source_id: harvard_public_domain_corpus_requested
  name: Harvard Library Public Domain Corpus
  homepage: https://library.harvard.edu/services-tools/harvard-library-public-domain-corpus
  access_type: request_access
  rights_summary: Approximately one million digitized public-domain books; request access for research/teaching/learning/creative activities.
  license_or_rights_url: https://library.harvard.edu/services-tools/harvard-library-public-domain-corpus
  date_range: broad
  languages: [multi]
  genres: [book]
  geographic_scope: [global]
  bulk_access_method: Request access through Harvard Library.
  expected_formats: [txt, json, metadata]
  priority: p1
  needs_contact: true
  harvester: null
  notes: Potentially the biggest high-value missed source. Add only docs/stub until access granted.
  verified_reference_urls:
    - https://library.harvard.edu/services-tools/harvard-library-public-domain-corpus
    - https://library.harvard.edu/services-tools/harvard-library-apis-datasets
```

### Canadiana / CRKN

```yaml
- source_id: canadiana_crkn_requested
  name: Canadiana / CRKN
  homepage: https://www.canadiana.ca/
  access_type: request_access
  rights_summary: Canadian historical books, periodicals, government publications; automated access should be requested, not scraped blindly.
  license_or_rights_url: https://www.canadiana.ca/
  date_range: broad
  languages: [en, fr, indigenous_languages, multi]
  genres: [book, newspaper, periodical, government, mixed]
  geographic_scope: [canada]
  bulk_access_method: Request controlled bulk OCR/metadata access.
  expected_formats: [txt, xml, pdf, image, metadata]
  priority: p1
  needs_contact: true
  harvester: null
  notes: Add stub only; ask permission before automated access.
  verified_reference_urls:
    - https://www.canadiana.ca/
```

### JSTOR Text Analysis / Data for Research

```yaml
- source_id: jstor_text_analysis_requested
  name: JSTOR Text Analysis / full-text datasets
  homepage: https://support.jstor.org/hc/en-us/articles/32479181127575-JSTOR-Text-Analysis-Support-Getting-Started
  access_type: request_access
  rights_summary: Full-text datasets can be requested for text analysis; process requires item IDs from JSTOR metadata.
  license_or_rights_url: https://support.jstor.org/hc/en-us/articles/32487330092695-JSTOR-Text-Analysis-Support-Working-with-JSTOR-Full-Text-Datasets
  date_range: broad, filter to public-domain/early journal content as terms allow
  languages: [multi]
  genres: [journal, science, periodical]
  geographic_scope: [global]
  bulk_access_method: Dataset request through JSTOR text analysis support.
  expected_formats: [txt, metadata]
  priority: p1
  needs_contact: true
  harvester: null
  notes: Do not ingest until access terms are clear.
  verified_reference_urls:
    - https://support.jstor.org/hc/en-us/articles/32479181127575-JSTOR-Text-Analysis-Support-Getting-Started
    - https://support.jstor.org/hc/en-us/articles/32487330092695-JSTOR-Text-Analysis-Support-Working-with-JSTOR-Full-Text-Datasets
```

### Trove

```yaml
- source_id: trove_requested
  name: Trove, National Library of Australia
  homepage: https://trove.nla.gov.au/
  access_type: api_key
  rights_summary: API/bulk access requires account, API key, and terms agreement; preserve rights per item.
  license_or_rights_url: https://trove.nla.gov.au/about/create-something/using-api
  date_range: broad, filter to public-domain/cutoff
  languages: [en]
  genres: [newspaper, book, periodical, mixed]
  geographic_scope: [australia]
  bulk_access_method: API key and bulk-download workflows.
  expected_formats: [txt, json, pdf, image]
  priority: p1
  needs_contact: false
  harvester: null
  notes: Add harvester only after API key configured locally. Do not hardcode credentials.
  verified_reference_urls:
    - https://trove.nla.gov.au/about/create-something/bulk-download
    - https://trove.nla.gov.au/about/create-something/using-api
```

### Papers Past

```yaml
- source_id: papers_past
  name: Papers Past newspaper open data
  homepage: https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot
  access_type: open
  rights_summary: Open data for historic New Zealand newspapers; includes METS/ALTO XML, not images.
  license_or_rights_url: https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot
  date_range: published more than 120 years ago in open data pilot
  languages: [en]
  genres: [newspaper]
  geographic_scope: [new_zealand]
  bulk_access_method: Download individual titles or full open data package.
  expected_formats: [mets, alto, xml]
  priority: p1
  needs_contact: false
  harvester: papers_past.py
  notes: Good first non-U.S. newspaper source. Implement after P0.
  verified_reference_urls:
    - https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot
    - https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot/data-standards-papers-past-newspaper-open-data-pilot
```

### Delpher open newspaper archive

```yaml
- source_id: delpher_open_newspapers
  name: Delpher open newspapers
  homepage: https://www.kb.nl/en/research-find/datasets/delpher-newspapers
  access_type: open
  rights_summary: OCR/ALTO/XML newspapers from 1618-1879 are free of copyright and may be used without restrictions per KB page.
  license_or_rights_url: https://www.kb.nl/en/research-find/datasets/delpher-newspapers
  date_range: 1618-1879 open set; later material requires conditions/license
  languages: [nl]
  genres: [newspaper]
  geographic_scope: [netherlands]
  bulk_access_method: 111GB split zip archive.
  expected_formats: [txt, alto, xml]
  priority: p1
  needs_contact: false
  harvester: delpher.py
  notes: Implement after English P0. Strictly avoid post-1879 restricted material unless licensed.
  verified_reference_urls:
    - https://www.kb.nl/en/research-find/datasets/delpher-newspapers
```

### Gallica / BnF

```yaml
- source_id: gallica_bnf
  name: Gallica / BnF APIs and datasets
  homepage: https://www.bnf.fr/en/gallica-bnf-digital-library
  access_type: open
  rights_summary: Free access digital library with APIs/datasets; rights vary per item and must be preserved.
  license_or_rights_url: https://www.bnf.fr/en/api-portal-and-datasets
  date_range: broad
  languages: [fr, multi]
  genres: [book, newspaper, periodical, mixed]
  geographic_scope: [france, global]
  bulk_access_method: BnF API portal and downloadable datasets.
  expected_formats: [txt, xml, image, metadata]
  priority: p1
  needs_contact: false
  harvester: null
  notes: Add discovery/manifest stub first; implement after rights model is robust.
  verified_reference_urls:
    - https://www.bnf.fr/en/gallica-bnf-digital-library
    - https://www.bnf.fr/en/api-portal-and-datasets
```

### Common Corpus / OpenCulture

```yaml
- source_id: common_corpus_openculture_stub
  name: Common Corpus / OpenCulture public-domain aggregate
  homepage: https://huggingface.co/collections/PleIAs/openculture
  access_type: open
  rights_summary: Aggregated public-domain/open datasets; use for bootstrapping or comparison, but prefer source-native ingestion for provenance.
  license_or_rights_url: https://huggingface.co/collections/PleIAs/openculture
  date_range: broad
  languages: [multi]
  genres: [book, newspaper, periodical, mixed]
  geographic_scope: [global]
  bulk_access_method: Hugging Face parquet datasets.
  expected_formats: [parquet]
  priority: p2
  needs_contact: false
  harvester: common_corpus_stub.py
  notes: Do not rely on this as primary source because source/cutoff provenance may be lossy. Use as comparison or gap-fill only.
  verified_reference_urls:
    - https://arxiv.org/html/2506.01732v1
    - https://huggingface.co/collections/PleIAs/openculture
```

---

## 5. Unified document/page schema

Create `data/schemas/document.schema.json` and implement matching Pydantic models.

Minimum document record:

```json
{
  "record_type": "document",
  "source_id": "gutenberg",
  "document_id": "source-stable-id",
  "parent_id": null,
  "page_id": null,
  "chunk_id": null,
  "title": "...",
  "subtitle": null,
  "author": "...",
  "contributors": [],
  "publication_date": "1894-03-12",
  "publication_year": 1894,
  "date_confidence": "exact|year|range|inferred|weak|unknown",
  "language": "en",
  "language_confidence": 0.99,
  "country_or_region": "US",
  "publication_place": "New York",
  "publisher": "...",
  "genre": "book|newspaper|periodical|legal|government|letter|science|early_modern|mixed",
  "subgenre": null,
  "rights": "...",
  "rights_url": "...",
  "source_url": "...",
  "citation": "...",
  "raw_text_path": "...",
  "clean_text_path": "...",
  "image_path": null,
  "ocr_engine": "source_ocr|tesseract|gpt5_4|pixtral_source_dataset|keyed_text|unknown",
  "ocr_quality_estimate": 0.0,
  "source_quality": "keyed_text|source_ocr|corrected_ocr|unknown",
  "content_hash": "xxh3/raw",
  "dedupe_hash": "normalized-simhash-or-minhash",
  "word_count": 0,
  "char_count": 0,
  "token_estimate": 0,
  "cutoff_bucket": "pre1850|1850_1875|1875_1900|1900_1913|post_cutoff_exclude|unknown_exclude",
  "sensitive_content": false,
  "metadata": {}
}
```

Minimum page/article/trial/case child record:

```json
{
  "record_type": "segment",
  "source_id": "chronicling_america",
  "document_id": "newspaper-issue-id",
  "page_id": "seq-1",
  "segment_id": "article-or-block-id",
  "granularity": "page|article|trial|case|chapter|entry",
  "title": "...",
  "publication_date": "1903-05-01",
  "date_confidence": "exact",
  "text": "...",
  "source_url": "...",
  "bbox": null,
  "metadata": {}
}
```

---

## 6. Rights audit: fail closed

Implement `data/process/rights_audit.py`.

Rules:

1. Default action for unknown rights: **exclude**.
2. Default action for missing date: **exclude from training**, but allow metadata-only quarantine.
3. Default action for post-cutoff date: **exclude**.
4. Only include public-domain/open-rights records when source registry says it is allowed.
5. For mixed-rights sources, require explicit item-level rights evidence.
6. Write audit report JSON and Markdown.

Output categories:

```text
include_train
include_eval_only
include_metadata_only
quarantine_unknown_rights
quarantine_unknown_date
exclude_post_cutoff
exclude_restricted
exclude_paid_or_no_access
```

CLI:

```bash
python -m data.process.rights_audit \
  --registry data/sources/registry.yaml \
  --metadata data/intermediate/all_metadata.jsonl \
  --cutoff 1913-12-31 \
  --out data/audits/rights_1913/
```

Acceptance tests:

- Unknown rights are excluded.
- Unknown dates are excluded unless `--allow-unknown-date eval_only` is set.
- `request_access` sources cannot be included without `access_granted=true` in a local, uncommitted config file.
- P0 open records with explicit public-domain rights pass.

---

## 7. Date audit and cutoff logic

Implement `data/process/date_audit.py`.

Date confidence hierarchy:

```text
exact date > year > range > inferred from bibliographic field > weak text guess > unknown
```

Cutoff rules:

- If exact/year date <= cutoff: include if rights pass.
- If date range ends <= cutoff: include.
- If date range crosses cutoff: exclude unless segment-level date is known pre-cutoff.
- If only source ingestion/download date exists: unknown, exclude.
- Project Gutenberg release date is not source publication date; do not use it as cutoff date.
- Internet Archive `date` may be noisy; label confidence and preserve raw field.

Add tests for:

```text
1850 cutoff
1875 cutoff
1900 cutoff
1913 cutoff
unknown date
range 1899-1915
PG release date only
newspaper exact issue date
```

---

## 8. Text normalization

Implement `data/process/normalize_text.py`.

The objective is not to modernize history. It is to remove machine/dataset boilerplate and repair obvious encoding/OCR artifacts.

Rules:

- Preserve historical spelling.
- Preserve period punctuation where possible.
- Do not modernize words.
- Do not silently expand abbreviations.
- Remove source boilerplate headers/footers only with source-specific tested logic.
- Normalize Unicode using NFKC only when safe; keep a raw text copy.
- Dehyphenate line endings only when confidence is high.
- Remove repeated page headers/footers using frequency heuristics.
- Tag any aggressive cleaning decisions in metadata.

CLI:

```bash
python -m data.process.normalize_text --in data/raw/... --out data/clean/...
```

---

## 9. Deduplication

Implement `data/process/dedupe.py`.

Levels:

1. Exact duplicate: `content_hash` on normalized text.
2. Near duplicate: SimHash/MinHash on 5-grams or 13-gram shingles.
3. Cross-source duplicates: IA/Hathi/Harvard/Gutenberg/BL overlap.
4. Newspaper reprints: keep for study but optionally downweight; do not always delete because reprints are historically meaningful.

Outputs:

```text
data/dedupe/clusters.jsonl
data/dedupe/drop_list.jsonl
data/dedupe/keep_list.jsonl
data/audits/dedupe_report.md
```

Policy:

- For books, keep highest-quality version by priority:
  1. keyed text
  2. high-quality corrected OCR
  3. source OCR with good score
  4. low-quality OCR
- For newspapers, preserve source occurrences but optionally mark reprint clusters.

---

## 10. OCR quality scoring and triage

Implement `data/process/ocr_quality.py` and `data/process/ocr_triage.py`.

Quality features:

```text
char_count
word_count
mean_line_length
line_count
alphabetic_ratio
digit_ratio
punctuation_ratio
garbage_symbol_ratio
replacement_char_count
long_token_ratio
dictionary_hit_rate
language_id_confidence
repeated_header_footer_score
hyphenation_score
column_scramble_score
unicode_weirdness_score
ocr_confidence_if_available
```

Output:

```json
{
  "source_id": "chronicling_america",
  "document_id": "...",
  "page_id": "...",
  "source_url": "...",
  "image_url": "...",
  "original_ocr_path": "...",
  "score": 0.42,
  "priority": "high|medium|low",
  "reason_codes": ["low_dictionary_rate", "column_scramble", "rare_source"],
  "recommended_action": "keep_source_ocr|send_to_gpt54|send_to_local_ocr|discard"
}
```

Triage priority formula:

```text
priority = historical_value * quality_problem * rights_ok * date_confidence * source_diversity_bonus
```

High-value examples:

- pre-1850 newspapers;
- rare regional newspapers;
- non-U.S./non-UK coverage;
- legal/government documents with tables;
- scientific pages with names/tables;
- pages where source OCR is clearly broken but images exist;
- serial fiction and periodical essays.

CLI:

```bash
python -m data.scripts.make_ocr_queue \
  --metadata data/intermediate/all_metadata.jsonl \
  --cutoff 1913-12-31 \
  --max-pages 1000 \
  --out data/ocr/queue/gpt54_pilot_1000.jsonl
```

---

## 11. GPT-5.4 OCR correction contract

Create `data/prompts/ocr_gpt54_contract.md`:

```markdown
# OCR correction contract for historical pages

You are transcribing a historical printed page for a public-domain corpus.

Goals:
- Produce faithful OCR/transcription.
- Preserve historical spelling, punctuation, capitalization, dialect, and typography as much as possible.
- Preserve reading order and block/column structure.
- Mark uncertainty instead of guessing.
- Do not modernize language.
- Do not summarize.
- Do not add facts.

Input:
- Page image.
- Existing OCR text, if available.
- Source metadata: title, date, publication, page id, URL.

Output valid JSON only:
{
  "page_id": "...",
  "transcription": "...",
  "blocks": [
    {
      "block_id": "b1",
      "type": "article|headline|advertisement|table|caption|footer|unknown",
      "text": "...",
      "reading_order": 1,
      "uncertain": false,
      "notes": ""
    }
  ],
  "unreadable_spans": [
    {"text": "[illegible]", "reason": "blurred|cutoff|ink|fold|unknown"}
  ],
  "layout_notes": "",
  "correction_notes": "",
  "confidence": 0.0
}

Hard constraints:
- If text is illegible, write [illegible] rather than inventing it.
- Do not silently correct historical spellings.
- Preserve tables using Markdown table syntax inside the text field if possible.
- Preserve line breaks only when meaningful for poetry, tables, ads, headings, or layout.
- For normal prose, paragraphs are acceptable.
```

Implement batch export/import:

```bash
python -m data.process.ocr_batch_export \
  --queue data/ocr/queue/gpt54_pilot_1000.jsonl \
  --out data/ocr/batches/batch_0001.jsonl

python -m data.process.ocr_batch_import \
  --batch-results data/ocr/results/batch_0001_results.jsonl \
  --out data/ocr/corrected/
```

Do not require OpenAI credentials unless running the actual OCR worker. The repo should produce queue files offline.

Reference docs to include in comments/README:

- OpenAI GPT-5.4 document understanding tips: https://developers.openai.com/cookbook/examples/multimodal/document_and_multimodal_understanding_tips
- OpenAI GPT-5.4 model guide: https://developers.openai.com/api/docs/guides/latest-model
- OpenAI Batch API guide: https://developers.openai.com/api/docs/guides/batch
- OpenAI pricing: https://developers.openai.com/api/docs/pricing

---

## 12. Corpus packing to parquet

Implement `data/process/pack_to_parquet.py`.

Training record schema:

```json
{
  "text": "...",
  "source_id": "...",
  "document_id": "...",
  "segment_id": "...",
  "title": "...",
  "author": "...",
  "publication_date": "...",
  "publication_year": 1894,
  "date_confidence": "exact|year|range|inferred|weak",
  "language": "en",
  "genre": "newspaper",
  "rights": "...",
  "source_url": "...",
  "source_quality": "source_ocr|corrected_ocr|keyed_text",
  "ocr_quality_estimate": 0.87,
  "token_estimate": 1234,
  "content_hash": "...",
  "dedupe_cluster_id": "...",
  "weight_hint": 1.0
}
```

Shard output:

```text
data/processed/corpus_1913_v0/
  manifest.json
  train/
    shard_000000.parquet
    shard_000001.parquet
  val/
    shard_000000.parquet
  eval/
    cutoff_eval.jsonl
    anachronism_eval.jsonl
    source_grounded_eval.jsonl
  reports/
    source_mix.md
    date_distribution.md
    rights_audit.md
    ocr_quality.md
    dedupe.md
```

CLI:

```bash
python -m data.scripts.build_training_shards \
  --registry data/sources/registry.yaml \
  --clean-dir data/clean \
  --cutoff 1913-12-31 \
  --languages en \
  --out data/processed/corpus_1913_v0 \
  --target-shard-mb 256 \
  --val-fraction 0.005 \
  --eval-holdout-sources old_bailey,bhl,chronicling_america
```

Shard packer must stream. Do not load all texts into RAM. Use bounded shuffle buffer or external shuffle.

---

## 13. Proof-of-concept corpus recipe

Add `data/scripts/build_poc_corpus.py` that can build a small corpus quickly.

Recommended POC source mix:

```yaml
poc_1913_v0:
  cutoff: "1913-12-31"
  language: "en"
  sources:
    gutenberg:
      max_documents: 5000
      include_if_date_known_only: true
    loc_selected_books:
      max_documents: 10000
    british_library_19c_books:
      max_documents: 10000
    american_stories:
      max_articles: 1000000
    chronicling_america:
      max_pages: 100000
      use_only_if_not_in_american_stories: true
    old_bailey:
      max_documents: all
    biodiversity_heritage_library:
      max_documents: 10000
    caselaw_access_project:
      max_documents: 100000
    tcp_public_domain:
      max_documents: all
    ncse_v2:
      max_documents: all
```

CLI:

```bash
python -m data.scripts.build_poc_corpus --config data/sources/poc_1913_v0.yaml
```

Output should include:

- total documents;
- total segments;
- estimated tokens;
- source mix;
- date distribution;
- rights exclusions;
- unknown-date exclusions;
- OCR quality histogram;
- dedupe clusters;
- training shard paths.

---

## 14. Nanochat training integration

Once parquet exists, connect it to nanochat training.

Requirements:

1. Add explicit `--parquet-dir` or `NANOCHAT_PARQUET_DIR` support.
2. Do not rely on FineWeb defaults for historical runs.
3. Add 3090 scripts:

```text
nanochat/historical_3090_base.sh
nanochat/historical_3090_mid.sh
nanochat/historical_3090_eval.sh
```

4. Default 3090 run:

```text
single GPU
max_seq_len=1024
depth=14 or 16
activation checkpointing on
chunked loss on
compile safe/default or disabled fallback
checkpoint save/resume on
```

5. Preserve metadata columns for future source-conditioned sampling if nanochat dataloader can support them.

---

## 15. Minimal eval sets to create now

Create these as JSONL files under `data/processed/corpus_1913_v0/eval/`.

### 15.1 Cutoff/anachronism eval

Examples:

```json
{"cutoff":"1913-12-31","prompt":"What caused the First World War?","expected_behavior":"Should not describe events after cutoff as known facts; should state uncertainty or contemporary tensions only."}
{"cutoff":"1913-12-31","prompt":"Who won the 1918 influenza pandemic?","expected_behavior":"Should not know the 1918 pandemic."}
{"cutoff":"1900-12-31","prompt":"Explain powered flight by the Wright brothers.","expected_behavior":"Should not know 1903 flight as established fact."}
```

### 15.2 Source-grounded QA eval

Use held-out passages from Old Bailey, BHL, LOC books, American Stories.

```json
{
  "source_id":"old_bailey",
  "date":"1888-04-02",
  "passage":"...",
  "question":"What was the charge in this proceeding?",
  "answer":"...",
  "evidence_spans":["..."]
}
```

### 15.3 Style/source eval

- newspaper brief from dated article;
- Victorian book continuation;
- legal case summary;
- scientific abstract/description;
- period letter rewrite.

---

## 16. Acceptance criteria for this implementation phase

A phase is complete when all of these work:

```bash
python -m data.scripts.harvest_source --source gutenberg --out /tmp/hnc/gutenberg --limit 50
python -m data.scripts.harvest_source --source loc_selected_books --out /tmp/hnc/loc --limit 50
python -m data.scripts.harvest_source --source old_bailey --out /tmp/hnc/old_bailey --limit 50
python -m data.scripts.audit_corpus --in /tmp/hnc --cutoff 1913-12-31 --out /tmp/hnc_audit
python -m data.scripts.make_ocr_queue --metadata /tmp/hnc_audit/metadata.jsonl --max-pages 10 --out /tmp/hnc_ocr_queue.jsonl
python -m data.scripts.build_training_shards --clean-dir /tmp/hnc_audit/include_train --cutoff 1913-12-31 --out /tmp/hnc_shards
pytest data/tests
```

And the output includes:

```text
/tmp/hnc_shards/manifest.json
/tmp/hnc_shards/train/*.parquet
/tmp/hnc_shards/val/*.parquet
/tmp/hnc_shards/reports/source_mix.md
/tmp/hnc_shards/reports/rights_audit.md
```

---

## 17. Non-goals for this phase

Do not implement yet:

- full HathiTrust or Harvard ingestion before access is granted;
- paid data sources;
- restricted newspaper scraping;
- broad multilingual model training;
- OCR correction of millions of pages;
- large rented cluster scripts;
- FSDP/ZeRO/tensor parallel;
- synthetic-data generation at scale.

Do create stubs and documentation that make those next steps easy.

---

## 18. Useful source/reference links

Open now:

- Project Gutenberg offline catalog and RDF/CSV: https://www.gutenberg.org/ebooks/offline_catalogs.html
- Project Gutenberg feed files: https://www.gutenberg.org/cache/epub/feeds/
- Project Gutenberg mirrors: https://www.gutenberg.org/MIRRORS.ALL
- Internet Archive OCR docs: https://archive.org/developers/ocr.html
- Internet Archive bulk OCR context: https://blog.openlibrary.org/2008/11/24/bulk-access-to-ocr-for-1-million-books/
- LOC Selected Digitized Books: https://data.labs.loc.gov/digitized-books/
- Chronicling America OCR: https://chroniclingamerica.loc.gov/ocr/
- Chronicling America migration/API/bulk note: https://www.loc.gov/ndnp/migration/
- American Stories project: https://dell-research-harvard.github.io/resources/americanstories
- American Stories HF: https://huggingface.co/datasets/dell-research-harvard/AmericanStories
- American Stories paper: https://arxiv.org/abs/2308.12477
- British Library BL Books HF: https://huggingface.co/datasets/TheBritishLibrary/blbooks
- BL books JSONL/metadata repo: https://github.com/davanstrien/digitised-books-ocr-and-metadata
- BHL data tools: https://about.biodiversitylibrary.org/tools-and-services/developer-and-data-tools/
- BHL AWS Open Data: https://registry.opendata.aws/bhl-open-data/
- Old Bailey data: https://www.oldbaileyonline.org/about/data
- Caselaw Access Project: https://case.law/
- CAP LIL page: https://lil.law.harvard.edu/our-work/caselaw-access-project/
- CAP HF mirror: https://huggingface.co/datasets/free-law/Caselaw_Access_Project
- EEBO-TCP OTA: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/5
- ECCO-TCP OTA: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/7
- Evans-TCP OTA: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/8
- EarlyPrint downloads: https://earlyprint.org/download/
- NCSE v2.0 dataset: https://rdr.ucl.ac.uk/articles/dataset/NCSE_v2_0_A_Dataset_of_OCR-Processed_19th_Century_English_Newspapers/28381610

Request/access later:

- HathiTrust research datasets: https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/
- Harvard Public Domain Corpus: https://library.harvard.edu/services-tools/harvard-library-public-domain-corpus
- Harvard APIs/datasets: https://library.harvard.edu/services-tools/harvard-library-apis-datasets
- JSTOR Text Analysis getting started: https://support.jstor.org/hc/en-us/articles/32479181127575-JSTOR-Text-Analysis-Support-Getting-Started
- JSTOR full-text dataset requests: https://support.jstor.org/hc/en-us/articles/32487330092695-JSTOR-Text-Analysis-Support-Working-with-JSTOR-Full-Text-Datasets
- Trove bulk download: https://trove.nla.gov.au/about/create-something/bulk-download
- Trove API: https://trove.nla.gov.au/about/create-something/using-api
- Papers Past open data: https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot
- Delpher newspapers: https://www.kb.nl/en/research-find/datasets/delpher-newspapers
- BnF Gallica: https://www.bnf.fr/en/gallica-bnf-digital-library
- BnF API/datasets: https://www.bnf.fr/en/api-portal-and-datasets
- OpenAI GPT-5.4 document understanding tips: https://developers.openai.com/cookbook/examples/multimodal/document_and_multimodal_understanding_tips
- OpenAI GPT-5.4 model guide: https://developers.openai.com/api/docs/guides/latest-model
- OpenAI Batch API: https://developers.openai.com/api/docs/guides/batch
- OpenAI pricing: https://developers.openai.com/api/docs/pricing
