# Historical Nanochat: Communication and Email Pack

Created: 2026-04-19

This file contains outreach strategy, source-specific notes, and email templates for requesting access to data that should not be scraped blindly. Use this after the first proof-of-concept corpus/model exists, unless a request is low-friction and can be sent immediately.

Companion files:

- `historical_nanochat_claude_code_action_plan.md` — immediate coding-agent implementation spec.
- `historical_nanochat_next_steps.md` — later roadmap.

---

## 1. Outreach positioning

The strongest framing is:

> We are building a non-commercial, provenance-preserving, time-locked historical language model trained on public-domain or research-accessible pre-cutoff materials. The goal is to evaluate whether historical-only models can reduce anachronism and generate source-grounded synthetic data for smaller educational and research models.

Emphasize:

- non-commercial research;
- public-domain or clearly licensed material only;
- no redistribution of restricted full text unless explicitly permitted;
- source identifiers, dates, rights statements, and citations preserved;
- reproducible audit trail;
- willingness to share proof-of-concept results, metadata schema, and rights pipeline;
- controlled bulk access instead of scraping public websites;
- ability to comply with rate limits and access terms;
- source-grounded synthetic data, not hallucinated historical roleplay.

---

## 2. One-paragraph project summary for emails

Use this in most requests:

> I’m working on a non-commercial research project to build and evaluate time-locked historical language models trained only on public-domain or research-accessible materials before specified cutoff dates. The project preserves provenance, publication dates, rights statements, source identifiers, and OCR quality metadata for every document. The immediate research question is whether historically trained models can reduce anachronism and generate source-grounded synthetic data for smaller educational and research models. We are first building a proof of concept with open sources such as Project Gutenberg, Internet Archive public-domain OCR, Chronicling America/American Stories, Library of Congress datasets, Old Bailey, Biodiversity Heritage Library, and public-domain book collections.

---

## 3. Attachments/links to include after the first proof of concept

When available, attach or link:

1. GitHub repo.
2. One-page project overview.
3. Corpus metadata schema.
4. Rights/date audit policy.
5. Proof-of-concept model card.
6. Evaluation report: anachronism tests, source-grounded QA, held-out validation loss.
7. OCR correction sample, if requesting image/OCR access.
8. Statement that restricted source text will not be redistributed.

Suggested attachment titles:

```text
historical_nanochat_project_overview.pdf
historical_nanochat_corpus_governance.md
historical_nanochat_poc_results.md
historical_nanochat_metadata_schema.json
```

---

## 4. Request targets and current notes

### 4.1 HathiTrust public-domain full-view text

Why it matters:

- Major book corpus.
- HathiTrust says public-domain full-view text can be downloaded in bulk for non-commercial research after approval.
- Good target after PoC because it can greatly improve book coverage.

Reference URLs:

- https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/
- https://www.hathitrust.org/contact/

Contact:

- HathiTrust general support: `support@hathitrust.org`
- HathiTrust Research Center help may also be relevant: `htrc-help@hathitrust.org`

Template:

```text
Subject: Research request for public-domain full-text access for historical language-model corpus

Hello,

I’m working on a non-commercial research project to build and evaluate time-locked historical language models trained only on public-domain or research-accessible materials before specified cutoff dates. The goal is to study whether source-grounded historical models can reduce anachronism and generate higher-quality synthetic data for smaller educational and research models.

We are currently building a proof of concept using openly available sources such as Project Gutenberg, Internet Archive public-domain OCR, Library of Congress datasets, Chronicling America / American Stories, Old Bailey, Biodiversity Heritage Library, and other documented public datasets. For the next phase, we would like to request access to HathiTrust public-domain full-view text suitable for bulk non-commercial research use.

We would preserve source metadata, rights information, volume identifiers, publication dates, provenance, and audit logs. We are specifically interested in public-domain materials and would not redistribute restricted full text. We can provide our metadata schema, rights/date filtering policy, deduplication approach, and preliminary proof-of-concept results if helpful.

The immediate research questions are:

1. Can a model trained only on pre-cutoff historical public-domain material reduce anachronism compared with modern general models?
2. Can such a model generate source-grounded Q&A, summarization, and style data that improves smaller historical models?
3. How much do OCR quality, date filtering, and corpus provenance affect downstream historical fidelity?

Would you be able to advise on the appropriate process for obtaining public-domain full-text access for this research use?

Thank you,
[Name]
[Affiliation or independent researcher]
[Project page / GitHub / preliminary PoC link]
```

---

### 4.2 Harvard Library Public Domain Corpus

Why it matters:

- Harvard describes the corpus as approximately one million digitized public-domain books.
- It is explicitly relevant to research/teaching/learning/creative activities and AI training use according to Harvard’s APIs/datasets page.
- This may be the highest-value request for a serious run.

Reference URLs:

- https://library.harvard.edu/services-tools/harvard-library-public-domain-corpus
- https://library.harvard.edu/services-tools/harvard-library-apis-datasets

Template:

```text
Subject: Request for access to Harvard Library Public Domain Corpus for historical LM research

Hello,

I’m writing to request access to the Harvard Library Public Domain Corpus for a non-commercial historical language-model research project.

The project aims to train and evaluate time-locked language models using public-domain historical sources, with careful preservation of provenance, publication dates, rights metadata, and source identifiers. The first proof of concept uses open datasets such as Project Gutenberg, Internet Archive public-domain OCR, Chronicling America / American Stories, Library of Congress data, Old Bailey, Biodiversity Heritage Library, and public-domain newspaper/book collections.

The Harvard Public Domain Corpus appears especially valuable because of its scale, metadata, and public-domain status. We would use it to study:

- historical language modeling before specific cutoff dates;
- anachronism reduction in generated historical text;
- source-grounded synthetic data generation for smaller models;
- the effect of OCR quality, deduplication, and metadata filtering on model behavior.

We would not redistribute the corpus itself unless expressly permitted. Any released derived artifacts would preserve attribution/provenance and comply with applicable terms. We are happy to share the planned metadata schema, filtering pipeline, rights/date audit process, and evaluation design.

Could you let me know the appropriate access process and any conditions we should be aware of?

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

Shorter version:

```text
Subject: Harvard Public Domain Corpus access question

Hello,

I’m working on a non-commercial historical language-model project using public-domain pre-cutoff sources. The project preserves provenance, rights, source IDs, and publication-date metadata and evaluates whether historical-only models can reduce anachronism and generate source-grounded synthetic data for smaller research/education models.

Could you advise on access to the Harvard Library Public Domain Corpus and any terms relevant to computational training/evaluation use? I can share our proof-of-concept report, metadata schema, and rights audit plan.

Thank you,
[Name]
```

---

### 4.3 Canadiana / CRKN

Why it matters:

- Important for Canadian books, government documents, newspapers, periodicals, and bilingual English/French coverage.
- Particularly valuable because the corpus otherwise risks becoming U.S./UK-heavy.
- Do not scrape without permission. Ask for controlled bulk OCR/metadata access.

Reference URL:

- https://www.canadiana.ca/

Template:

```text
Subject: Permission request for controlled bulk access to public-domain Canadiana OCR/metadata

Hello,

I’m working on a non-commercial research project on historical language models and source-grounded synthetic data. We are building a dated, provenance-preserving corpus of public-domain or research-accessible historical texts for training and evaluation.

Canadiana is highly relevant because it contains Canadian books, newspapers, periodicals, government publications, and other historical materials that would substantially improve geographic coverage beyond U.S. and British sources. We noticed that automated access should not be assumed, so I’m writing to ask whether controlled bulk access to public-domain OCR and metadata would be possible for research use.

Our intended use is:

- ingest public-domain OCR and metadata;
- preserve source identifiers, rights statements, dates, titles, page/document provenance, and citation information;
- deduplicate against other public-domain collections;
- train and evaluate time-locked historical language models;
- generate source-grounded synthetic educational/research examples from public-domain passages.

We can comply with rate limits, use existing bulk/export mechanisms if available, avoid unnecessary load on public services, and share our processing plan before beginning. We would not redistribute restricted source text unless permitted.

Would CRKN/Canadiana be open to discussing access for this project, especially after we complete our first proof-of-concept model and evaluation report?

Thank you,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.4 JSTOR Text Analysis / Data for Research

Why it matters:

- Early journal content is high-value for science, social science, humanities, legal, and periodical language.
- JSTOR full-text analysis datasets are request-based and require item IDs from bibliographic metadata.
- Good after PoC once the requested set can be defined precisely.

Reference URLs:

- https://support.jstor.org/hc/en-us/articles/32479181127575-JSTOR-Text-Analysis-Support-Getting-Started
- https://support.jstor.org/hc/en-us/articles/32487330092695-JSTOR-Text-Analysis-Support-Working-with-JSTOR-Full-Text-Datasets
- https://support.jstor.org/hc/en-us/articles/32485272590487-JSTOR-Text-Analysis-Support-Working-with-JSTOR-Bibliographic-Metadata

Template:

```text
Subject: Text Analysis dataset request: early journal content for historical language-model evaluation

Hello,

I’m preparing a non-commercial research project on time-locked historical language models and source-grounded synthetic data. We are particularly interested in public-domain or otherwise research-accessible early journal content for model training and evaluation.

The project preserves provenance and rights metadata for every document and uses dated corpora to test whether historical models can reduce anachronism and generate higher-quality source-grounded synthetic data for smaller models. Early journal content would be valuable for scientific, literary, historical, legal, and social-science language that is underrepresented in book-only corpora.

We understand that full-text dataset requests require item IDs from JSTOR bibliographic metadata. Could you advise on the appropriate workflow for identifying and requesting a larger-scale early journal content dataset for text analysis, including OCR text and metadata where available?

We can provide a detailed project description, data-security plan, source-governance process, and information about intended derived outputs. We would not redistribute restricted full text unless expressly permitted.

Thank you for any guidance on the request process.

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.5 British Library historical newspaper data / Living with Machines guidance

Why it matters:

- British newspapers would greatly improve UK periodical coverage beyond books and NCSE v2.
- Some Living with Machines materials and metadata are open, but broader British Library newspaper OCR access can be complicated.
- Ask for guidance rather than assuming a route.

Reference URLs:

- https://livingwithmachines.ac.uk/lwm-digital-residency-accessing-and-using-historical-newspaper-data/
- https://bl.iro.bl.uk/collections/1ecde964-4860-4f66-af33-e2b8ba487bf9

Template:

```text
Subject: Research inquiry: historical newspaper OCR/text access for time-locked LM project

Hello,

I’m working on a non-commercial historical language-model project focused on pre-cutoff public-domain and research-accessible materials. The goal is to evaluate whether models trained on dated historical corpora can produce lower-anachronism, source-grounded synthetic data for smaller educational and research models.

We are already using open public-domain datasets such as Chronicling America, American Stories, Project Gutenberg, Internet Archive public-domain OCR, Old Bailey, Biodiversity Heritage Library, and public-domain book collections. British Library historical newspapers and related open datasets would be highly valuable for improving UK coverage, periodical language, and evaluation quality.

Could you advise whether there are bulk-access routes, open datasets, or research access programs for historical newspaper OCR/text suitable for non-commercial computational analysis? We would preserve all source metadata, rights statements, dates, titles, and page identifiers, and would not redistribute restricted full text unless permitted.

We are happy to share our proof-of-concept results and corpus-governance plan before requesting larger access.

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.6 Trove / National Library of Australia

Why it matters:

- Australian newspapers, books, magazines, gazettes.
- Great geographic diversification.
- Trove bulk/API access requires account, API key, and agreement to API terms.

Reference URLs:

- https://trove.nla.gov.au/about/create-something/bulk-download
- https://trove.nla.gov.au/about/create-something/using-api

Template:

```text
Subject: Trove API/bulk access for public-domain historical language-model research

Hello,

I’m working on a non-commercial research project that builds and evaluates time-locked historical language models from public-domain or research-accessible pre-cutoff sources. The project preserves source identifiers, rights statements, publication dates, page/article provenance, and OCR quality metadata.

Trove’s newspapers, gazettes, books, and periodicals would be extremely valuable for adding Australian coverage and reducing U.S./UK bias in the corpus. We plan to use the Trove API and/or bulk-download functionality in accordance with Trove API terms, rate limits, and item-level rights metadata.

Could you advise whether there are recommended practices for larger-scale public-domain OCR/text harvesting for this type of non-commercial computational research? We are happy to provide a project description, metadata schema, and proof-of-concept results.

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.7 BnF / Gallica

Why it matters:

- Major French-language books, newspapers, periodicals, manuscripts, and images.
- Useful for a later multilingual or French-focused historical model.
- APIs/datasets exist; rights vary by item.

Reference URLs:

- https://www.bnf.fr/en/gallica-bnf-digital-library
- https://www.bnf.fr/en/api-portal-and-datasets

Template:

```text
Subject: Gallica/BnF API and dataset guidance for historical language-model research

Hello,

I’m working on a non-commercial historical language-model research project using public-domain or research-accessible pre-cutoff sources. The project preserves provenance, publication dates, rights metadata, source identifiers, and OCR quality information for every document.

Gallica/BnF materials would be valuable for French-language and multilingual historical modeling, especially books, press, and periodicals. We would like to understand the recommended API or dataset route for computational access to public-domain OCR/text and metadata, and any constraints we should observe for derived model training/evaluation.

We would preserve item-level rights and attribution metadata and would not redistribute restricted full text unless permitted.

Could you point us to the appropriate documentation or contact for this kind of research use?

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.8 National Library of New Zealand / Papers Past

Why it matters:

- Papers Past has open data with METS/ALTO XML for historical newspapers.
- This may not require email for open datasets, but email can be useful for best practices or larger use.

Reference URLs:

- https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot
- https://natlib.govt.nz/about-us/open-data/papers-past-metadata/papers-past-newspaper-open-data-pilot/data-standards-papers-past-newspaper-open-data-pilot

Template:

```text
Subject: Papers Past open newspaper data use in historical language-model research

Hello,

I’m working on a non-commercial historical language-model project using public-domain/open historical corpora with preserved provenance and rights metadata. Papers Past newspaper open data appears to be a strong source for New Zealand historical newspaper text through METS/ALTO XML.

We plan to ingest only open data, preserve source identifiers and rights statements, and cite Papers Past/National Library of New Zealand in our corpus documentation. The data would be used for historical language modeling, anachronism evaluation, and source-grounded synthetic-data generation.

Are there recommended citation, attribution, or rate/access practices we should follow for this use beyond the published open-data documentation?

Thank you,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

### 4.9 KB / Delpher

Why it matters:

- The open Delpher newspaper archive covers 1618-1879 OCR/ALTO/XML and is explicitly described as free of copyright and usable without restrictions.
- Later dates require conditions/licensing.
- Good to ask only if we want post-1879 material or clarification.

Reference URL:

- https://www.kb.nl/en/research-find/datasets/delpher-newspapers

Template for later-date access:

```text
Subject: Research inquiry: Delpher newspaper OCR access beyond open 1618-1879 archive

Hello,

I’m working on a non-commercial historical language-model project that preserves provenance, publication dates, rights metadata, and OCR quality information for every document. We are interested in using Delpher newspaper OCR/ALTO/XML data to improve Dutch-language and multilingual historical coverage.

We understand that the open Delpher newspaper archive covers 1618-1879 and that later newspaper material may require additional conditions or licensing. Could you advise whether research access is possible for post-1879 OCR/text for non-commercial computational analysis, and what conditions would apply?

We would preserve item-level rights and attribution metadata and would not redistribute restricted full text unless permitted.

Best regards,
[Name]
[Affiliation / independent researcher]
[Project link / PoC link]
```

---

## 5. Short follow-up email template

```text
Subject: Re: [original subject]

Hello,

I’m following up on the note below about public-domain/research-accessible historical text data for a non-commercial historical language-model project.

The short version is that we are trying to avoid scraping and instead use proper bulk/API/request channels with provenance, rights metadata, and source identifiers preserved. I’d be grateful for any guidance on the right access path or person to contact.

Thank you,
[Name]
```

---

## 6. “We have a PoC now” update email

```text
Subject: Proof-of-concept results for historical language-model data request

Hello,

I’m following up with a brief proof-of-concept update for the historical language-model project I wrote about earlier.

We have now completed an initial corpus/model experiment using open sources including [list sources]. The pipeline preserves source identifiers, rights statements, publication-date confidence, OCR quality estimates, deduplication clusters, and cutoff eligibility for every record. We also built an evaluation set for anachronism and source-grounded question answering.

The preliminary results suggest that [one-sentence result, e.g. “source-grounded historical data improves small-model performance on dated QA while reducing post-cutoff anachronisms compared with generic prompting”]. We are now preparing a larger, better-audited corpus and would like to include [their collection] if access terms allow.

Links:
- Project repo: [link]
- PoC report: [link]
- Corpus governance/metadata schema: [link]
- Model card or demo: [link]

Would it be possible to discuss controlled access to [specific data] for this next phase?

Thank you,
[Name]
```

---

## 7. Internal tracking table

Maintain this in a project issue or spreadsheet:

```markdown
| Source | Contact/status | Date contacted | Reply | Terms | Data requested | Access granted? | Next action |
|---|---|---:|---|---|---|---|---|
| HathiTrust | support@hathitrust.org |  |  |  | public-domain full-view text |  |  |
| Harvard PDC | request form/contact |  |  |  | public-domain corpus |  |  |
| Canadiana/CRKN | contact form/email |  |  |  | public-domain OCR + metadata |  |  |
| JSTOR | Text Analysis request |  |  |  | early journal item IDs/full text |  |  |
| BL newspapers | BL/LwM contact |  |  |  | OCR/text guidance |  |  |
| Trove | API account/key |  |  |  | public-domain OCR/text via API |  |  |
| BnF/Gallica | API/dataset guidance |  |  |  | public-domain French texts |  |  |
```

---

## 8. Things not to say

Avoid:

- “We want to scrape your site.”
- “We will train an AI on all your data regardless of terms.”
- “We will redistribute the corpus.”
- “We need everything.”
- “This is commercial.”
- “We don’t know how rights will be handled.”

Prefer:

- “controlled bulk access”;
- “public-domain or research-accessible materials”;
- “provenance-preserving”;
- “rights and date audit”;
- “non-commercial research”;
- “source-grounded synthetic data”;
- “we will not redistribute restricted full text unless expressly permitted.”

---

## 9. Useful boilerplate for project page

```text
Historical Nanochat is a research project exploring time-locked historical language models trained on public-domain and research-accessible sources before specified cutoff dates. The project focuses on provenance preservation, rights-aware corpus construction, OCR quality auditing, anachronism evaluation, and source-grounded synthetic-data generation. Rather than building a generic historical roleplay chatbot, the goal is to measure whether historically trained models can serve as reliable teachers for smaller models while staying grounded in dated primary sources.
```

---

## 10. Data-use pledge draft

```text
We will preserve source identifiers, rights statements, publication dates, citation information, and item/page provenance for every record. We will exclude unknown-rights and unknown-date records by default. We will not redistribute restricted full text unless explicitly permitted. We will document all filtering and deduplication decisions. Derived model releases will include corpus composition summaries and known limitations. Synthetic data generation will be source-grounded and will include evidence spans when possible.
```
