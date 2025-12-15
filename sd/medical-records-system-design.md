# System Design: Medical Records Processing & Analysis for Personal Injury Cases

## Table of Contents
1. [Problem Statement & Requirements](#1-problem-statement--requirements)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Deep-Dives](#3-component-deep-dives)
4. [Data Models](#4-data-models)
5. [API Design](#5-api-design)
6. [Scale & Performance](#6-scale--performance)
7. [Security & Compliance](#7-security--compliance)
8. [Failure Handling & Reliability](#8-failure-handling--reliability)
9. [Cost Optimization](#9-cost-optimization)
10. [Trade-offs & Alternatives](#10-trade-offs--alternatives)
11. [Interview Discussion Points](#11-interview-discussion-points)

---

## 1. Problem Statement & Requirements

### Context
Personal injury law firms handle thousands of cases involving medical records—often hundreds to thousands of pages per case. Attorneys need to:
- Understand a client's complete medical history
- Identify injuries, treatments, and recovery timelines
- Calculate damages (medical expenses, future care needs)
- Find inconsistencies or gaps that opposing counsel might exploit
- Create chronologies for demand letters, depositions, and trial

**Current Pain Point**: A paralegal manually reviewing 500 pages of medical records takes 8-20 hours. This is Eve Legal's core value proposition.

### Functional Requirements

| Requirement | Description |
|-------------|-------------|
| **Document Ingestion** | Accept PDFs, scanned images, faxes, DICOM files from various sources |
| **OCR & Text Extraction** | Convert scanned/image-based documents to searchable text |
| **Structure Recognition** | Identify document types (ER records, imaging reports, billing statements, etc.) |
| **Entity Extraction** | Extract dates, providers, diagnoses (ICD codes), procedures (CPT codes), medications, vitals |
| **Timeline Generation** | Create chronological view of all medical events |
| **Summarization** | Generate case summaries highlighting key injuries and treatments |
| **Gap Detection** | Identify missing records, treatment gaps, or inconsistencies |
| **Damages Calculation** | Aggregate medical expenses, project future costs |
| **Search & Query** | Natural language queries across case documents |
| **Export** | Generate formatted chronologies for demand letters/court filings |

### Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Latency** | Initial processing: < 5 min for 100-page document batch |
| **Throughput** | Handle 10,000+ document uploads per day across all tenants |
| **Accuracy** | > 95% entity extraction accuracy, < 2% hallucination rate |
| **Availability** | 99.9% uptime (legal deadlines are hard deadlines) |
| **Compliance** | HIPAA, SOC 2 Type II, state bar ethics rules |
| **Multi-tenancy** | Complete data isolation between law firms (attorney-client privilege) |
| **Auditability** | Full audit trail for all AI outputs (defensibility) |

### Capacity Estimation

```
Assumptions:
- 500 law firms
- Average 200 active cases per firm
- Average 300 pages of medical records per case
- 20% of cases receive new documents daily

Daily volume:
- Active cases: 500 × 200 = 100,000 cases
- Daily document uploads: 100,000 × 0.20 = 20,000 cases updated
- Pages processed daily: 20,000 × 50 pages (avg new docs) = 1,000,000 pages

Storage (annual):
- Raw documents: 1M pages × 200KB avg × 365 = ~73 TB/year
- Extracted text + metadata: ~15 TB/year
- Vector embeddings: ~5 TB/year
```

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Web App (React)  │  Mobile App  │  API Clients  │  Intake Voice Agent          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Authentication  │  Rate Limiting  │  Tenant Routing  │  Request Validation     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   DOCUMENT SERVICE   │  │   ANALYSIS SERVICE   │  │    QUERY SERVICE     │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ • Upload handling    │  │ • Entity extraction  │  │ • Natural language   │
│ • OCR orchestration  │  │ • Timeline building  │  │ • Semantic search    │
│ • Document parsing   │  │ • Summarization      │  │ • Chronology export  │
│ • Classification     │  │ • Gap detection      │  │ • Citation retrieval │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MESSAGE QUEUE (SQS/Kafka)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ocr-queue  │  extraction-queue  │  embedding-queue  │  analysis-queue          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│    OCR WORKERS       │  │  EXTRACTION WORKERS  │  │  EMBEDDING WORKERS   │
├──────────────────────┤  ├──────────────────────┤  ├──────────────────────┤
│ • Tesseract/Azure    │  │ • LLM extraction     │  │ • Text chunking      │
│ • Document AI        │  │ • NER models         │  │ • Vector generation  │
│ • Image enhancement  │  │ • Code normalization │  │ • Index updates      │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                          │
├─────────────────────┬─────────────────────┬─────────────────────────────────────┤
│   PostgreSQL        │   Vector DB         │   Object Storage                    │
│   (per-tenant       │   (Turbopuffer -    │   (S3 - encrypted)                  │
│    schemas)         │    per-case index)  │                                     │
├─────────────────────┼─────────────────────┼─────────────────────────────────────┤
│ • Case metadata     │ • Document chunks   │ • Raw PDFs                          │
│ • Extracted entities│ • Semantic search   │ • OCR outputs                       │
│ • Timelines         │ • Hybrid retrieval  │ • Generated reports                 │
│ • Audit logs        │                     │ • Backups                           │
└─────────────────────┴─────────────────────┴─────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  LLM APIs           │  Medical Code DBs   │  Case Management    │  Analytics    │
│  (OpenAI, Claude)   │  (ICD-10, CPT)      │  System Integrations│  (Metrics)    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep-Dives

### 3.1 Document Ingestion Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION FLOW                                │
└──────────────────────────────────────────────────────────────────────────────┘

  Upload Request                                              
       │                                                      
       ▼                                                      
┌──────────────┐     ┌──────────────┐     ┌──────────────┐   
│   Validate   │────▶│  Virus Scan  │────▶│   Encrypt    │   
│   (type,size)│     │  (ClamAV)    │     │  (AES-256)   │   
└──────────────┘     └──────────────┘     └──────────────┘   
                                                │            
                                                ▼            
┌──────────────┐     ┌──────────────┐     ┌──────────────┐   
│  Store Raw   │────▶│  Queue OCR   │────▶│   Return     │   
│  (S3+metadata)│    │   Task       │     │   Job ID     │   
└──────────────┘     └──────────────┘     └──────────────┘   
```

**Supported Input Formats:**
| Format | Source | Processing |
|--------|--------|------------|
| PDF (text-based) | Medical portals, EMR exports | Direct text extraction |
| PDF (scanned) | Faxes, old records | OCR pipeline |
| TIFF/JPEG | Faxed records | Image preprocessing + OCR |
| DICOM | Imaging centers | Metadata extraction + optional image analysis |
| HL7/FHIR | Hospital integrations | Direct structured parsing |

**Key Design Decisions:**

1. **Async Processing**: All heavy processing is async with job status polling
2. **Idempotency**: Upload requests include client-generated UUID to prevent duplicates
3. **Chunked Upload**: Large files (>10MB) use multipart upload to S3
4. **Early Validation**: Reject invalid files before expensive processing

### 3.2 OCR & Text Extraction

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OCR PIPELINE                                        │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Page       │────▶│  Image          │────▶│  Layout         │
│  Splitting  │     │  Enhancement    │     │  Detection      │
└─────────────┘     └─────────────────┘     └─────────────────┘
                           │                        │
                           ▼                        ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │  • Deskewing    │     │  • Tables       │
                    │  • Denoising    │     │  • Headers      │
                    │  • Contrast     │     │  • Paragraphs   │
                    │  • Binarization │     │  • Forms        │
                    └─────────────────┘     └─────────────────┘
                                                   │
                                                   ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  OCR Engine     │────▶│  Confidence     │────▶│  Post-          │
│  (Azure Doc AI) │     │  Scoring        │     │  Processing     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                       │
                                                       ▼
                                               ┌─────────────────┐
                                               │  • Spell check  │
                                               │  • Medical dict │
                                               │  • Structure    │
                                               └─────────────────┘
```

**OCR Strategy by Document Quality:**

| Quality Score | Strategy | Expected Accuracy |
|---------------|----------|-------------------|
| High (>0.9) | Direct extraction | 99%+ |
| Medium (0.7-0.9) | Standard OCR | 95-99% |
| Low (0.5-0.7) | Enhanced preprocessing + OCR | 90-95% |
| Very Low (<0.5) | Multiple OCR engines + voting | 85-90% |

**Medical-Specific Enhancements:**
- Custom dictionary with 50,000+ medical terms
- ICD-10 and CPT code pattern recognition
- Medication name fuzzy matching (Levenshtein + phonetic)
- Handwriting recognition for physician notes (Azure/Google specialized models)

### 3.3 Document Classification

```python
# Classification hierarchy for medical records
DOCUMENT_TYPES = {
    "clinical": {
        "emergency": ["er_report", "trauma_note", "triage_assessment"],
        "inpatient": ["admission_note", "discharge_summary", "progress_note", "operative_report"],
        "outpatient": ["office_visit", "consultation", "follow_up"],
        "diagnostic": ["lab_results", "imaging_report", "pathology_report", "ekg"],
    },
    "administrative": {
        "billing": ["itemized_bill", "insurance_claim", "eob", "superbill"],
        "records": ["face_sheet", "consent_form", "authorization", "medical_history"],
    },
    "therapeutic": {
        "physical_therapy": ["pt_evaluation", "pt_progress", "pt_discharge"],
        "mental_health": ["psych_evaluation", "therapy_notes", "treatment_plan"],
        "pharmacy": ["prescription", "medication_list", "pharmacy_record"],
    }
}
```

**Classification Pipeline:**

1. **Rule-Based Pre-Classification** (fast, cheap)
   - Header/footer pattern matching
   - Form field detection
   - Known provider templates

2. **ML Classification** (slower, more accurate)
   - Fine-tuned BERT model on 100K+ labeled medical documents
   - Multi-label classification (document can be multiple types)
   - Confidence threshold: 0.85 for auto-classification, else human review queue

3. **LLM Fallback** (expensive, highest accuracy)
   - For low-confidence cases
   - Structured prompt with document excerpt

### 3.4 Entity Extraction Engine

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ENTITY EXTRACTION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         RAW DOCUMENT TEXT       │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Rule     │   │   NER     │   │   LLM     │
            │  Engine   │   │  Models   │   │ Extraction│
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ Dates     │   │ Diagnoses │   │ Complex   │
            │ Codes     │   │ Procedures│   │ Relations │
            │ Providers │   │ Medications│  │ Causation │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │     ENTITY RECONCILIATION       │
                    │  • Deduplication                │
                    │  • Confidence scoring           │
                    │  • Conflict resolution          │
                    │  • Code normalization           │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      STRUCTURED OUTPUT          │
                    └─────────────────────────────────┘
```

**Entity Types & Extraction Methods:**

| Entity Type | Primary Method | Normalization |
|-------------|---------------|---------------|
| Dates | Regex + NER | ISO 8601 |
| ICD-10 Codes | Pattern matching | Code validation against CMS database |
| CPT Codes | Pattern matching | Code validation + description lookup |
| Medications | NER + RxNorm lookup | RxNorm CUI normalization |
| Providers | NER + NPI lookup | NPI validation |
| Body Parts | Medical NER | SNOMED CT mapping |
| Vitals | Rule-based | Unit standardization |
| Monetary Amounts | Regex | USD normalization |

**Extraction Prompt Template (for LLM):**

```
You are extracting structured medical information from a {document_type}.

Document text:
---
{document_chunk}
---

Extract the following entities in JSON format:
- dates: Array of {date, event_description, confidence}
- diagnoses: Array of {code, description, type: "primary"|"secondary", confidence}
- procedures: Array of {code, description, date, provider, confidence}
- medications: Array of {name, dosage, frequency, prescriber, confidence}
- providers: Array of {name, specialty, facility, npi_if_present}
- findings: Array of {finding, severity, body_part, confidence}

Rules:
1. Only extract information explicitly stated in the text
2. Include confidence score (0-1) for each extraction
3. If information is ambiguous, note it in a "notes" field
4. Do not infer or assume information not present
```

### 3.5 Timeline Generation

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        TIMELINE GENERATION                                    │
└──────────────────────────────────────────────────────────────────────────────┘

    Extracted Entities                     Timeline Output
    (from all documents)                   
           │                               ┌────────────────────────────┐
           ▼                               │  2023-01-15                │
    ┌─────────────┐                        │  ├─ ER Visit (Doc #12)     │
    │   Date      │                        │  │  • MVA, cervical strain │
    │  Clustering │                        │  │  • X-ray: negative      │
    └─────────────┘                        │  │  • Rx: Flexeril         │
           │                               │                            │
           ▼                               │  2023-01-22                │
    ┌─────────────┐                        │  ├─ PCP Follow-up (#15)    │
    │   Event     │                        │  │  • Continued neck pain  │
    │  Merging    │────────────────────────│  │  • Referred to PT       │
    └─────────────┘                        │                            │
           │                               │  2023-02-01 - 2023-04-15   │
           ▼                               │  ├─ PT Sessions (12x)      │
    ┌─────────────┐                        │  │  • ROM improved 60%     │
    │  Narrative  │                        │                            │
    │ Generation  │                        │  [GAP DETECTED: 45 days]   │
    └─────────────┘                        │                            │
           │                               │  2023-06-01                │
           ▼                               │  └─ Orthopedic Consult     │
    ┌─────────────┐                        │     • MRI recommended      │
    │   Gap       │                        └────────────────────────────┘
    │ Detection   │
    └─────────────┘
```

**Timeline Event Schema:**

```typescript
interface TimelineEvent {
  id: string;
  date: Date;
  date_precision: "day" | "week" | "month" | "approximate";
  event_type: "visit" | "procedure" | "diagnosis" | "medication" | "imaging" | "therapy";
  
  // Source tracking
  source_documents: Array<{
    document_id: string;
    page_numbers: number[];
    confidence: number;
  }>;
  
  // Event details
  provider: Provider;
  facility: Facility;
  diagnoses: Diagnosis[];
  procedures: Procedure[];
  medications: Medication[];
  findings: string[];
  
  // For damages calculation
  billed_amount?: number;
  paid_amount?: number;
  
  // AI-generated
  summary: string;
  significance: "routine" | "significant" | "critical";
  
  // Audit
  extraction_method: "rule" | "ner" | "llm";
  human_verified: boolean;
}
```

### 3.6 RAG Architecture for Medical Records

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        RAG QUERY FLOW                                         │
└──────────────────────────────────────────────────────────────────────────────┘

     User Query: "What treatment did the patient receive for their back injury?"
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │        QUERY PROCESSING         │
                    │  • Intent classification        │
                    │  • Medical term expansion       │
                    │  • Query rewriting              │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   VECTOR SEARCH     │         │   KEYWORD SEARCH    │
        │   (Turbopuffer)     │         │   (PostgreSQL FTS)  │
        │                     │         │                     │
        │ • Semantic matching │         │ • BM25 ranking      │
        │ • Per-case index    │         │ • ICD/CPT codes     │
        │ • Top-k retrieval   │         │ • Exact phrases     │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │      HYBRID RE-RANKING          │
                    │  • RRF (Reciprocal Rank Fusion) │
                    │  • Document type boosting       │
                    │  • Recency weighting            │
                    │  • Cross-encoder reranking      │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      CONTEXT ASSEMBLY           │
                    │  • Chunk expansion              │
                    │  • Section context inclusion    │
                    │  • Citation metadata            │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      LLM GENERATION             │
                    │  • Grounded response            │
                    │  • Inline citations             │
                    │  • Confidence indicators        │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │      CITATION VERIFICATION      │
                    │  • Source validation            │
                    │  • Hallucination detection      │
                    │  • Link generation              │
                    └─────────────────────────────────┘
```

**Chunking Strategy for Medical Documents:**

```python
class MedicalChunker:
    """
    Semantic-aware chunking for medical records.
    Standard chunking breaks context; medical records need special handling.
    """
    
    CHUNK_CONFIG = {
        "target_size": 512,  # tokens
        "overlap": 64,       # tokens
        "max_size": 1024,    # never exceed
    }
    
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        chunks = []
        
        # Strategy 1: Section-based chunking (preferred)
        if document.has_clear_sections:
            for section in document.sections:
                if section.token_count <= self.CHUNK_CONFIG["max_size"]:
                    chunks.append(self.create_chunk(section, "section"))
                else:
                    # Large section: split at paragraph boundaries
                    chunks.extend(self.split_section(section))
        
        # Strategy 2: Semantic boundary detection
        else:
            # Use sentence embeddings to find natural break points
            chunks = self.semantic_chunking(document.text)
        
        # Enrich each chunk with context
        for chunk in chunks:
            chunk.metadata = {
                "document_id": document.id,
                "document_type": document.type,
                "provider": document.provider,
                "date": document.date,
                "page_numbers": chunk.pages,
                "section_header": chunk.section_header,
                "preceding_context": self.get_preceding_context(chunk),
            }
        
        return chunks
```

**Why Per-Case Indexing (Turbopuffer architecture):**

```
Traditional approach:              Eve's approach:
┌────────────────────┐            ┌────────────────────┐
│   Global Index     │            │  Case-001 Index    │
│   (all tenants)    │            ├────────────────────┤
│                    │            │  Case-002 Index    │
│  • Query filters   │            ├────────────────────┤
│  • Tenant ID check │            │  Case-003 Index    │
│  • Risk of leakage │            ├────────────────────┤
│  • Hard to scale   │            │       ...          │
└────────────────────┘            └────────────────────┘

Benefits of per-case:
✓ Absolute data isolation (attorney-client privilege)
✓ No filter overhead at query time
✓ Easy to delete entire case (GDPR/right to forget)
✓ Natural sharding by case
✓ Sparse access patterns match serverless model
```

---

## 4. Data Models

### PostgreSQL Schema (per-tenant schema)

```sql
-- Each tenant (law firm) gets its own schema
CREATE SCHEMA tenant_acme_law;
SET search_path TO tenant_acme_law;

-- Core entities
CREATE TABLE cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(100),  -- Firm's case number
    client_name VARCHAR(255) NOT NULL,
    case_type VARCHAR(50) NOT NULL,  -- 'personal_injury', 'med_mal', etc.
    incident_date DATE,
    status VARCHAR(50) DEFAULT 'active',
    
    -- Computed/cached fields
    total_billed DECIMAL(12,2) DEFAULT 0,
    total_paid DECIMAL(12,2) DEFAULT 0,
    document_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT valid_case_type CHECK (case_type IN ('personal_injury', 'med_mal', 'workers_comp', 'premises_liability'))
);

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    
    -- Source tracking
    original_filename VARCHAR(500),
    storage_path VARCHAR(1000) NOT NULL,  -- S3 path
    file_hash VARCHAR(64) NOT NULL,  -- SHA-256 for dedup
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    page_count INTEGER,
    
    -- Classification
    document_type VARCHAR(100),
    document_subtype VARCHAR(100),
    classification_confidence FLOAT,
    
    -- Processing status
    processing_status VARCHAR(50) DEFAULT 'pending',
    ocr_completed_at TIMESTAMPTZ,
    extraction_completed_at TIMESTAMPTZ,
    embedding_completed_at TIMESTAMPTZ,
    
    -- Audit
    uploaded_by UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT unique_file_per_case UNIQUE (case_id, file_hash)
);

CREATE TABLE medical_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    
    -- Temporal
    event_date DATE NOT NULL,
    event_date_precision VARCHAR(20) DEFAULT 'day',
    event_end_date DATE,  -- For date ranges (e.g., PT from X to Y)
    
    -- Classification
    event_type VARCHAR(50) NOT NULL,
    significance VARCHAR(20) DEFAULT 'routine',
    
    -- Provider info
    provider_name VARCHAR(255),
    provider_npi VARCHAR(10),
    facility_name VARCHAR(255),
    
    -- Clinical content
    chief_complaint TEXT,
    diagnoses JSONB DEFAULT '[]',  -- Array of {code, description, type}
    procedures JSONB DEFAULT '[]',
    medications JSONB DEFAULT '[]',
    findings JSONB DEFAULT '[]',
    
    -- Financial
    billed_amount DECIMAL(10,2),
    paid_amount DECIMAL(10,2),
    
    -- Source tracking (critical for citations)
    source_documents JSONB NOT NULL,  -- [{doc_id, pages, confidence}]
    
    -- AI metadata
    extraction_method VARCHAR(20),
    confidence_score FLOAT,
    human_verified BOOLEAN DEFAULT FALSE,
    verified_by UUID,
    verified_at TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Extracted entities (normalized)
CREATE TABLE diagnoses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    event_id UUID REFERENCES medical_events(id) ON DELETE SET NULL,
    
    icd10_code VARCHAR(10),
    description TEXT NOT NULL,
    diagnosis_type VARCHAR(20),  -- 'primary', 'secondary', 'admitting'
    body_part VARCHAR(100),
    laterality VARCHAR(20),  -- 'left', 'right', 'bilateral'
    
    first_documented DATE,
    last_documented DATE,
    
    source_documents JSONB,
    confidence FLOAT,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Full-text search index
CREATE INDEX idx_documents_fts ON documents USING gin(to_tsvector('english', original_filename));
CREATE INDEX idx_events_date ON medical_events(case_id, event_date);
CREATE INDEX idx_diagnoses_code ON diagnoses(icd10_code);

-- Audit log (append-only)
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    actor_id UUID,
    actor_type VARCHAR(20),  -- 'user', 'system', 'ai'
    changes JSONB,
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Vector Store Schema (Turbopuffer)

```python
# Per-case namespace in Turbopuffer
# Namespace format: "tenant_{tenant_id}_case_{case_id}"

chunk_schema = {
    "id": "string",           # UUID
    "vector": "float[1536]",  # OpenAI embedding dimension
    
    # Filterable attributes
    "document_id": "string",
    "document_type": "string",
    "event_date": "string",   # ISO format for range queries
    "page_number": "int",
    
    # Stored but not indexed
    "text": "string",
    "section_header": "string",
    "preceding_context": "string",
}
```

---

## 5. API Design

### Document Upload

```yaml
POST /api/v1/cases/{case_id}/documents
Content-Type: multipart/form-data
Authorization: Bearer {token}

Request:
  - file: binary (required)
  - document_type: string (optional, for pre-classification)
  - source: string (optional) - "client_upload", "provider_portal", "fax"
  - metadata: object (optional)

Response: 201 Created
{
  "document_id": "doc_abc123",
  "status": "processing",
  "job_id": "job_xyz789",
  "estimated_completion": "2024-01-15T10:30:00Z",
  "_links": {
    "status": "/api/v1/jobs/job_xyz789",
    "document": "/api/v1/documents/doc_abc123"
  }
}
```

### Processing Status (Polling/Webhook)

```yaml
GET /api/v1/jobs/{job_id}

Response: 200 OK
{
  "job_id": "job_xyz789",
  "status": "completed",  # pending, processing, completed, failed
  "progress": {
    "ocr": "completed",
    "classification": "completed", 
    "extraction": "completed",
    "embedding": "completed"
  },
  "result": {
    "document_id": "doc_abc123",
    "page_count": 47,
    "document_type": "discharge_summary",
    "classification_confidence": 0.94,
    "events_extracted": 12,
    "entities_extracted": {
      "diagnoses": 5,
      "procedures": 3,
      "medications": 8
    }
  },
  "errors": [],
  "completed_at": "2024-01-15T10:28:00Z"
}
```

### Timeline Query

```yaml
GET /api/v1/cases/{case_id}/timeline
Authorization: Bearer {token}

Query Parameters:
  - start_date: ISO date (optional)
  - end_date: ISO date (optional)
  - event_types: comma-separated (optional)
  - include_gaps: boolean (default: true)
  - format: "json" | "markdown" | "docx"

Response: 200 OK
{
  "case_id": "case_123",
  "timeline": [
    {
      "date": "2023-01-15",
      "events": [
        {
          "id": "evt_001",
          "type": "emergency_visit",
          "provider": "Metro General ER",
          "summary": "MVA with cervical strain, negative imaging",
          "diagnoses": [
            {"code": "S13.4XXA", "description": "Sprain of cervical spine, initial"}
          ],
          "sources": [
            {"document_id": "doc_abc", "pages": [1, 2, 3], "confidence": 0.95}
          ]
        }
      ]
    },
    {
      "date": "2023-03-01",
      "gap": {
        "days": 45,
        "expected_events": ["follow_up", "physical_therapy"],
        "significance": "high"
      }
    }
  ],
  "summary": {
    "total_events": 24,
    "date_range": {"start": "2023-01-15", "end": "2023-06-01"},
    "gaps_detected": 2,
    "total_billed": 47500.00,
    "total_paid": 31200.00
  }
}
```

### Natural Language Query

```yaml
POST /api/v1/cases/{case_id}/query
Authorization: Bearer {token}

Request:
{
  "query": "What imaging studies were performed and what did they show?",
  "options": {
    "include_citations": true,
    "max_sources": 10,
    "response_format": "detailed"  # "brief", "detailed", "bullet_points"
  }
}

Response: 200 OK
{
  "answer": "The patient underwent several imaging studies following the accident:\n\n1. **Cervical X-ray (01/15/2023)** at Metro General ER showed no acute fracture or dislocation, with mild degenerative changes at C5-C6. [1]\n\n2. **MRI Cervical Spine (03/20/2023)** at Advanced Imaging Center revealed a small central disc protrusion at C5-C6 with mild neural foraminal narrowing. No cord compression was noted. [2]\n\n3. **Follow-up X-ray (05/01/2023)** showed stable alignment with no interval change. [3]",
  
  "citations": [
    {
      "id": 1,
      "document_id": "doc_abc",
      "document_name": "ER_Report_01152023.pdf",
      "pages": [4],
      "excerpt": "RADIOLOGY: Cervical spine 3-view demonstrates...",
      "confidence": 0.97
    },
    {
      "id": 2,
      "document_id": "doc_def",
      "document_name": "MRI_Report_03202023.pdf",
      "pages": [1, 2],
      "excerpt": "FINDINGS: At C5-C6 there is a small central...",
      "confidence": 0.95
    }
  ],
  
  "metadata": {
    "sources_searched": 47,
    "sources_used": 3,
    "processing_time_ms": 2340,
    "model": "gpt-4-turbo"
  }
}
```

---

## 6. Scale & Performance

### Processing Pipeline Scaling

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     SCALING ARCHITECTURE                                      │
└──────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────┐
                        │   SQS Queues    │
                        │  (per stage)    │
                        └─────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  OCR Workers  │   │ Extraction    │   │  Embedding    │
    │  (GPU-based)  │   │ Workers (CPU) │   │  Workers      │
    │               │   │               │   │               │
    │  EC2 g4dn    │   │  EC2 c6i      │   │  Lambda       │
    │  Auto-scale   │   │  Auto-scale   │   │  (256 conc)   │
    │  2-20 inst    │   │  5-50 inst    │   │               │
    └───────────────┘   └───────────────┘   └───────────────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ┌─────────────────────────────────────────────────────┐
    │              Auto-Scaling Triggers                   │
    │  • Queue depth > 1000: scale up                     │
    │  • Queue depth < 100: scale down                    │
    │  • Processing time > SLA: scale up                  │
    │  • Time-of-day patterns (business hours surge)      │
    └─────────────────────────────────────────────────────┘
```

### Performance Targets

| Operation | Target Latency | P99 Latency | Throughput |
|-----------|---------------|-------------|------------|
| Document upload (acknowledgment) | < 500ms | < 1s | 1000 req/s |
| OCR (per page) | < 3s | < 10s | 500 pages/min |
| Entity extraction (per doc) | < 30s | < 2min | 100 docs/min |
| Timeline generation | < 5s | < 15s | 200 req/s |
| RAG query | < 3s | < 8s | 500 req/s |
| Full case processing (500 pages) | < 15min | < 30min | N/A |

### Caching Strategy

```python
CACHE_CONFIG = {
    # Redis cache layers
    "layers": {
        "l1_local": {
            "type": "in-memory",
            "ttl": 60,  # seconds
            "max_size": "100MB",
            "use_for": ["hot_queries", "session_data"]
        },
        "l2_redis": {
            "type": "redis_cluster",
            "ttl": 3600,
            "use_for": ["timeline_cache", "entity_cache", "embedding_cache"]
        },
        "l3_cdn": {
            "type": "cloudfront",
            "ttl": 86400,
            "use_for": ["static_documents", "generated_reports"]
        }
    },
    
    # Cache invalidation triggers
    "invalidation": {
        "document_uploaded": ["timeline_cache", "entity_cache"],
        "event_modified": ["timeline_cache"],
        "case_updated": ["all_case_caches"]
    }
}
```

### Database Optimization

```sql
-- Partitioning strategy for large tables
CREATE TABLE medical_events (
    -- ... columns ...
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE medical_events_2024_01 PARTITION OF medical_events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Connection pooling (PgBouncer config)
-- pool_mode = transaction
-- max_client_conn = 10000
-- default_pool_size = 100

-- Read replicas for query-heavy workloads
-- Primary: writes + critical reads
-- Replica 1: RAG queries, search
-- Replica 2: Analytics, reporting
```

---

## 7. Security & Compliance

### HIPAA Compliance Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     HIPAA SECURITY CONTROLS                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENCRYPTION                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  At Rest                          │  In Transit                             │
│  • S3: SSE-S3 or SSE-KMS         │  • TLS 1.3 minimum                      │
│  • RDS: AES-256                  │  • Certificate pinning                  │
│  • Redis: TLS + auth             │  • mTLS for internal services           │
│  • Per-tenant encryption keys    │                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACCESS CONTROLS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Authentication                   │  Authorization                          │
│  • SSO (SAML 2.0, OIDC)          │  • RBAC with tenant isolation           │
│  • MFA required                  │  • Row-level security in PostgreSQL     │
│  • Session timeout (15 min)      │  • Document-level ACLs                  │
│  • IP allowlisting (optional)    │  • Principle of least privilege         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUDIT & MONITORING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Logging                          │  Monitoring                             │
│  • All PHI access logged         │  • Real-time anomaly detection          │
│  • Immutable audit trail         │  • Failed access alerts                 │
│  • 7-year retention              │  • Data exfiltration detection          │
│  • Log integrity verification    │  • SOC integration                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Multi-Tenant Isolation

```python
class TenantMiddleware:
    """
    Enforce tenant isolation at every layer.
    """
    
    def __call__(self, request):
        # Extract tenant from JWT
        tenant_id = self.extract_tenant(request.auth_token)
        
        # Set tenant context for entire request
        set_current_tenant(tenant_id)
        
        # Configure database connection
        db_connection.set_schema(f"tenant_{tenant_id}")
        
        # Configure vector store namespace
        vector_store.set_namespace(f"tenant_{tenant_id}")
        
        # Configure S3 prefix
        s3_client.set_prefix(f"tenants/{tenant_id}/")
        
        # Add tenant to all log entries
        logger.set_context(tenant_id=tenant_id)
        
        return self.get_response(request)


class TenantAwareQuery:
    """
    Automatically scope all queries to current tenant.
    Prevents cross-tenant data access even with bugs.
    """
    
    def execute(self, query):
        tenant_id = get_current_tenant()
        
        # Row-level security as defense in depth
        # Even if app logic fails, DB enforces isolation
        query = f"""
            SET app.current_tenant = '{tenant_id}';
            {query}
        """
        
        return self.db.execute(query)
```

### LLM Security Considerations

```python
class SecureLLMClient:
    """
    Security wrapper for LLM API calls.
    """
    
    def __init__(self):
        self.providers = {
            "openai": OpenAIClient(
                organization="org_xxx",
                api_key=secrets.get("OPENAI_API_KEY")
            ),
            "anthropic": AnthropicClient(
                api_key=secrets.get("ANTHROPIC_API_KEY")
            )
        }
    
    def complete(self, prompt: str, context: str) -> str:
        # 1. PII detection and scrubbing (optional, for extra safety)
        # Note: For legal use, we typically need PII in context
        # But we can scrub from prompts sent to lower-trust models
        
        # 2. Prompt injection detection
        if self.detect_injection(prompt):
            raise SecurityError("Potential prompt injection detected")
        
        # 3. Rate limiting per tenant
        self.check_rate_limit(get_current_tenant())
        
        # 4. Token budget enforcement
        self.check_token_budget(get_current_tenant())
        
        # 5. Execute with timeout
        response = self.providers["openai"].chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
            ],
            max_tokens=4096,
            timeout=30
        )
        
        # 6. Output validation
        self.validate_output(response.choices[0].message.content)
        
        # 7. Audit logging
        self.log_llm_call(prompt, response, get_current_tenant())
        
        return response.choices[0].message.content
    
    def detect_injection(self, text: str) -> bool:
        """Detect common prompt injection patterns."""
        patterns = [
            r"ignore previous instructions",
            r"disregard the above",
            r"new instructions:",
            r"system prompt:",
        ]
        return any(re.search(p, text, re.I) for p in patterns)
```

---

## 8. Failure Handling & Reliability

### Retry Strategies

```python
RETRY_CONFIG = {
    "ocr": {
        "max_retries": 3,
        "backoff": "exponential",
        "base_delay": 5,
        "max_delay": 60,
        "retry_on": ["timeout", "rate_limit", "service_unavailable"],
        "fallback": "alternate_ocr_provider"
    },
    "llm_extraction": {
        "max_retries": 2,
        "backoff": "exponential",
        "base_delay": 10,
        "max_delay": 120,
        "retry_on": ["timeout", "rate_limit"],
        "fallback": "simpler_model"  # GPT-4 -> GPT-3.5
    },
    "embedding": {
        "max_retries": 5,
        "backoff": "linear",
        "base_delay": 1,
        "max_delay": 30,
        "batch_size_reduction": True  # Reduce batch size on failure
    }
}
```

### Dead Letter Queue Processing

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FAILURE RECOVERY FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────┘

     Main Queue                DLQ                   Recovery
         │                      │                       │
         ▼                      ▼                       ▼
    ┌─────────┐  fail 3x   ┌─────────┐  analyze    ┌─────────┐
    │ Process │───────────▶│   DLQ   │────────────▶│ Triage  │
    └─────────┘            └─────────┘             └─────────┘
                                                        │
                           ┌────────────────────────────┼────────────────┐
                           ▼                            ▼                ▼
                    ┌─────────────┐            ┌─────────────┐   ┌─────────────┐
                    │   Auto-     │            │   Manual    │   │   Alert     │
                    │   Retry     │            │   Review    │   │   & Skip    │
                    │             │            │             │   │             │
                    │ • Different │            │ • Bad scan  │   │ • Corrupt   │
                    │   provider  │            │ • Edge case │   │   file      │
                    │ • Smaller   │            │ • New doc   │   │ • Invalid   │
                    │   chunks    │            │   type      │   │   format    │
                    └─────────────┘            └─────────────┘   └─────────────┘
```

### Graceful Degradation

```python
class ResilientPipeline:
    """
    Graceful degradation when components fail.
    """
    
    async def process_document(self, document: Document) -> ProcessingResult:
        result = ProcessingResult(document_id=document.id)
        
        # OCR: Required - fail if unavailable
        try:
            text = await self.ocr_service.extract(document)
        except OCRServiceError:
            # Try backup provider
            text = await self.backup_ocr.extract(document)
        
        result.text = text
        result.ocr_status = "completed"
        
        # Classification: Degradable
        try:
            doc_type = await self.classifier.classify(text)
        except ClassifierError:
            doc_type = "unknown"  # Continue with unknown type
            result.add_warning("Classification failed, manual review needed")
        
        result.document_type = doc_type
        
        # Entity Extraction: Degradable with multiple strategies
        try:
            entities = await self.extractor.extract(text, doc_type)
        except LLMError:
            # Fallback to rule-based extraction
            entities = await self.rule_extractor.extract(text)
            result.add_warning("Using rule-based extraction, reduced accuracy")
        except Exception:
            # Minimum viable: just store text for search
            entities = {}
            result.add_warning("Entity extraction failed, text-only indexing")
        
        result.entities = entities
        
        # Embedding: Degradable
        try:
            await self.embed_and_index(document.id, text)
            result.embedding_status = "completed"
        except EmbeddingError:
            result.embedding_status = "failed"
            result.add_warning("Search indexing failed, will retry")
            await self.queue_for_retry(document.id, "embedding")
        
        return result
```

---

## 9. Cost Optimization

### Cost Breakdown (Estimated Monthly)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│               MONTHLY COST ESTIMATE (500 firms, 1M pages/month)              │
├────────────────────────────────────────┬─────────────────────────────────────┤
│  Component                             │  Monthly Cost                       │
├────────────────────────────────────────┼─────────────────────────────────────┤
│  LLM API Calls                         │                                     │
│  ├─ Entity extraction (GPT-4-turbo)    │  $15,000 - $25,000                 │
│  ├─ Summarization                      │  $5,000 - $10,000                  │
│  ├─ RAG queries                        │  $8,000 - $15,000                  │
│  └─ Embeddings (text-embedding-3)      │  $1,000 - $2,000                   │
├────────────────────────────────────────┼─────────────────────────────────────┤
│  OCR Services                          │                                     │
│  ├─ Azure Document AI                  │  $10,000 - $15,000                 │
│  └─ Backup provider                    │  $2,000 - $3,000                   │
├────────────────────────────────────────┼─────────────────────────────────────┤
│  Infrastructure                        │                                     │
│  ├─ EC2 (workers, API servers)         │  $8,000 - $12,000                  │
│  ├─ RDS PostgreSQL                     │  $3,000 - $5,000                   │
│  ├─ S3 Storage + transfer              │  $2,000 - $4,000                   │
│  ├─ Turbopuffer (vector DB)            │  $300 - $500                       │
│  ├─ Redis/ElastiCache                  │  $1,000 - $2,000                   │
│  └─ Networking, misc                   │  $1,000 - $2,000                   │
├────────────────────────────────────────┼─────────────────────────────────────┤
│  TOTAL                                 │  $55,000 - $95,000/month           │
│  Per firm (500 firms)                  │  $110 - $190/month                 │
│  Per page processed                    │  $0.055 - $0.095                   │
└────────────────────────────────────────┴─────────────────────────────────────┘
```

### Optimization Strategies

```python
class CostOptimizer:
    """
    Strategies to reduce processing costs while maintaining quality.
    """
    
    # 1. Model tiering - use cheaper models when possible
    MODEL_TIERS = {
        "simple_extraction": "gpt-3.5-turbo",      # $0.0005/1K tokens
        "complex_extraction": "gpt-4-turbo",       # $0.01/1K tokens  
        "summarization": "claude-3-sonnet",        # $0.003/1K tokens
        "classification": "fine-tuned-bert",       # Self-hosted, ~free
    }
    
    # 2. Caching embeddings - never embed same text twice
    async def get_embedding(self, text: str) -> List[float]:
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        
        cached = await self.cache.get(f"emb:{cache_key}")
        if cached:
            return cached  # Saved $0.0001
        
        embedding = await self.embedding_model.embed(text)
        await self.cache.set(f"emb:{cache_key}", embedding, ttl=86400*30)
        return embedding
    
    # 3. Smart batching - reduce API call overhead
    async def batch_extract(self, documents: List[Document]):
        # Group similar documents for batch processing
        batches = self.group_by_type(documents)
        
        for doc_type, batch in batches.items():
            # Process up to 5 documents in single prompt
            combined_prompt = self.create_batch_prompt(batch)
            response = await self.llm.complete(combined_prompt)
            self.parse_batch_response(response, batch)
    
    # 4. Progressive processing - stop early when possible
    async def extract_entities(self, text: str, doc_type: str):
        # Try rule-based first (free)
        rule_entities = self.rule_extractor.extract(text)
        
        if self.is_sufficient(rule_entities, doc_type):
            return rule_entities  # Saved $0.02
        
        # Only use LLM for complex cases
        llm_entities = await self.llm_extractor.extract(text)
        return self.merge_entities(rule_entities, llm_entities)
```

---

## 10. Trade-offs & Alternatives

### Key Design Decisions

| Decision | Chosen Approach | Alternative | Trade-off |
|----------|-----------------|-------------|-----------|
| **Tenant isolation** | Per-tenant PostgreSQL schema | Shared tables with tenant_id | Higher complexity, stronger isolation |
| **Vector store** | Turbopuffer (per-case index) | Pinecone (global index) | Higher ops overhead, but privacy guarantee |
| **OCR** | Azure Document AI | Tesseract + custom | Higher cost, but better accuracy on medical docs |
| **LLM provider** | Multi-provider (OpenAI + Anthropic) | Single provider | More complexity, but redundancy + best model per task |
| **Processing** | Async queue-based | Sync request-response | Added complexity, but necessary for scale |
| **Chunking** | Semantic section-aware | Fixed-size sliding window | Higher implementation cost, better retrieval |

### When to Use Different Approaches

```python
# Decision matrix for processing strategies

def choose_processing_strategy(document: Document) -> ProcessingStrategy:
    
    # High-value cases: Maximum accuracy
    if document.case.estimated_value > 500_000:
        return ProcessingStrategy(
            ocr="azure_premium",
            extraction_model="gpt-4-turbo",
            human_review=True,
            dual_extraction=True  # Run twice, compare
        )
    
    # Standard cases: Balanced
    if document.case.estimated_value > 50_000:
        return ProcessingStrategy(
            ocr="azure_standard",
            extraction_model="gpt-4-turbo",
            human_review=False,
            confidence_threshold=0.9
        )
    
    # Lower-value cases: Cost-optimized
    return ProcessingStrategy(
        ocr="tesseract_enhanced",
        extraction_model="gpt-3.5-turbo",
        human_review=False,
        confidence_threshold=0.85
    )
```

### Future Considerations

1. **Fine-tuned models**: As volume grows, fine-tune extraction models on labeled data to reduce LLM costs and improve accuracy

2. **On-premise processing**: For highly sensitive clients, offer on-premise deployment option

3. **Real-time streaming**: As latency requirements tighten, move from batch to streaming processing

4. **Multi-modal analysis**: Integrate medical imaging analysis (X-rays, MRIs) beyond text extraction

---

## 11. Interview Discussion Points

### Questions They Might Ask

**Q: How do you handle documents with poor scan quality?**
> A: Multi-stage approach: (1) Image preprocessing (deskewing, denoising, contrast enhancement), (2) Multiple OCR engines with voting, (3) Confidence scoring with automatic flagging for human review, (4) LLM-based error correction using medical context.

**Q: How do you prevent hallucinations in medical summaries?**
> A: (1) RAG grounding - every claim must cite source, (2) Structured extraction over free-form generation, (3) Post-generation verification against source docs, (4) Confidence scoring with thresholds, (5) "I don't know" training for uncertain cases, (6) Human-in-loop for low-confidence outputs.

**Q: How do you maintain attorney-client privilege with a shared platform?**
> A: (1) Per-tenant database schemas, (2) Per-case vector indices, (3) Tenant-scoped encryption keys, (4) No cross-tenant training or data access, (5) SOC 2 Type II certification, (6) Audit logs for all access, (7) Option for firm-specific LLM instances.

**Q: How would you handle a 10x increase in document volume?**
> A: (1) Horizontal scaling of worker pools (queue-based architecture supports this), (2) Database read replicas for query load, (3) More aggressive caching, (4) Batch processing optimization, (5) Consider dedicated compute for largest tenants.

**Q: What's your approach to handling different medical record formats?**
> A: (1) Template library for common EMR systems (Epic, Cerner, etc.), (2) Generic fallback parser, (3) Document classification to route to appropriate parser, (4) Continuous learning from corrections to improve parsers.

### Questions to Ask Them

1. "How do you currently handle the trade-off between processing speed and extraction accuracy? Do customers prefer faster results or more accurate results?"

2. "What's your approach to measuring and improving entity extraction accuracy? Do you have labeled datasets for evaluation?"

3. "How do you handle the integration with different case management systems that law firms use?"

4. "What's the most challenging document type you've encountered, and how did you solve it?"

5. "How does the Reasoning Mode architecture differ from standard chain-of-thought prompting? What evaluation framework do you use?"

---

## Appendix: Quick Reference

### System Context Diagram
```
┌─────────────────────────────────────────────────────────────────────┐
│                      LAW FIRM USERS                                  │
│  (Attorneys, Paralegals, Case Managers)                             │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MEDICAL RECORDS SYSTEM                            │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Ingest  │─▶│  Process │─▶│  Analyze │─▶│  Query   │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Medical    │ │    OCR       │ │    LLM       │ │   Medical    │
│   Portals    │ │   Services   │ │   Providers  │ │   Code DBs   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Key Metrics to Monitor

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| Latency | P95 OCR time | < 5s/page | > 10s |
| Latency | P95 extraction time | < 30s/doc | > 2min |
| Latency | P95 query time | < 3s | > 8s |
| Quality | Entity extraction accuracy | > 95% | < 90% |
| Quality | Hallucination rate | < 2% | > 5% |
| Reliability | Processing success rate | > 99% | < 98% |
| Cost | Per-page processing cost | < $0.10 | > $0.15 |

---

*This system design document is optimized for a 45-60 minute system design interview. Focus on the high-level architecture first, then dive into 2-3 components based on interviewer interest.*
