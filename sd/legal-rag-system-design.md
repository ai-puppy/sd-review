# System Design: RAG for Legal Document Analysis

## Table of Contents
1. [Problem Statement & Why RAG is Hard for Legal](#1-problem-statement--why-rag-is-hard-for-legal)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Document Processing Pipeline](#3-document-processing-pipeline)
4. [Chunking Strategies (Critical!)](#4-chunking-strategies-critical)
5. [Embedding & Indexing](#5-embedding--indexing)
6. [Retrieval Pipeline](#6-retrieval-pipeline)
7. [Generation with Citations](#7-generation-with-citations)
8. [Hallucination Prevention](#8-hallucination-prevention)
9. [Multi-Document Reasoning](#9-multi-document-reasoning)
10. [Scale & Performance](#10-scale--performance)
11. [Cost Optimization](#11-cost-optimization)
12. [Interview Discussion Points](#12-interview-discussion-points)

---

## 1. Problem Statement & Why RAG is Hard for Legal

### The Challenge

Standard RAG works well for simple Q&A over documents. But legal documents break standard RAG in several ways:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              WHY STANDARD RAG FAILS FOR LEGAL DOCUMENTS                      │
└─────────────────────────────────────────────────────────────────────────────┘

Problem 1: CROSS-REFERENCES
───────────────────────────
Page 47: "Subject to the limitations in Section 12.3..."
Page 12: "12.3 Limitation of Liability: Notwithstanding anything 
         to the contrary, except as set forth in Exhibit B..."
Exhibit B (page 203): "The following exclusions apply..."

→ Understanding page 47 requires context from 3 different locations!


Problem 2: DEFINITIONS CHANGE MEANING
─────────────────────────────────────
Page 2: "'Confidential Information' means any information 
        disclosed by either party that is marked confidential 
        or that reasonably should be understood to be confidential."

Page 89: "Provider shall not disclose any Confidential Information 
         to third parties."

→ Can't understand page 89 without the definition from page 2!


Problem 3: NEGATIONS & EXCEPTIONS
─────────────────────────────────
"The indemnifying party shall defend all claims, 
 EXCEPT for claims arising from:
 (a) gross negligence,
 (b) willful misconduct, or
 (c) breach of Section 7.2"

→ Simple semantic search might return this for "what claims are covered?"
   but miss that it's actually listing EXCLUSIONS


Problem 4: TEMPORAL & CONDITIONAL LOGIC
───────────────────────────────────────
"Upon termination, and subject to Section 14.2, the receiving 
 party shall, within thirty (30) days, unless otherwise agreed 
 in writing, return or destroy all Confidential Information, 
 PROVIDED THAT the receiving party may retain one (1) archival 
 copy solely for legal compliance purposes."

→ Multiple conditions, exceptions, and timeframes in one sentence
```

### Stanford Research: The Hallucination Problem

A 2024 Stanford study found that even RAG-augmented legal AI tools hallucinate **17-33% of the time** on legal research tasks. Key findings:

| System Type | Hallucination Rate | Citation Accuracy |
|-------------|-------------------|-------------------|
| Base LLM (no RAG) | 58-82% | N/A |
| RAG with basic chunking | 17-33% | 65-78% |
| RAG with legal-specific design | 5-12% | 89-95% |

**Our target: < 5% hallucination rate with > 95% citation accuracy**

### Functional Requirements

| Requirement | Description |
|-------------|-------------|
| **Document Ingestion** | Handle contracts, pleadings, discovery responses, depositions (PDF, DOCX, scanned) |
| **Semantic Search** | Natural language queries across document corpus |
| **Cross-Reference Resolution** | Understand "see Section X" and pull in context |
| **Definition Awareness** | Apply defined terms throughout document |
| **Multi-Document Queries** | "Compare indemnification clauses across all vendor contracts" |
| **Cited Answers** | Every claim must link to source with page/paragraph |
| **Reasoning Transparency** | Show how answer was derived (chain-of-thought) |

### Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Latency** | < 5 seconds for single-document query, < 15s for multi-doc |
| **Accuracy** | > 95% retrieval precision, < 5% hallucination rate |
| **Scale** | Support 10,000+ page document sets (e.g., M&A due diligence) |
| **Isolation** | Complete tenant/case separation (attorney-client privilege) |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LEGAL RAG SYSTEM ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              USER QUERY
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY PROCESSOR                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Intent    │  │   Query     │  │   Legal     │  │  Reference  │        │
│  │   Classify  │  │   Rewrite   │  │   Term      │  │  Detection  │        │
│  │             │  │             │  │   Expansion │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL ENGINE                                    │
│                                                                              │
│    ┌──────────────────┐              ┌──────────────────┐                   │
│    │  VECTOR SEARCH   │              │  KEYWORD SEARCH  │                   │
│    │  (Semantic)      │              │  (BM25/Lexical)  │                   │
│    │                  │              │                  │                   │
│    │  • Dense embed   │              │  • Legal terms   │                   │
│    │  • Similarity    │              │  • Exact phrases │                   │
│    │  • Top-k = 50    │              │  • Section refs  │                   │
│    └────────┬─────────┘              └────────┬─────────┘                   │
│             │                                 │                              │
│             └────────────┬────────────────────┘                              │
│                          ▼                                                   │
│              ┌──────────────────────┐                                        │
│              │   HYBRID FUSION      │                                        │
│              │   (RRF Algorithm)    │                                        │
│              └──────────┬───────────┘                                        │
│                         ▼                                                    │
│              ┌──────────────────────┐                                        │
│              │   CROSS-ENCODER      │                                        │
│              │   RE-RANKING         │                                        │
│              └──────────┬───────────┘                                        │
│                         ▼                                                    │
│              ┌──────────────────────┐                                        │
│              │   CONTEXT EXPANSION  │                                        │
│              │   • Definitions      │                                        │
│              │   • Cross-refs       │                                        │
│              │   • Parent sections  │                                        │
│              └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GENERATION ENGINE                                    │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ CONTEXT ASSEMBLY│───▶│   LLM GENERATE  │───▶│   CITATION      │         │
│  │ • Order chunks  │    │   • GPT-4/Claude│    │   VERIFICATION  │         │
│  │ • Add metadata  │    │   • Structured  │    │   • Source check│         │
│  │ • Token budget  │    │   • Chain-of-   │    │   • Quote verify│         │
│  │                 │    │     thought     │    │   • Link gen    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                           CITED RESPONSE
                    + Reasoning Trace (optional)
```

---

## 3. Document Processing Pipeline

### Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT INGESTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

   PDF/DOCX Upload
         │
         ▼
┌─────────────────┐
│  TEXT EXTRACTION│     • PyMuPDF for native PDFs
│                 │     • OCR for scanned (Tesseract/Azure)
│                 │     • python-docx for Word files
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   STRUCTURE     │     • Detect headers, sections, paragraphs
│   RECOGNITION   │     • Identify tables, lists, exhibits
│                 │     • Page/line number mapping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DEFINITION     │     • Extract "X means..." patterns
│  EXTRACTION     │     • Build definition dictionary
│                 │     • Map defined terms to locations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CROSS-REF      │     • Detect "Section X", "Exhibit Y"
│  DETECTION      │     • Build reference graph
│                 │     • Resolve forward/backward refs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SEMANTIC       │     • Legal-aware chunking (see Section 4)
│  CHUNKING       │     • Preserve clause boundaries
│                 │     • Maintain hierarchy
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EMBEDDING &    │     • Generate vector embeddings
│  INDEXING       │     • Store with rich metadata
│                 │     • Build inverted index
└─────────────────┘
```

### Document Structure Recognition

```python
class LegalDocumentParser:
    """
    Parse legal documents while preserving structure.
    """
    
    # Common legal document patterns
    SECTION_PATTERNS = [
        r'^ARTICLE\s+[IVX\d]+[.:]?\s*(.+)$',           # ARTICLE I. DEFINITIONS
        r'^Section\s+[\d.]+[.:]?\s*(.+)$',             # Section 1.1. Purpose
        r'^[\d]+\.[\d]+\.?[\d]*\s+(.+)$',              # 1.1 or 1.1.1
        r'^\([a-z]\)\s+(.+)$',                         # (a) subsection
        r'^\([ivx]+\)\s+(.+)$',                        # (i), (ii), (iii)
    ]
    
    DEFINITION_PATTERNS = [
        r'"([^"]+)"\s+(?:means?|shall mean|refers? to|is defined as)',
        r'\'([^\']+)\'\s+(?:means?|shall mean|refers? to)',
        r'([A-Z][a-zA-Z\s]+)\s+(?:means?|shall mean)\s+',
    ]
    
    CROSS_REF_PATTERNS = [
        r'(?:Section|Article|Paragraph|Clause)\s+([\d.]+)',
        r'Exhibit\s+([A-Z])',
        r'Schedule\s+([\d.]+|[A-Z])',
        r'(?:as defined|set forth) in\s+(?:Section|Article)\s+([\d.]+)',
    ]
    
    def parse(self, document: Document) -> ParsedLegalDocument:
        # Extract text with position information
        text_blocks = self.extract_with_positions(document)
        
        # Build document tree
        doc_tree = self.build_hierarchy(text_blocks)
        
        # Extract definitions
        definitions = self.extract_definitions(text_blocks)
        
        # Build cross-reference graph
        xref_graph = self.build_xref_graph(text_blocks)
        
        return ParsedLegalDocument(
            tree=doc_tree,
            definitions=definitions,
            cross_references=xref_graph,
            metadata=self.extract_metadata(document)
        )
    
    def extract_definitions(self, blocks: List[TextBlock]) -> Dict[str, Definition]:
        """
        Extract defined terms and their definitions.
        
        Example:
        "Confidential Information" means any non-public information...
        
        Returns:
        {
            "Confidential Information": Definition(
                term="Confidential Information",
                definition="any non-public information...",
                location=Location(page=2, section="1.1"),
                variations=["confidential information", "CI"]
            )
        }
        """
        definitions = {}
        
        for block in blocks:
            for pattern in self.DEFINITION_PATTERNS:
                matches = re.finditer(pattern, block.text, re.IGNORECASE)
                for match in matches:
                    term = match.group(1).strip()
                    # Get the full definition (usually continues after "means")
                    definition_text = self.extract_definition_text(
                        block.text, 
                        match.end()
                    )
                    
                    definitions[term] = Definition(
                        term=term,
                        definition=definition_text,
                        location=block.location,
                        variations=self.generate_variations(term)
                    )
        
        return definitions
```

---

## 4. Chunking Strategies (Critical!)

This is the **most important part** of legal RAG. Bad chunking = bad retrieval = bad answers.

### Why Standard Chunking Fails

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STANDARD CHUNKING vs LEGAL CHUNKING                       │
└─────────────────────────────────────────────────────────────────────────────┘

STANDARD CHUNKING (512 tokens, 50 overlap):
──────────────────────────────────────────

Original contract section:
┌────────────────────────────────────────────────────────────────────────────┐
│ 7.1 Indemnification by Provider. Provider shall indemnify, defend, and    │
│ hold harmless Client and its officers, directors, employees, and agents   │
│ (collectively, the "Client Indemnitees") from and against any and all     │
│ claims, damages, losses, costs, and expenses (including reasonable        │
│ attorneys' fees) arising out of or relating to: (a) any breach of         │
│ Provider's representations or warranties; (b) Provider's gross negligence │
│ or willful misconduct; (c) any violation of applicable law by Provider;   │
│ or (d) any claim that the Services infringe a third party's intellectual  │
│ property rights; PROVIDED, HOWEVER, that Provider shall not be liable     │
│ for any claims to the extent arising from: (i) Client's modification of   │
│ the Services; (ii) Client's combination of the Services with other        │
│ products not provided by Provider; or (iii) Client's use of the Services  │
│ in violation of this Agreement.                                            │
└────────────────────────────────────────────────────────────────────────────┘

After standard chunking (BROKEN):
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│ Chunk 1:                        │  │ Chunk 2:                        │
│ "7.1 Indemnification by         │  │ "violation of applicable law    │
│ Provider. Provider shall        │  │ by Provider; or (d) any claim   │
│ indemnify, defend, and hold     │  │ that the Services infringe a    │
│ harmless Client and its         │  │ third party's intellectual      │
│ officers, directors..."         │  │ property rights; PROVIDED..."   │
│                                 │  │                                 │
│ ✗ Missing what's covered        │  │ ✗ Missing the "PROVIDED" clause │
│ ✗ Definition of Client          │  │ ✗ Cut off in middle             │
│   Indemnitees incomplete        │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘

Query: "What is Provider NOT liable for?"
→ Chunk 2 might be retrieved, but the crucial "PROVIDED, HOWEVER" exception 
  clause is split across chunks!


LEGAL-AWARE CHUNKING:
─────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ Chunk: Section 7.1 (complete)                                               │
│                                                                             │
│ [Full section text preserved as single unit]                                │
│                                                                             │
│ Metadata:                                                                   │
│ • section_id: "7.1"                                                         │
│ • section_title: "Indemnification by Provider"                              │
│ • parent_section: "7. INDEMNIFICATION"                                      │
│ • defined_terms_used: ["Client Indemnitees", "Services", "Provider"]        │
│ • cross_refs: ["Section 7.2", "Section 3.1"]                                │
│ • has_exceptions: true                                                      │
│ • exception_type: "carve-out"                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Legal-Aware Chunking Algorithm

```python
class LegalChunker:
    """
    Chunk legal documents while preserving semantic integrity.
    """
    
    CONFIG = {
        "target_size": 512,      # Target tokens per chunk
        "max_size": 2048,        # Never exceed (even for full sections)
        "min_size": 100,         # Don't create tiny fragments
        "overlap_sentences": 2,  # For split sections only
    }
    
    # Chunk boundary hierarchy (strongest to weakest)
    BOUNDARY_HIERARCHY = [
        "article",      # ARTICLE I, ARTICLE II
        "section",      # Section 1.1, Section 1.2
        "subsection",   # 1.1.1, 1.1.2
        "paragraph",    # (a), (b), (c)
        "sentence",     # Last resort
    ]
    
    def chunk_document(self, doc: ParsedLegalDocument) -> List[Chunk]:
        chunks = []
        
        # Strategy 1: Section-based chunking (preferred)
        for section in doc.tree.sections:
            section_chunks = self.chunk_section(section, doc)
            chunks.extend(section_chunks)
        
        # Strategy 2: Handle exhibits/schedules separately
        for exhibit in doc.exhibits:
            exhibit_chunks = self.chunk_exhibit(exhibit, doc)
            chunks.extend(exhibit_chunks)
        
        # Enrich all chunks with context
        for chunk in chunks:
            self.enrich_chunk(chunk, doc)
        
        return chunks
    
    def chunk_section(self, section: Section, doc: ParsedLegalDocument) -> List[Chunk]:
        """
        Chunk a section, respecting legal structure.
        """
        token_count = self.count_tokens(section.text)
        
        # Case 1: Section fits in target size → keep whole
        if token_count <= self.CONFIG["target_size"]:
            return [self.create_chunk(section, "complete_section")]
        
        # Case 2: Section exceeds max → must split
        if token_count > self.CONFIG["max_size"]:
            return self.split_large_section(section, doc)
        
        # Case 3: Section is medium-sized → keep whole but note size
        # Keeping legal clauses intact is worth the extra tokens
        return [self.create_chunk(section, "oversized_section")]
    
    def split_large_section(self, section: Section, doc: ParsedLegalDocument) -> List[Chunk]:
        """
        Split large sections at natural legal boundaries.
        """
        chunks = []
        
        # Try to split at subsection boundaries first
        if section.has_subsections:
            for subsection in section.subsections:
                sub_chunks = self.chunk_section(subsection, doc)
                # Add parent context to each subsection chunk
                for chunk in sub_chunks:
                    chunk.parent_context = self.get_section_header(section)
                chunks.extend(sub_chunks)
            return chunks
        
        # Try to split at paragraph markers: (a), (b), (c)
        if section.has_paragraphs:
            current_chunk_parts = []
            current_tokens = 0
            
            for para in section.paragraphs:
                para_tokens = self.count_tokens(para.text)
                
                if current_tokens + para_tokens > self.CONFIG["target_size"]:
                    # Save current chunk
                    if current_chunk_parts:
                        chunks.append(self.create_chunk_from_parts(
                            current_chunk_parts, 
                            section,
                            "paragraph_split"
                        ))
                    current_chunk_parts = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk_parts.append(para)
                    current_tokens += para_tokens
            
            # Don't forget last chunk
            if current_chunk_parts:
                chunks.append(self.create_chunk_from_parts(
                    current_chunk_parts,
                    section,
                    "paragraph_split"
                ))
            
            return chunks
        
        # Last resort: sentence-level splitting
        return self.sentence_split(section)
    
    def enrich_chunk(self, chunk: Chunk, doc: ParsedLegalDocument):
        """
        Add rich metadata to each chunk for better retrieval and generation.
        """
        chunk.metadata = {
            # Location info
            "document_id": doc.id,
            "document_name": doc.name,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "page_numbers": chunk.page_numbers,
            
            # Hierarchy
            "parent_section": chunk.parent_section,
            "child_sections": chunk.child_sections,
            "depth": chunk.depth,  # How deep in document hierarchy
            
            # Legal structure
            "chunk_type": chunk.chunk_type,  # "definition", "obligation", "exception", etc.
            "has_exceptions": self.detect_exceptions(chunk.text),
            "has_conditions": self.detect_conditions(chunk.text),
            
            # Cross-references (critical!)
            "outbound_refs": self.extract_references(chunk.text),
            "inbound_refs": doc.get_references_to(chunk.section_id),
            
            # Defined terms used in this chunk
            "defined_terms": self.find_defined_terms(chunk.text, doc.definitions),
            
            # For retrieval boosting
            "importance_score": self.calculate_importance(chunk, doc),
        }
        
        # Add contextual header for retrieval
        chunk.retrieval_text = self.build_retrieval_text(chunk)
    
    def build_retrieval_text(self, chunk: Chunk) -> str:
        """
        Build enhanced text for embedding that includes context.
        
        This helps semantic search understand what the chunk is about.
        """
        parts = []
        
        # Add section path for context
        if chunk.section_title:
            parts.append(f"[Section: {chunk.section_id} - {chunk.section_title}]")
        
        if chunk.parent_section:
            parts.append(f"[Parent: {chunk.parent_section}]")
        
        # Add chunk type hint
        if chunk.metadata.get("chunk_type"):
            parts.append(f"[Type: {chunk.metadata['chunk_type']}]")
        
        # Add the actual text
        parts.append(chunk.text)
        
        # Add defined terms context (helps with queries about specific terms)
        if chunk.metadata.get("defined_terms"):
            terms = ", ".join(chunk.metadata["defined_terms"][:5])
            parts.append(f"[Uses defined terms: {terms}]")
        
        return "\n".join(parts)
    
    def detect_chunk_type(self, text: str) -> str:
        """
        Classify what type of legal content this chunk contains.
        """
        text_lower = text.lower()
        
        if any(p in text_lower for p in ['"means"', 'shall mean', 'is defined as']):
            return "definition"
        
        if any(p in text_lower for p in ['shall indemnify', 'hold harmless', 'indemnification']):
            return "indemnification"
        
        if any(p in text_lower for p in ['warrants', 'represents', 'representation']):
            return "representation_warranty"
        
        if any(p in text_lower for p in ['shall not', 'prohibited', 'restriction']):
            return "restriction"
        
        if any(p in text_lower for p in ['terminate', 'termination', 'expiration']):
            return "termination"
        
        if any(p in text_lower for p in ['confidential', 'non-disclosure', 'proprietary']):
            return "confidentiality"
        
        if any(p in text_lower for p in ['governing law', 'jurisdiction', 'venue']):
            return "governing_law"
        
        if any(p in text_lower for p in ['provided that', 'provided, however', 'except', 'excluding']):
            return "exception"
        
        return "general"
```

### Chunk Metadata Schema

```typescript
interface LegalChunk {
  // Unique identifier
  id: string;
  
  // Content
  text: string;                    // Original text
  retrieval_text: string;          // Enhanced text for embedding
  
  // Location
  document_id: string;
  section_id: string;              // "7.1.2"
  section_title: string;           // "Limitation of Liability"
  page_numbers: number[];          // [45, 46]
  char_start: number;
  char_end: number;
  
  // Hierarchy
  parent_section: string | null;   // "7.1"
  child_sections: string[];        // ["7.1.2.1", "7.1.2.2"]
  depth: number;                   // 2 (for section 7.1)
  
  // Legal structure
  chunk_type: ChunkType;           // "indemnification", "exception", etc.
  has_exceptions: boolean;
  has_conditions: boolean;
  has_definitions: boolean;
  
  // Cross-references (critical for context expansion)
  outbound_refs: Reference[];      // Sections this chunk references
  inbound_refs: Reference[];       // Sections that reference this chunk
  
  // Defined terms
  defined_terms: string[];         // ["Confidential Information", "Services"]
  
  // Retrieval metadata
  importance_score: number;        // For boosting in search
  embedding_model: string;         // "text-embedding-3-large"
}

type ChunkType = 
  | "definition"
  | "obligation" 
  | "restriction"
  | "representation_warranty"
  | "indemnification"
  | "termination"
  | "confidentiality"
  | "exception"
  | "governing_law"
  | "general";
```

---

## 5. Embedding & Indexing

### Embedding Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EMBEDDING PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Legal Chunk    │
                    │                 │
                    │  "Section 7.1   │
                    │  Provider shall │
                    │  indemnify..."  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ RETRIEVAL TEXT  │           │ ORIGINAL TEXT   │
    │ (for embedding) │           │ (for display)   │
    │                 │           │                 │
    │ "[Section 7.1 - │           │ Stored as-is    │
    │  Indemnification│           │ for citations   │
    │  by Provider]   │           │                 │
    │ [Parent: 7...]  │           │                 │
    │ Provider shall..│           │                 │
    └────────┬────────┘           └─────────────────┘
             │
             ▼
    ┌─────────────────┐
    │  EMBEDDING      │
    │  MODEL          │
    │                 │
    │  OpenAI         │
    │  text-embedding │
    │  -3-large       │
    │  (3072 dims)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  VECTOR +       │
    │  METADATA       │
    │                 │
    │  [0.02, -0.15,  │──────────▶  Turbopuffer
    │   0.87, ...]    │             (per-case index)
    │  + all metadata │
    └─────────────────┘
```

### Per-Case Index Architecture

```python
class LegalVectorStore:
    """
    Manage vector indices for legal documents.
    Uses per-case isolation for attorney-client privilege.
    """
    
    def __init__(self, client: TurbopufferClient):
        self.client = client
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dim = 3072
    
    def get_namespace(self, tenant_id: str, case_id: str) -> str:
        """
        Generate isolated namespace for each case.
        
        This ensures complete data isolation - queries for one case
        can never accidentally return results from another case.
        """
        return f"tenant_{tenant_id}_case_{case_id}"
    
    def index_document(
        self, 
        tenant_id: str, 
        case_id: str, 
        document: ParsedLegalDocument,
        chunks: List[LegalChunk]
    ):
        """
        Index all chunks from a document.
        """
        namespace = self.get_namespace(tenant_id, case_id)
        
        # Generate embeddings in batches
        embeddings = self.embed_chunks(chunks)
        
        # Prepare vectors with metadata
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.id,
                "vector": embedding,
                "attributes": {
                    # Core identifiers
                    "document_id": document.id,
                    "document_name": document.name,
                    "section_id": chunk.section_id,
                    
                    # For filtering
                    "chunk_type": chunk.chunk_type,
                    "page_number": chunk.page_numbers[0] if chunk.page_numbers else None,
                    "importance_score": chunk.importance_score,
                    
                    # For context expansion (stored as JSON strings)
                    "outbound_refs": json.dumps(chunk.outbound_refs),
                    "defined_terms": json.dumps(chunk.defined_terms),
                    
                    # Full text for display
                    "text": chunk.text,
                    "section_title": chunk.section_title,
                }
            })
        
        # Upsert to vector store
        self.client.upsert(
            namespace=namespace,
            vectors=vectors
        )
        
        # Also build BM25 index for keyword search
        self.build_keyword_index(namespace, chunks)
    
    def search(
        self,
        tenant_id: str,
        case_id: str,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 20
    ) -> List[SearchResult]:
        """
        Hybrid search: vector + keyword.
        """
        namespace = self.get_namespace(tenant_id, case_id)
        
        # 1. Vector search
        query_embedding = self.embed_query(query)
        vector_results = self.client.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k * 2,  # Get more for fusion
            filters=filters
        )
        
        # 2. Keyword search (BM25)
        keyword_results = self.keyword_search(
            namespace=namespace,
            query=query,
            top_k=top_k * 2,
            filters=filters
        )
        
        # 3. Hybrid fusion using Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60  # RRF constant
        )
        
        return fused_results[:top_k]
```

---

## 6. Retrieval Pipeline

### Multi-Stage Retrieval

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MULTI-STAGE RETRIEVAL PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

User Query: "What happens if Provider breaches confidentiality?"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: QUERY PROCESSING                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Intent Classification: "consequence_query"                                  │
│  Key Entities: ["Provider", "breach", "confidentiality"]                    │
│  Expanded Terms: ["Confidential Information", "disclosure", "violation"]    │
│  Reference Detection: None                                                   │
│                                                                              │
│  Rewritten Queries:                                                          │
│  • "Provider breach confidentiality consequences"                           │
│  • "breach of confidential information remedies"                            │
│  • "confidentiality violation liability"                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: INITIAL RETRIEVAL (Broad)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Vector Search (top 30):           Keyword Search (top 30):                 │
│  • Section 8.3 (0.87)              • Section 8.1 "Confidential" (12.4)     │
│  • Section 8.1 (0.84)              • Section 8.3 "breach" (11.2)           │
│  • Section 12.2 (0.79)             • Section 12.2 "remedies" (9.8)         │
│  • Section 7.1 (0.76)              • Section 7.1 "indemnify" (8.5)         │
│  • ...                             • ...                                    │
│                                                                              │
│  RRF Fusion → 40 unique candidates                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CROSS-ENCODER RE-RANKING                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Model: cross-encoder/ms-marco-MiniLM-L-12-v2 (or legal fine-tuned)         │
│                                                                              │
│  For each candidate, score relevance to original query:                      │
│  • Section 8.3 "Breach Remedies": 0.94 ✓                                    │
│  • Section 8.1 "Definition of CI": 0.72 (context only)                      │
│  • Section 12.2 "Limitation": 0.89 ✓                                        │
│  • Section 7.1 "Indemnification": 0.85 ✓                                    │
│                                                                              │
│  Top 10 after re-ranking                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: CONTEXT EXPANSION                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For each top chunk, pull in:                                                │
│                                                                              │
│  1. DEFINITIONS                                                              │
│     Section 8.3 uses "Confidential Information"                             │
│     → Pull definition from Section 1.5                                       │
│                                                                              │
│  2. CROSS-REFERENCES                                                         │
│     Section 8.3 says "subject to Section 12.2"                              │
│     → Pull Section 12.2 (Limitation of Liability)                           │
│                                                                              │
│  3. PARENT CONTEXT                                                           │
│     Section 8.3 is under "ARTICLE 8: CONFIDENTIALITY"                       │
│     → Pull article header for context                                        │
│                                                                              │
│  4. RELATED SECTIONS                                                         │
│     Section 7.1 (Indemnification) references Section 8                      │
│     → Already in results, keep it                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                           FINAL CONTEXT SET
                    (10-15 chunks with full context)
```

### Query Processing

```python
class LegalQueryProcessor:
    """
    Process user queries for optimal retrieval.
    """
    
    def __init__(self, definitions: Dict[str, Definition]):
        self.definitions = definitions
        self.legal_synonyms = self.load_legal_synonyms()
    
    def process(self, query: str, document: ParsedLegalDocument) -> ProcessedQuery:
        # 1. Classify intent
        intent = self.classify_intent(query)
        
        # 2. Extract key entities
        entities = self.extract_entities(query)
        
        # 3. Detect section references
        references = self.detect_references(query)
        
        # 4. Expand with defined terms
        expanded_terms = self.expand_with_definitions(entities, document.definitions)
        
        # 5. Add legal synonyms
        synonyms = self.expand_with_synonyms(entities)
        
        # 6. Generate query variations
        variations = self.generate_variations(query, intent, expanded_terms)
        
        return ProcessedQuery(
            original=query,
            intent=intent,
            entities=entities,
            references=references,
            expanded_terms=expanded_terms,
            variations=variations
        )
    
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify what type of legal question this is.
        """
        query_lower = query.lower()
        
        # Definition queries
        if any(p in query_lower for p in ['what is', 'what does', 'define', 'meaning of']):
            return QueryIntent.DEFINITION
        
        # Obligation queries
        if any(p in query_lower for p in ['must', 'shall', 'required to', 'obligated']):
            return QueryIntent.OBLIGATION
        
        # Permission queries
        if any(p in query_lower for p in ['can', 'may', 'allowed to', 'permitted']):
            return QueryIntent.PERMISSION
        
        # Consequence queries
        if any(p in query_lower for p in ['what happens if', 'consequence', 'result of', 'breach']):
            return QueryIntent.CONSEQUENCE
        
        # Comparison queries
        if any(p in query_lower for p in ['compare', 'difference between', 'versus', 'vs']):
            return QueryIntent.COMPARISON
        
        # Timeline queries
        if any(p in query_lower for p in ['when', 'deadline', 'within', 'days', 'period']):
            return QueryIntent.TIMELINE
        
        # Exception queries
        if any(p in query_lower for p in ['except', 'unless', 'exclusion', 'not covered']):
            return QueryIntent.EXCEPTION
        
        return QueryIntent.GENERAL
    
    def expand_with_definitions(
        self, 
        entities: List[str], 
        definitions: Dict[str, Definition]
    ) -> List[str]:
        """
        If query mentions a defined term, include variations.
        
        Example:
        Query: "confidential information disclosure"
        Definitions: {"Confidential Information": ...}
        
        Expanded: ["confidential information", "CI", "proprietary information"]
        """
        expanded = set(entities)
        
        for entity in entities:
            entity_lower = entity.lower()
            
            for term, definition in definitions.items():
                if entity_lower in term.lower() or term.lower() in entity_lower:
                    expanded.add(term)
                    expanded.update(definition.variations)
        
        return list(expanded)
```

### Reciprocal Rank Fusion (RRF)

```python
def reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], 
    k: int = 60
) -> List[SearchResult]:
    """
    Combine multiple ranked lists using RRF.
    
    RRF Score = Σ 1/(k + rank_i) for each list i
    
    Why RRF works well:
    - Doesn't require score normalization between lists
    - Handles different scoring scales (cosine sim vs BM25)
    - Rewards documents that appear in multiple lists
    - Parameter k (default 60) controls fusion behavior
    
    Example:
    Doc A: Vector rank 1, Keyword rank 5
    Doc B: Vector rank 3, Keyword rank 2
    
    RRF_A = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318
    RRF_B = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320
    
    → Doc B wins despite lower vector rank, because keyword rank is better
    """
    
    # Calculate RRF score for each document
    doc_scores = defaultdict(float)
    doc_data = {}
    
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            doc_id = result.id
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] += rrf_score
            doc_data[doc_id] = result  # Keep latest result data
    
    # Sort by RRF score
    sorted_docs = sorted(
        doc_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Return results with RRF scores
    return [
        SearchResult(
            id=doc_id,
            score=rrf_score,
            **doc_data[doc_id].dict()
        )
        for doc_id, rrf_score in sorted_docs
    ]
```

### Context Expansion

```python
class ContextExpander:
    """
    Expand retrieved chunks with necessary context.
    """
    
    def expand(
        self, 
        chunks: List[LegalChunk], 
        document: ParsedLegalDocument,
        query: ProcessedQuery
    ) -> List[ExpandedChunk]:
        """
        For each retrieved chunk, pull in related context.
        """
        expanded = []
        
        for chunk in chunks:
            context_parts = []
            
            # 1. Add definitions used in this chunk
            for term in chunk.defined_terms:
                if term in document.definitions:
                    defn = document.definitions[term]
                    context_parts.append(ContextPart(
                        type="definition",
                        source=f"Section {defn.location.section_id}",
                        text=f'"{term}" means {defn.definition}'
                    ))
            
            # 2. Resolve cross-references
            for ref in chunk.outbound_refs:
                if self.is_relevant_reference(ref, query):
                    ref_chunk = document.get_section(ref.section_id)
                    if ref_chunk:
                        context_parts.append(ContextPart(
                            type="cross_reference",
                            source=f"Section {ref.section_id}",
                            text=ref_chunk.text[:500]  # Truncate if long
                        ))
            
            # 3. Add parent section header for context
            if chunk.parent_section:
                parent = document.get_section(chunk.parent_section)
                if parent:
                    context_parts.append(ContextPart(
                        type="parent_context",
                        source=f"Section {chunk.parent_section}",
                        text=parent.title or parent.text[:200]
                    ))
            
            # 4. Check if this chunk has exceptions elsewhere
            exception_sections = document.get_exceptions_for(chunk.section_id)
            for exc_section in exception_sections:
                context_parts.append(ContextPart(
                    type="exception",
                    source=f"Section {exc_section.section_id}",
                    text=exc_section.text
                ))
            
            expanded.append(ExpandedChunk(
                main_chunk=chunk,
                context=context_parts,
                total_tokens=self.calculate_tokens(chunk, context_parts)
            ))
        
        return expanded
    
    def is_relevant_reference(self, ref: Reference, query: ProcessedQuery) -> bool:
        """
        Decide if a cross-reference should be included.
        
        We can't include ALL references (too much context).
        Include if:
        - Reference is to a definition we need
        - Reference is mentioned in the query
        - Reference is to an exception/limitation clause
        """
        # Always include definition references
        if ref.ref_type == "definition":
            return True
        
        # Include if user asked about this section
        if ref.section_id in query.references:
            return True
        
        # Include exceptions and limitations (often critical)
        if ref.ref_type in ["exception", "limitation", "carve_out"]:
            return True
        
        # Skip general cross-references to avoid context bloat
        return False
```

---

## 7. Generation with Citations

### Prompt Engineering for Legal RAG

```python
class LegalRAGGenerator:
    """
    Generate answers with proper citations.
    """
    
    SYSTEM_PROMPT = """You are a legal document analyst. Your task is to answer questions about legal documents accurately and with proper citations.

CRITICAL RULES:
1. ONLY use information from the provided context. Do not use outside knowledge.
2. EVERY factual claim must have a citation in [brackets].
3. If the context doesn't contain enough information, say "The provided documents do not contain information about..."
4. If you're uncertain, express that uncertainty explicitly.
5. Pay close attention to exceptions, conditions, and defined terms.
6. Quote exact language when it's important (e.g., for legal terms of art).

CITATION FORMAT:
- Use [§X.Y] for section references
- Use [Doc: filename, p.N] for page references
- Use [Def: term] when applying a defined term

HANDLING EXCEPTIONS:
When a clause has exceptions (e.g., "provided that", "except", "unless"), you MUST mention them.
Example: "Provider must indemnify Client [§7.1], except for claims arising from Client's negligence [§7.1(c)]."

HANDLING DEFINED TERMS:
When the answer involves a defined term, apply its definition.
Example: If "Confidential Information" is defined, explain what it includes."""

    def generate(
        self,
        query: str,
        context: List[ExpandedChunk],
        definitions: Dict[str, Definition]
    ) -> GeneratedAnswer:
        
        # Build context string
        context_str = self.build_context_string(context, definitions)
        
        # Build the prompt
        user_prompt = f"""
QUESTION: {query}

DOCUMENT CONTEXT:
{context_str}

RELEVANT DEFINITIONS:
{self.format_definitions(definitions, context)}

Please answer the question based solely on the provided context. Include citations for every claim.
"""
        
        # Generate with GPT-4 / Claude
        response = self.llm.generate(
            system=self.SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.1,  # Low temperature for accuracy
            max_tokens=2000
        )
        
        # Parse and verify citations
        answer = self.parse_response(response)
        verified_answer = self.verify_citations(answer, context)
        
        return verified_answer
    
    def build_context_string(
        self, 
        chunks: List[ExpandedChunk],
        definitions: Dict[str, Definition]
    ) -> str:
        """
        Build formatted context for the prompt.
        """
        parts = []
        
        for i, chunk in enumerate(chunks):
            # Main chunk
            parts.append(f"""
--- EXCERPT {i+1} ---
Source: Section {chunk.main_chunk.section_id} - {chunk.main_chunk.section_title}
Document: {chunk.main_chunk.document_name}
Pages: {chunk.main_chunk.page_numbers}

{chunk.main_chunk.text}
""")
            
            # Add context parts
            for ctx in chunk.context:
                if ctx.type == "definition":
                    parts.append(f"\n[Definition from {ctx.source}]: {ctx.text}")
                elif ctx.type == "cross_reference":
                    parts.append(f"\n[Referenced in {ctx.source}]: {ctx.text}")
                elif ctx.type == "exception":
                    parts.append(f"\n[EXCEPTION in {ctx.source}]: {ctx.text}")
        
        return "\n".join(parts)
```

### Citation Verification

```python
class CitationVerifier:
    """
    Verify that generated citations are accurate.
    """
    
    def verify(
        self, 
        answer: str, 
        context: List[ExpandedChunk]
    ) -> VerifiedAnswer:
        """
        Check each citation in the answer against source documents.
        """
        # Extract citations from answer
        citations = self.extract_citations(answer)
        
        verified_citations = []
        issues = []
        
        for citation in citations:
            # Find the source chunk
            source_chunk = self.find_source(citation, context)
            
            if not source_chunk:
                issues.append(CitationIssue(
                    citation=citation,
                    issue_type="source_not_found",
                    message=f"Citation {citation.reference} not found in context"
                ))
                continue
            
            # Verify the claim against the source
            claim_text = citation.claim_text
            source_text = source_chunk.main_chunk.text
            
            verification = self.verify_claim(claim_text, source_text)
            
            if verification.is_accurate:
                verified_citations.append(VerifiedCitation(
                    claim=claim_text,
                    source_section=citation.reference,
                    source_text=verification.supporting_text,
                    confidence=verification.confidence
                ))
            else:
                issues.append(CitationIssue(
                    citation=citation,
                    issue_type="claim_not_supported",
                    message=verification.reason
                ))
        
        return VerifiedAnswer(
            answer=answer,
            verified_citations=verified_citations,
            issues=issues,
            overall_confidence=self.calculate_confidence(verified_citations, issues)
        )
    
    def verify_claim(self, claim: str, source: str) -> ClaimVerification:
        """
        Use LLM to verify if claim is supported by source.
        """
        prompt = f"""
Verify if the following claim is accurately supported by the source text.

CLAIM: {claim}

SOURCE TEXT: {source}

Respond with:
1. SUPPORTED / NOT_SUPPORTED / PARTIALLY_SUPPORTED
2. If supported, quote the specific text that supports it
3. If not supported, explain why

Be strict: the claim must be directly supported, not just vaguely related.
"""
        
        response = self.llm.generate(prompt, temperature=0)
        return self.parse_verification(response)
```

---

## 8. Hallucination Prevention

### Multi-Layer Defense

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HALLUCINATION PREVENTION LAYERS                           │
└─────────────────────────────────────────────────────────────────────────────┘

LAYER 1: RETRIEVAL GROUNDING
────────────────────────────
• Answer can ONLY come from retrieved context
• No general knowledge allowed
• System prompt explicitly forbids outside knowledge

LAYER 2: CITATION REQUIREMENTS  
─────────────────────────────
• Every claim must have citation
• No uncited factual statements allowed
• LLM prompted to refuse if can't cite

LAYER 3: POST-GENERATION VERIFICATION
─────────────────────────────────────
• Parse out all claims and citations
• Verify each citation exists in source
• Verify claim matches source text
• Flag or remove unverified claims

LAYER 4: CONFIDENCE SCORING
──────────────────────────
• Score answer confidence based on:
  - Retrieval relevance scores
  - Citation verification rate
  - Semantic similarity to sources
• Low confidence → human review

LAYER 5: STRUCTURED OUTPUT
─────────────────────────
• For some queries, use structured extraction
• JSON schema forces specific fields
• Easier to verify than free text
```

### Structured Output for High-Stakes Queries

```python
class StructuredLegalExtractor:
    """
    For high-stakes queries, use structured extraction instead of free-form generation.
    Structured outputs are easier to verify and less prone to hallucination.
    """
    
    def extract_indemnification_analysis(
        self,
        query: str,
        context: List[ExpandedChunk]
    ) -> IndemnificationAnalysis:
        """
        Extract structured indemnification information.
        """
        
        schema = {
            "indemnifying_party": "string - who provides indemnification",
            "indemnified_parties": "array of strings - who is protected",
            "covered_claims": [
                {
                    "description": "string - what's covered",
                    "source_section": "string - citation",
                    "conditions": "array of strings - any conditions"
                }
            ],
            "excluded_claims": [
                {
                    "description": "string - what's excluded",
                    "source_section": "string - citation"
                }
            ],
            "caps_or_limits": {
                "has_cap": "boolean",
                "cap_amount": "string or null",
                "cap_type": "string - per-claim, aggregate, etc.",
                "source_section": "string"
            },
            "procedure": {
                "notice_requirement": "string or null",
                "notice_deadline": "string or null",
                "control_of_defense": "string - who controls",
                "source_section": "string"
            }
        }
        
        prompt = f"""
Analyze the indemnification provisions in the following context and extract structured information.

CONTEXT:
{self.format_context(context)}

Extract the following information as JSON matching this schema:
{json.dumps(schema, indent=2)}

RULES:
1. Only include information explicitly stated in the context
2. Use null for fields where information is not available
3. Include the source section reference for each piece of information
4. Be precise about party names - use exact names from the document
"""
        
        response = self.llm.generate(
            prompt,
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # Parse and validate
        data = json.loads(response)
        validated = self.validate_extraction(data, context)
        
        return IndemnificationAnalysis(**validated)
    
    def validate_extraction(
        self, 
        data: dict, 
        context: List[ExpandedChunk]
    ) -> dict:
        """
        Validate that extracted data matches source documents.
        """
        # Check each citation exists
        for field_name, field_value in self.iterate_fields(data):
            if "source_section" in str(field_name):
                section_id = field_value
                if not self.section_exists(section_id, context):
                    # Remove or flag invalid citation
                    data = self.flag_field(data, field_name, "citation_not_found")
        
        return data
```

### Confidence Scoring

```python
class ConfidenceScorer:
    """
    Calculate confidence score for RAG answers.
    """
    
    def score(
        self,
        answer: GeneratedAnswer,
        retrieval_results: List[SearchResult],
        verification: VerifiedAnswer
    ) -> float:
        """
        Compute overall confidence score (0-1).
        
        Components:
        1. Retrieval confidence: How relevant were the retrieved chunks?
        2. Coverage confidence: Did we find all relevant sections?
        3. Citation confidence: Were all citations verified?
        4. Semantic confidence: Does answer match sources semantically?
        """
        
        # 1. Retrieval confidence (based on search scores)
        if retrieval_results:
            avg_retrieval_score = sum(r.score for r in retrieval_results[:5]) / 5
            # Normalize to 0-1 range
            retrieval_conf = min(avg_retrieval_score / 0.85, 1.0)  
        else:
            retrieval_conf = 0.0
        
        # 2. Coverage confidence
        # Check if we likely found all relevant sections
        query_sections = self.detect_sections_in_query(answer.query)
        found_sections = {r.section_id for r in retrieval_results}
        if query_sections:
            coverage_conf = len(query_sections & found_sections) / len(query_sections)
        else:
            coverage_conf = 1.0 if len(retrieval_results) >= 3 else 0.5
        
        # 3. Citation confidence
        total_citations = len(verification.verified_citations) + len(verification.issues)
        if total_citations > 0:
            citation_conf = len(verification.verified_citations) / total_citations
        else:
            citation_conf = 0.0  # No citations is bad
        
        # 4. Semantic confidence (does answer align with sources?)
        answer_embedding = self.embed(answer.text)
        source_embedding = self.embed(" ".join(
            c.claim for c in verification.verified_citations
        ))
        semantic_conf = self.cosine_similarity(answer_embedding, source_embedding)
        
        # Weighted combination
        final_score = (
            0.25 * retrieval_conf +
            0.20 * coverage_conf +
            0.35 * citation_conf +  # Citations most important
            0.20 * semantic_conf
        )
        
        return final_score
    
    def get_confidence_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
```

---

## 9. Multi-Document Reasoning

### Cross-Document Queries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-DOCUMENT QUERY FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Query: "Compare the indemnification obligations across all three vendor contracts"

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: IDENTIFY DOCUMENTS                                                  │
│                                                                              │
│  Documents in scope:                                                         │
│  • Vendor_A_Contract.pdf (45 pages)                                         │
│  • Vendor_B_MSA.pdf (62 pages)                                              │
│  • Vendor_C_Agreement.pdf (38 pages)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: PARALLEL RETRIEVAL                                                  │
│                                                                              │
│  For each document, retrieve indemnification sections:                       │
│                                                                              │
│  Vendor A: Section 8 (Indemnification) → chunks 45-48                       │
│  Vendor B: Article VII (Indemnity) → chunks 112-118                         │
│  Vendor C: Section 12 (Indemnification) → chunks 67-70                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: STRUCTURED EXTRACTION (per document)                                │
│                                                                              │
│  Extract same schema from each:                                              │
│  • Who indemnifies whom?                                                     │
│  • What's covered?                                                           │
│  • What's excluded?                                                          │
│  • Any caps/limits?                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: COMPARATIVE ANALYSIS                                                │
│                                                                              │
│  LLM compares structured extractions:                                        │
│  • Similarities across all three                                             │
│  • Key differences                                                           │
│  • Risk assessment                                                           │
│  • Recommendations                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        COMPARISON TABLE + ANALYSIS
                          (with citations to each doc)
```

### Multi-Document Implementation

```python
class MultiDocumentAnalyzer:
    """
    Handle queries that span multiple documents.
    """
    
    async def compare_clauses(
        self,
        query: str,
        document_ids: List[str],
        clause_type: str  # e.g., "indemnification", "termination", "confidentiality"
    ) -> ComparisonResult:
        """
        Compare specific clause types across multiple documents.
        """
        
        # Step 1: Retrieve relevant sections from each document
        doc_sections = {}
        for doc_id in document_ids:
            sections = await self.retriever.search(
                query=f"{clause_type} provisions obligations",
                document_id=doc_id,
                filters={"chunk_type": clause_type},
                top_k=10
            )
            doc_sections[doc_id] = sections
        
        # Step 2: Extract structured data from each
        extractions = {}
        for doc_id, sections in doc_sections.items():
            extraction = await self.extract_structured(
                sections=sections,
                extraction_type=clause_type
            )
            extractions[doc_id] = extraction
        
        # Step 3: Generate comparison
        comparison = await self.generate_comparison(
            extractions=extractions,
            comparison_type=clause_type
        )
        
        return comparison
    
    async def generate_comparison(
        self,
        extractions: Dict[str, StructuredExtraction],
        comparison_type: str
    ) -> ComparisonResult:
        """
        Generate natural language comparison with table.
        """
        
        prompt = f"""
Compare the following {comparison_type} provisions from {len(extractions)} documents.

EXTRACTED DATA:
{self.format_extractions(extractions)}

Provide:
1. A comparison table showing key attributes side-by-side
2. Summary of similarities
3. Summary of key differences
4. Risk assessment (which terms are more/less favorable)
5. Recommendations

Use citations [Doc: X, §Y] for each claim.
"""
        
        response = await self.llm.generate(prompt)
        
        # Parse into structured comparison
        return self.parse_comparison(response, extractions)
```

---

## 10. Scale & Performance

### Performance Targets

| Operation | Target | P99 | Notes |
|-----------|--------|-----|-------|
| Query embedding | < 100ms | < 200ms | Batch when possible |
| Vector search | < 50ms | < 150ms | Per-case index helps |
| Keyword search | < 30ms | < 100ms | PostgreSQL FTS |
| Re-ranking (10 docs) | < 500ms | < 1s | Cross-encoder |
| Context expansion | < 200ms | < 500ms | Parallel fetches |
| LLM generation | < 3s | < 8s | Depends on context size |
| **Total E2E** | **< 5s** | **< 12s** | Single doc query |

### Caching Strategy

```python
CACHE_LAYERS = {
    "embedding_cache": {
        "type": "redis",
        "ttl": 86400 * 30,  # 30 days
        "key_format": "emb:sha256({text})",
        "description": "Cache embeddings - same text always gets same embedding"
    },
    
    "query_cache": {
        "type": "redis",
        "ttl": 3600,  # 1 hour
        "key_format": "query:{tenant}:{case}:{query_hash}",
        "description": "Cache full query results - same query, same answer"
    },
    
    "chunk_cache": {
        "type": "redis",
        "ttl": 86400,  # 24 hours
        "key_format": "chunk:{document_id}:{section_id}",
        "description": "Cache expanded chunks with context"
    },
    
    "definition_cache": {
        "type": "local_memory",
        "ttl": 3600,
        "key_format": "def:{document_id}",
        "description": "Cache parsed definitions per document"
    }
}

class CachedRAGPipeline:
    """
    RAG pipeline with multi-layer caching.
    """
    
    async def query(self, query: str, case_id: str) -> Answer:
        # Check query cache first
        cache_key = self.query_cache_key(query, case_id)
        cached = await self.cache.get(cache_key)
        if cached:
            return cached  # Full cache hit
        
        # Process query
        processed_query = self.process_query(query)
        
        # Get embedding (cached)
        embedding = await self.get_cached_embedding(processed_query.text)
        
        # Retrieve (not cached - results change with document updates)
        results = await self.retrieve(embedding, case_id)
        
        # Expand context (partially cached)
        expanded = await self.expand_with_cache(results)
        
        # Generate (not cached - want fresh generation)
        answer = await self.generate(query, expanded)
        
        # Cache the result
        await self.cache.set(cache_key, answer, ttl=3600)
        
        return answer
```

### Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HORIZONTAL SCALING                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                         Load Balancer
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌───────────┐     ┌───────────┐     ┌───────────┐
      │  API Pod  │     │  API Pod  │     │  API Pod  │
      │    #1     │     │    #2     │     │    #3     │
      └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
      ┌───────────┐     ┌───────────┐     ┌───────────┐
      │   Redis   │     │ Turbopuff │     │PostgreSQL │
      │  Cluster  │     │   (per-   │     │  (read    │
      │           │     │   case)   │     │  replicas)│
      └───────────┘     └───────────┘     └───────────┘

Scaling triggers:
• API Pods: CPU > 70% or request latency P95 > 3s
• Redis: Memory > 80% or connections > 10k
• PostgreSQL: Add read replicas when query load increases
• Turbopuffer: Serverless, scales automatically
```

---

## 11. Cost Optimization

### Cost Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PER-QUERY COST BREAKDOWN (500-page contract)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DOCUMENT INGESTION (one-time per document):                                 │
│  ├─ Text extraction:                    ~$0.00 (local processing)           │
│  ├─ Chunking & parsing:                 ~$0.00 (local processing)           │
│  ├─ Embeddings (500 chunks × $0.00013): ~$0.065                             │
│  └─ Storage (vectors + text):           ~$0.02/month                        │
│      TOTAL INGESTION:                   ~$0.085                              │
│                                                                              │
│  PER-QUERY COSTS:                                                            │
│  ├─ Query embedding:                    ~$0.00013                            │
│  ├─ Vector search:                      ~$0.0001 (Turbopuffer)              │
│  ├─ Re-ranking (10 docs):               ~$0.001 (local model)               │
│  ├─ Context assembly:                   ~$0.00 (local)                      │
│  ├─ LLM generation:                                                          │
│  │   ├─ Input (4K context tokens):      ~$0.04 (GPT-4 turbo)               │
│  │   └─ Output (500 tokens):            ~$0.015                             │
│  └─ Citation verification:              ~$0.01 (mini LLM call)              │
│      TOTAL PER-QUERY:                   ~$0.07                               │
│                                                                              │
│  AT SCALE (10,000 queries/month):                                            │
│  └─ Monthly cost:                       ~$700 + infrastructure              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cost Optimization Strategies

```python
class CostOptimizedRAG:
    """
    Strategies to reduce RAG costs while maintaining quality.
    """
    
    # 1. Tiered model selection
    MODEL_TIERS = {
        "simple_lookup": {
            "model": "gpt-3.5-turbo",
            "cost_per_1k_tokens": 0.0005,
            "use_when": "definition lookups, simple factual queries"
        },
        "standard_analysis": {
            "model": "gpt-4-turbo",
            "cost_per_1k_tokens": 0.01,
            "use_when": "most queries, moderate complexity"
        },
        "complex_reasoning": {
            "model": "gpt-4",
            "cost_per_1k_tokens": 0.03,
            "use_when": "multi-document comparison, legal reasoning"
        }
    }
    
    def select_model(self, query: ProcessedQuery, context_size: int) -> str:
        # Simple definition lookup → cheap model
        if query.intent == QueryIntent.DEFINITION:
            return self.MODEL_TIERS["simple_lookup"]["model"]
        
        # Complex comparison → expensive model
        if query.intent == QueryIntent.COMPARISON:
            return self.MODEL_TIERS["complex_reasoning"]["model"]
        
        # Default to standard
        return self.MODEL_TIERS["standard_analysis"]["model"]
    
    # 2. Context window optimization
    def optimize_context(
        self, 
        chunks: List[ExpandedChunk], 
        token_budget: int = 4000
    ) -> List[ExpandedChunk]:
        """
        Fit best context within token budget.
        
        Strategy: Include most relevant chunks fully,
        then truncate less relevant ones.
        """
        # Sort by relevance
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)
        
        selected = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            if current_tokens + chunk.token_count <= token_budget:
                selected.append(chunk)
                current_tokens += chunk.token_count
            elif current_tokens < token_budget * 0.9:
                # Truncate this chunk to fit
                remaining = token_budget - current_tokens
                truncated = self.truncate_chunk(chunk, remaining)
                selected.append(truncated)
                break
        
        return selected
    
    # 3. Embedding caching (huge savings)
    async def get_embedding_with_cache(self, text: str) -> List[float]:
        cache_key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
        
        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)  # Saved $0.00013
        
        # Generate and cache
        embedding = await self.embedding_model.embed(text)
        await self.redis.set(cache_key, json.dumps(embedding), ex=86400*30)
        
        return embedding
```

---

## 12. Interview Discussion Points

### Questions They Might Ask

**Q: Why do you need both vector search AND keyword search?**

> **A:** They solve different problems. Vector search finds semantically similar content — "indemnification" matches "hold harmless" even though words differ. But keyword search is better for: (1) Legal terms of art that must be exact, like "force majeure" or specific section numbers like "Section 7.1"; (2) Defined terms that are capitalized specifically; (3) Rare terms where we don't have good embeddings. Hybrid with RRF fusion gives us best of both.

**Q: How do you handle a query about Section 7.1 when Section 7.1 references Sections 3, 5, and 12?**

> **A:** Context expansion. After retrieving Section 7.1, we parse its cross-references, determine which are relevant to the query (not all references matter), and pull those in. We maintain a reference graph during ingestion so this is fast. We also pull in any definitions used in Section 7.1. The key is being selective — we can't include ALL references or context explodes.

**Q: What's your chunking strategy for a 50-page indemnification section?**

> **A:** Hierarchy-aware splitting. First, try to split at article/section boundaries. If still too big, split at subsection (1.1, 1.2) boundaries. If still too big, split at paragraph markers ((a), (b), (c)). Last resort: sentence-level with overlap. Each chunk keeps metadata about its parent section and siblings. The key insight: keeping a legal clause intact is worth larger chunks — breaking "EXCEPT for..." away from what it excepts destroys meaning.

**Q: How do you prevent the LLM from hallucinating case law or legal conclusions?**

> **A:** Five layers: (1) Retrieval grounding — system prompt forbids outside knowledge; (2) Citation requirements — every claim needs [§X.Y] citation; (3) Post-generation verification — parse citations, check they exist in source; (4) Structured extraction for high-stakes queries — JSON schema is harder to hallucinate than prose; (5) Confidence scoring — low confidence triggers human review. We also fine-tune on "I don't know" examples so the model learns to admit uncertainty.

**Q: How would you handle a query about a contract that was amended three times?**

> **A:** Document relationship modeling. During ingestion, we detect "Amendment to Agreement dated..." and build a document graph. Each amendment knows its parent. For queries, we either: (1) Search the "effective" version (original + all amendments merged), or (2) Search all versions and let the user specify, or (3) Show version history. The retriever understands that Amendment 2 might supersede Section 5 of Amendment 1. This is complex — good follow-up question to ask them how they handle it!

### Questions to Ask Them

1. **"How do you handle conflicting clauses — like when Amendment 2 contradicts the original agreement? Does your system surface both or resolve the conflict?"**

2. **"What's your approach to chunking strategy? Have you experimented with different sizes and found an optimal range for legal documents specifically?"**

3. **"Do you do any fine-tuning on legal documents, or is it all prompt engineering with foundation models? I'd love to hear about evaluation datasets for legal accuracy."**

4. **"For multi-document queries like 'compare indemnification across all vendor contracts' — do you do parallel retrieval then merge, or is there a more sophisticated approach?"**

5. **"How do you measure hallucination rate in production? Do you have human review samples, automated checks, or both?"**

---

## Quick Reference: Key Algorithms

### Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ 1/(k + rank_i(d))  for each ranking list i
```
- k = 60 (typical)
- Combines multiple rankings without score normalization
- Documents appearing in multiple lists get boosted

### Semantic vs Keyword Search Use Cases

| Use Vector Search For | Use Keyword Search For |
|----------------------|------------------------|
| Conceptual queries | Exact phrases |
| Paraphrased content | Section numbers ("7.1") |
| Related concepts | Defined terms (capitalized) |
| Intent matching | Legal terms of art |

### Chunking Decision Tree
```
Section token count:
├── < 512: Keep whole (ideal)
├── 512-2048: Keep whole if legal clause (acceptable)
├── > 2048: Must split
    ├── Has subsections? → Split at subsection boundaries
    ├── Has paragraphs (a,b,c)? → Split at paragraph boundaries  
    └── Neither? → Split at sentences (last resort)
```

---

*This system design is optimized for a 45-60 minute interview. Start with the high-level architecture, then dive into chunking (the most critical piece for legal RAG), then retrieval strategy based on interviewer interest.*
