# System Design: E-Discovery Document Processing System

## Table of Contents
1. [Problem Statement & EDRM Framework](#1-problem-statement--edrm-framework)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Document Ingestion Pipeline](#3-document-ingestion-pipeline)
4. [Deduplication System](#4-deduplication-system)
5. [Relevance Scoring & Prioritization](#5-relevance-scoring--prioritization)
6. [Privilege Review Automation](#6-privilege-review-automation)
7. [Cost Optimization Strategy](#7-cost-optimization-strategy)
8. [Batch Processing Architecture](#8-batch-processing-architecture)
9. [Production & Export System](#9-production--export-system)
10. [Scale & Performance](#10-scale--performance)
11. [Quality Control & Defensibility](#11-quality-control--defensibility)
12. [Interview Discussion Points](#12-interview-discussion-points)

---

## 1. Problem Statement & EDRM Framework

### The E-Discovery Challenge

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    E-DISCOVERY SCALE & COMPLEXITY                            │
└─────────────────────────────────────────────────────────────────────────────┘

TYPICAL DISCOVERY SCENARIO:
───────────────────────────
Litigation: Product liability lawsuit against manufacturer
Discovery Request: "All documents relating to safety testing of Product X"

Data Sources:
• 50 custodians (employees)
• Email servers (Exchange, Gmail)
• File shares (SharePoint, network drives)
• Slack/Teams messages
• Cloud storage (Box, Dropbox)
• Local hard drives

Volume:
• 2 million documents collected
• 500GB of data
• Multiple file types (email, PDF, Word, Excel, images, CAD files)

Requirements:
• Process within 30-day deadline
• Identify ~50,000 relevant documents
• Review for privilege (attorney-client, work product)
• Produce in court-specified format (TIFF + load file)
• Defensible process (can withstand court scrutiny)

COST WITHOUT AI:
─────────────────
• Manual review: $1-3 per document
• 2M documents × $2 = $4,000,000 in review costs
• 6+ months of attorney time

COST WITH AI-ASSISTED REVIEW:
─────────────────────────────
• AI relevance scoring: ~$0.05 per document
• Review only top 10% (200K docs): $400,000
• Time: 4-6 weeks
• 90% cost reduction
```

### EDRM (Electronic Discovery Reference Model)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EDRM FRAMEWORK                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│INFORMATION│  │          │  │          │  │          │  │          │
│GOVERNANCE │──│IDENTIFI- │──│PRESERVA- │──│COLLECTION│──│PROCESSING│
│           │  │CATION    │  │TION      │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └─────┬────┘
                                                              │
        ┌─────────────────────────────────────────────────────┘
        │
        ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│          │  │          │  │          │  │          │
│  REVIEW  │──│ ANALYSIS │──│PRODUCTION│──│PRESENTA- │
│          │  │          │  │          │  │TION      │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

OUR SYSTEM FOCUSES ON:
──────────────────────
✓ PROCESSING   - Ingest, dedupe, extract metadata, OCR
✓ REVIEW       - AI-assisted relevance & privilege review
✓ ANALYSIS     - Clustering, key documents, timeline
✓ PRODUCTION   - Export in court-required formats
```

### Functional Requirements

| Requirement | Description |
|-------------|-------------|
| **Bulk Ingestion** | Handle millions of documents from multiple sources |
| **Format Support** | Email (PST, MBOX, MSG), Office docs, PDFs, images, archives |
| **Metadata Extraction** | Dates, authors, recipients, file properties |
| **Deduplication** | Exact and near-duplicate detection |
| **Full-Text Search** | Fast search across all document content |
| **AI Relevance Scoring** | Classify documents by relevance to discovery request |
| **Privilege Detection** | Identify potentially privileged documents |
| **Batch Review** | Workflows for human review with AI assistance |
| **Production** | Export in TIFF, PDF, native with load files |
| **Audit Trail** | Complete chain of custody and processing logs |

### Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Throughput** | Process 100,000+ documents per hour |
| **Latency** | Interactive search < 2 seconds |
| **Scale** | Handle 10M+ document matters |
| **Accuracy** | >95% recall on relevant documents |
| **Defensibility** | Withstand court challenges on methodology |
| **Cost** | < $0.10 per document average processing cost |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    E-DISCOVERY SYSTEM ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                         DATA SOURCES
    ┌──────────┬──────────┬──────────┬──────────┬──────────┐
    │  Email   │  Files   │  Cloud   │  Slack/  │  Mobile  │
    │  (PST)   │ (Shares) │  (Box)   │  Teams   │  Devices │
    └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘
         │          │          │          │          │
         └──────────┴──────────┼──────────┴──────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Format    │  │  Metadata   │  │    OCR      │  │   Text      │        │
│  │  Converter  │  │  Extractor  │  │   Engine    │  │  Extractor  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DEDUPLICATION ENGINE                              │   │
│  │  • MD5/SHA256 hash (exact)     • SimHash (near-duplicate)           │   │
│  │  • Email threading             • Attachment deduplication           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ANALYSIS ENGINE                                 │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │   Relevance   │  │   Privilege   │  │   Entity      │            │   │
│  │  │   Classifier  │  │   Detector    │  │   Extraction  │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │   Clustering  │  │   Timeline    │  │   Key Doc     │            │   │
│  │  │              │  │   Builder     │  │   Detection   │            │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                           │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   PostgreSQL    │  │  Elasticsearch  │  │    Turbopuffer  │             │
│  │   (Metadata)    │  │  (Full-text)    │  │    (Vectors)    │             │
│  │                 │  │                 │  │                 │             │
│  │  • Doc metadata │  │  • Text search  │  │  • Semantic     │             │
│  │  • Review state │  │  • Facets       │  │    search       │             │
│  │  • Audit logs   │  │  • Aggregations │  │  • Clustering   │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │       S3        │  │     Redis       │                                   │
│  │  (Documents)    │  │   (Cache/Queue) │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                    │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Review UI      │  │  Search API     │  │  Production     │             │
│  │  (React)        │  │  (GraphQL)      │  │  Engine         │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Matter-Based Data Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA MODEL HIERARCHY                                 │
└─────────────────────────────────────────────────────────────────────────────┘

TENANT (Law Firm)
    │
    └── MATTER (Case/Lawsuit)
            │
            ├── CUSTODIAN (Person whose data was collected)
            │       │
            │       └── COLLECTION (Data source snapshot)
            │               │
            │               └── DOCUMENT (Individual file)
            │                       │
            │                       ├── PAGES (For multi-page docs)
            │                       ├── ATTACHMENTS (Email attachments)
            │                       └── METADATA (Extracted properties)
            │
            ├── SEARCH (Saved searches)
            │
            ├── TAG (Review coding)
            │
            ├── PRODUCTION (Export jobs)
            │
            └── PRIVILEGE_LOG (Withheld documents)
```

---

## 3. Document Ingestion Pipeline

### Ingestion Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOCUMENT INGESTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

    Upload (S3, SFTP, Direct)
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: INTAKE                                                             │
│                                                                              │
│  • Virus scan (ClamAV)                                                      │
│  • File type detection (libmagic)                                           │
│  • Archive extraction (ZIP, RAR, 7z, PST, MBOX)                            │
│  • Assign unique DocID (UUID + hash)                                        │
│  • Record chain of custody                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: EXTRACTION                                                         │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Email     │  │   Office    │  │    PDF      │  │   Image     │        │
│  │  Processor  │  │  Processor  │  │  Processor  │  │  Processor  │        │
│  │             │  │             │  │             │  │             │        │
│  │ • Headers   │  │ • Text      │  │ • Text      │  │ • OCR       │        │
│  │ • Body      │  │ • Metadata  │  │ • OCR       │  │ • EXIF      │        │
│  │ • Attach    │  │ • Embedded  │  │ • Metadata  │  │ • Text      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: METADATA ENRICHMENT                                                │
│                                                                              │
│  Standard Metadata:                    Computed Metadata:                   │
│  • Filename, path, extension           • MD5, SHA256 hash                   │
│  • File size, page count               • Word count, language               │
│  • Created, modified, accessed         • Has attachments                    │
│  • Author, last modified by            • Email thread ID                    │
│  • Subject, title                      • Dedup family ID                    │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: TEXT PROCESSING                                                    │
│                                                                              │
│  • Language detection                                                        │
│  • Text normalization (encoding, whitespace)                                │
│  • Named entity extraction (people, orgs, dates, $)                        │
│  • Sensitive data detection (SSN, CC, PHI)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: INDEXING                                                           │
│                                                                              │
│  • Elasticsearch: Full-text index with metadata facets                      │
│  • PostgreSQL: Structured metadata, review state                           │
│  • Turbopuffer: Vector embeddings for semantic search                      │
│  • S3: Native files, extracted text, TIFF conversions                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Format-Specific Processing

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, BinaryIO
from enum import Enum
import hashlib

class FileType(Enum):
    EMAIL_PST = "email_pst"
    EMAIL_MBOX = "email_mbox"
    EMAIL_MSG = "email_msg"
    EMAIL_EML = "email_eml"
    OFFICE_DOC = "office_doc"
    OFFICE_DOCX = "office_docx"
    OFFICE_XLS = "office_xls"
    OFFICE_XLSX = "office_xlsx"
    OFFICE_PPT = "office_ppt"
    OFFICE_PPTX = "office_pptx"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"
    ARCHIVE = "archive"
    OTHER = "other"

@dataclass
class ExtractedDocument:
    """Result of document extraction."""
    doc_id: str
    original_path: str
    file_type: FileType
    
    # Content
    text_content: str
    text_pages: List[str]  # Per-page text for multi-page docs
    
    # Metadata
    metadata: dict
    
    # Hashes for deduplication
    md5_hash: str
    sha256_hash: str
    content_hash: str  # Hash of normalized text (for near-dedup)
    
    # Family relationships
    parent_id: Optional[str] = None  # For attachments
    children: List[str] = None  # Attachment IDs
    
    # Processing info
    ocr_applied: bool = False
    ocr_confidence: float = 1.0
    extraction_errors: List[str] = None


class DocumentProcessor(ABC):
    """Base class for document processors."""
    
    @abstractmethod
    def can_process(self, file_type: FileType) -> bool:
        pass
    
    @abstractmethod
    async def extract(self, file: BinaryIO, metadata: dict) -> ExtractedDocument:
        pass


class EmailProcessor(DocumentProcessor):
    """Process email files (PST, MBOX, MSG, EML)."""
    
    def can_process(self, file_type: FileType) -> bool:
        return file_type in [
            FileType.EMAIL_PST, FileType.EMAIL_MBOX,
            FileType.EMAIL_MSG, FileType.EMAIL_EML
        ]
    
    async def extract(self, file: BinaryIO, metadata: dict) -> List[ExtractedDocument]:
        """
        Extract emails from container.
        
        PST/MBOX can contain thousands of emails - return list.
        """
        file_type = metadata.get("file_type")
        
        if file_type == FileType.EMAIL_PST:
            return await self._extract_pst(file, metadata)
        elif file_type == FileType.EMAIL_MBOX:
            return await self._extract_mbox(file, metadata)
        else:
            return [await self._extract_single_email(file, metadata)]
    
    async def _extract_pst(self, file: BinaryIO, metadata: dict) -> List[ExtractedDocument]:
        """Extract emails from Outlook PST file."""
        import pypff  # Python PST library
        
        documents = []
        pst = pypff.file()
        pst.open_file_object(file)
        
        # Recursively process folders
        async def process_folder(folder, path=""):
            for i in range(folder.number_of_sub_messages):
                message = folder.get_sub_message(i)
                doc = await self._convert_email_to_document(message, metadata, path)
                documents.append(doc)
                
                # Process attachments
                for j in range(message.number_of_attachments):
                    attachment = message.get_attachment(j)
                    attach_doc = await self._process_attachment(
                        attachment, doc.doc_id, metadata
                    )
                    if attach_doc:
                        documents.append(attach_doc)
                        doc.children.append(attach_doc.doc_id)
            
            # Recurse into subfolders
            for i in range(folder.number_of_sub_folders):
                subfolder = folder.get_sub_folder(i)
                await process_folder(subfolder, f"{path}/{subfolder.name}")
        
        await process_folder(pst.root_folder)
        return documents
    
    async def _convert_email_to_document(
        self, 
        message, 
        metadata: dict,
        folder_path: str
    ) -> ExtractedDocument:
        """Convert email message to ExtractedDocument."""
        
        # Extract headers
        headers = {
            "from": message.sender_name or message.sender_email,
            "to": self._parse_recipients(message.get_recipients()),
            "cc": self._parse_recipients(message.get_cc()),
            "bcc": self._parse_recipients(message.get_bcc()),
            "subject": message.subject,
            "sent_date": message.delivery_time,
            "received_date": message.creation_time,
            "message_id": message.transport_message_headers.get("Message-ID"),
            "in_reply_to": message.transport_message_headers.get("In-Reply-To"),
            "thread_id": self._compute_thread_id(message),
        }
        
        # Extract body
        body_text = message.plain_text_body or ""
        if not body_text and message.html_body:
            body_text = self._html_to_text(message.html_body)
        
        # Combine for full text
        full_text = f"""
From: {headers['from']}
To: {headers['to']}
Subject: {headers['subject']}
Date: {headers['sent_date']}

{body_text}
"""
        
        # Compute hashes
        content_bytes = full_text.encode('utf-8')
        
        return ExtractedDocument(
            doc_id=self._generate_doc_id(),
            original_path=f"{metadata.get('source_path')}{folder_path}",
            file_type=FileType.EMAIL_MSG,
            text_content=full_text,
            text_pages=[full_text],
            metadata={
                **headers,
                "folder_path": folder_path,
                "attachment_count": message.number_of_attachments,
            },
            md5_hash=hashlib.md5(content_bytes).hexdigest(),
            sha256_hash=hashlib.sha256(content_bytes).hexdigest(),
            content_hash=self._compute_content_hash(body_text),
            children=[],
        )
    
    def _compute_thread_id(self, message) -> str:
        """
        Compute email thread ID for conversation threading.
        
        Use: Subject (normalized) + participants
        """
        subject = message.subject or ""
        # Normalize subject (remove Re:, Fwd:, etc.)
        normalized_subject = self._normalize_subject(subject)
        
        # Get all participants
        participants = set()
        participants.add(message.sender_email)
        for r in message.get_recipients():
            participants.add(r.email)
        
        # Sort for consistency
        sorted_participants = sorted(participants)
        
        # Hash
        thread_string = f"{normalized_subject}|{'|'.join(sorted_participants)}"
        return hashlib.md5(thread_string.encode()).hexdigest()


class PDFProcessor(DocumentProcessor):
    """Process PDF files with OCR fallback."""
    
    def can_process(self, file_type: FileType) -> bool:
        return file_type == FileType.PDF
    
    async def extract(self, file: BinaryIO, metadata: dict) -> ExtractedDocument:
        import pymupdf  # PyMuPDF
        
        doc = pymupdf.open(stream=file.read(), filetype="pdf")
        
        pages_text = []
        needs_ocr = False
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if len(text.strip()) < 50:
                # Likely scanned - needs OCR
                needs_ocr = True
                text = await self._ocr_page(page)
            
            pages_text.append(text)
        
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
        
        # Extract PDF metadata
        pdf_metadata = {
            "title": doc.metadata.get("title"),
            "author": doc.metadata.get("author"),
            "creator": doc.metadata.get("creator"),
            "producer": doc.metadata.get("producer"),
            "creation_date": doc.metadata.get("creationDate"),
            "modification_date": doc.metadata.get("modDate"),
            "page_count": len(doc),
        }
        
        content_bytes = full_text.encode('utf-8')
        
        return ExtractedDocument(
            doc_id=self._generate_doc_id(),
            original_path=metadata.get("source_path"),
            file_type=FileType.PDF,
            text_content=full_text,
            text_pages=pages_text,
            metadata=pdf_metadata,
            md5_hash=hashlib.md5(content_bytes).hexdigest(),
            sha256_hash=hashlib.sha256(content_bytes).hexdigest(),
            content_hash=self._compute_content_hash(full_text),
            ocr_applied=needs_ocr,
        )
    
    async def _ocr_page(self, page) -> str:
        """OCR a single PDF page."""
        # Render page to image
        pix = page.get_pixmap(dpi=300)
        image_bytes = pix.tobytes("png")
        
        # Send to OCR service (Azure Document AI)
        result = await self.ocr_service.recognize(image_bytes)
        
        return result.text


class IngestionOrchestrator:
    """
    Orchestrate document ingestion at scale.
    """
    
    def __init__(
        self,
        processors: List[DocumentProcessor],
        queue: 'TaskQueue',
        storage: 'DocumentStorage',
        index: 'SearchIndex',
    ):
        self.processors = {p: p for p in processors}
        self.queue = queue
        self.storage = storage
        self.index = index
    
    async def ingest_collection(
        self,
        matter_id: str,
        custodian_id: str,
        source_path: str,
        options: dict = None,
    ) -> 'IngestionJob':
        """
        Ingest a collection of documents.
        
        Creates async job that processes in parallel.
        """
        job = IngestionJob(
            matter_id=matter_id,
            custodian_id=custodian_id,
            source_path=source_path,
            options=options or {},
        )
        
        # Enumerate files and queue for processing
        files = await self._enumerate_files(source_path)
        
        for file_info in files:
            await self.queue.enqueue(
                task_type="process_document",
                payload={
                    "job_id": job.id,
                    "file_path": file_info.path,
                    "file_type": file_info.detected_type,
                    "matter_id": matter_id,
                    "custodian_id": custodian_id,
                },
                priority=self._calculate_priority(file_info),
            )
        
        job.total_files = len(files)
        await job.save()
        
        return job
    
    async def process_document(self, task: dict):
        """
        Process a single document (called by worker).
        """
        file_path = task["file_path"]
        file_type = FileType(task["file_type"])
        
        # Find appropriate processor
        processor = self._get_processor(file_type)
        
        # Download file
        file_data = await self.storage.download(file_path)
        
        # Extract content
        documents = await processor.extract(
            file_data,
            metadata={
                "source_path": file_path,
                "file_type": file_type,
                "matter_id": task["matter_id"],
                "custodian_id": task["custodian_id"],
            }
        )
        
        # Store and index each document
        for doc in documents:
            # Store raw file and extracted data
            await self.storage.store_document(doc)
            
            # Index for search
            await self.index.index_document(doc)
            
            # Queue for analysis (dedup, relevance, etc.)
            await self.queue.enqueue(
                task_type="analyze_document",
                payload={"doc_id": doc.doc_id},
            )
```

---

## 4. Deduplication System

### Deduplication Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DEDUPLICATION STRATEGY                                  │
└─────────────────────────────────────────────────────────────────────────────┘

WHY DEDUPE MATTERS:
───────────────────
• Same document collected from multiple custodians
• Email sent to 10 people = 10 copies
• Documents saved to multiple locations
• Without dedupe: Review same doc 10x = 10x cost

DEDUPLICATION LEVELS:

LEVEL 1: EXACT DUPLICATE (MD5/SHA256)
─────────────────────────────────────
• Byte-for-byte identical files
• Fast: O(1) hash lookup
• Catches: Same file, multiple locations

Example:
  /user1/docs/contract.pdf  →  MD5: abc123
  /user2/docs/contract.pdf  →  MD5: abc123
  → EXACT DUPLICATE


LEVEL 2: NEAR DUPLICATE (SimHash/MinHash)
─────────────────────────────────────────
• Similar content, different formatting
• Catches: Minor edits, formatting changes, OCR variations

Example:
  Document A: "The quick brown fox jumps over the lazy dog."
  Document B: "The quick brown fox jumps over the lazy dog"
  → NEAR DUPLICATE (missing period)


LEVEL 3: EMAIL THREADING
────────────────────────
• Group emails in same conversation
• Parent email contains all child content
• Review only latest in thread

Example:
  Email 1: "Let's discuss the contract"
  Email 2: RE: "Let's discuss the contract" + "Sure, when?"
  Email 3: RE: RE: "Let's discuss the contract" + "Sure, when?" + "Tomorrow"
  → Email 3 contains all content, suppress 1 & 2


LEVEL 4: FAMILY DEDUPLICATION
─────────────────────────────
• Email + attachments as family
• Dedupe attachment across families
• Keep attachment with earliest email

Example:
  Email A (Jan 1) → attachment.pdf (MD5: xyz)
  Email B (Jan 5) → attachment.pdf (MD5: xyz)
  → Keep attachment with Email A, link from Email B
```

### Deduplication Implementation

```python
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
from collections import defaultdict
import hashlib
import re

@dataclass
class DuplicateGroup:
    """Group of duplicate documents."""
    group_id: str
    master_doc_id: str  # Primary document to keep
    duplicate_doc_ids: List[str]  # Suppressed duplicates
    duplicate_type: str  # "exact", "near", "email_thread", "family"
    similarity_score: float  # 1.0 for exact, <1.0 for near


class DeduplicationEngine:
    """
    Multi-level deduplication engine.
    """
    
    def __init__(self, db, config: dict = None):
        self.db = db
        self.config = config or {}
        self.similarity_threshold = self.config.get("similarity_threshold", 0.95)
    
    async def deduplicate_matter(self, matter_id: str) -> Dict[str, DuplicateGroup]:
        """
        Run full deduplication on a matter.
        """
        # Get all documents
        documents = await self.db.get_documents(matter_id)
        
        duplicate_groups = {}
        
        # Level 1: Exact duplicates (fast)
        exact_groups = await self._find_exact_duplicates(documents)
        duplicate_groups.update(exact_groups)
        
        # Level 2: Near duplicates (slower)
        near_groups = await self._find_near_duplicates(documents, exact_groups)
        duplicate_groups.update(near_groups)
        
        # Level 3: Email threading
        email_threads = await self._build_email_threads(documents)
        duplicate_groups.update(email_threads)
        
        # Level 4: Family deduplication
        family_groups = await self._dedupe_families(documents)
        duplicate_groups.update(family_groups)
        
        # Save deduplication results
        await self._save_dedup_results(matter_id, duplicate_groups)
        
        return duplicate_groups
    
    async def _find_exact_duplicates(
        self, 
        documents: List[ExtractedDocument]
    ) -> Dict[str, DuplicateGroup]:
        """
        Find exact duplicates using hash.
        
        O(n) - single pass with hash table.
        """
        hash_to_docs = defaultdict(list)
        
        for doc in documents:
            # Use MD5 for speed (SHA256 for verification)
            hash_to_docs[doc.md5_hash].append(doc)
        
        groups = {}
        for hash_val, docs in hash_to_docs.items():
            if len(docs) > 1:
                # Sort by date to pick earliest as master
                docs.sort(key=lambda d: d.metadata.get("date_created") or "9999")
                master = docs[0]
                duplicates = docs[1:]
                
                group = DuplicateGroup(
                    group_id=f"exact_{hash_val[:8]}",
                    master_doc_id=master.doc_id,
                    duplicate_doc_ids=[d.doc_id for d in duplicates],
                    duplicate_type="exact",
                    similarity_score=1.0,
                )
                groups[group.group_id] = group
        
        return groups
    
    async def _find_near_duplicates(
        self,
        documents: List[ExtractedDocument],
        exact_groups: Dict[str, DuplicateGroup],
    ) -> Dict[str, DuplicateGroup]:
        """
        Find near-duplicates using SimHash.
        
        SimHash: Locality-sensitive hash that produces similar
        hashes for similar documents.
        """
        # Skip documents already in exact duplicate groups
        exact_dup_ids = set()
        for group in exact_groups.values():
            exact_dup_ids.update(group.duplicate_doc_ids)
        
        candidates = [d for d in documents if d.doc_id not in exact_dup_ids]
        
        # Compute SimHash for each document
        simhashes = {}
        for doc in candidates:
            simhashes[doc.doc_id] = self._compute_simhash(doc.text_content)
        
        # Find similar pairs using hamming distance
        groups = {}
        processed = set()
        
        for doc in candidates:
            if doc.doc_id in processed:
                continue
            
            similar_docs = []
            doc_hash = simhashes[doc.doc_id]
            
            for other_doc in candidates:
                if other_doc.doc_id == doc.doc_id or other_doc.doc_id in processed:
                    continue
                
                other_hash = simhashes[other_doc.doc_id]
                similarity = self._simhash_similarity(doc_hash, other_hash)
                
                if similarity >= self.similarity_threshold:
                    similar_docs.append((other_doc, similarity))
                    processed.add(other_doc.doc_id)
            
            if similar_docs:
                processed.add(doc.doc_id)
                
                group = DuplicateGroup(
                    group_id=f"near_{doc.doc_id[:8]}",
                    master_doc_id=doc.doc_id,
                    duplicate_doc_ids=[d.doc_id for d, _ in similar_docs],
                    duplicate_type="near",
                    similarity_score=sum(s for _, s in similar_docs) / len(similar_docs),
                )
                groups[group.group_id] = group
        
        return groups
    
    def _compute_simhash(self, text: str, hash_bits: int = 128) -> int:
        """
        Compute SimHash of text.
        
        Algorithm:
        1. Tokenize text into features (words, n-grams)
        2. Hash each feature
        3. For each bit position, sum +1 if bit is 1, -1 if bit is 0
        4. Final hash: bit is 1 if sum > 0, else 0
        """
        # Tokenize
        tokens = self._tokenize(text)
        
        # Initialize bit counts
        bit_counts = [0] * hash_bits
        
        for token in tokens:
            # Hash token
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # Update bit counts
            for i in range(hash_bits):
                if (token_hash >> i) & 1:
                    bit_counts[i] += 1
                else:
                    bit_counts[i] -= 1
        
        # Generate final hash
        simhash = 0
        for i in range(hash_bits):
            if bit_counts[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _simhash_similarity(self, hash1: int, hash2: int) -> float:
        """
        Compute similarity between two SimHashes.
        
        Uses Hamming distance (number of differing bits).
        """
        # XOR gives bits that differ
        diff = hash1 ^ hash2
        
        # Count set bits (Hamming weight)
        hamming_distance = bin(diff).count('1')
        
        # Convert to similarity (0-1)
        hash_bits = 128
        similarity = 1 - (hamming_distance / hash_bits)
        
        return similarity
    
    async def _build_email_threads(
        self,
        documents: List[ExtractedDocument],
    ) -> Dict[str, DuplicateGroup]:
        """
        Group emails into threads and suppress earlier messages.
        
        Logic: The most recent email in a thread contains the full
        conversation history (quoted text). Review only the latest.
        """
        # Filter to emails only
        emails = [d for d in documents if d.file_type in [
            FileType.EMAIL_MSG, FileType.EMAIL_EML
        ]]
        
        # Group by thread ID
        threads = defaultdict(list)
        for email in emails:
            thread_id = email.metadata.get("thread_id")
            if thread_id:
                threads[thread_id].append(email)
        
        groups = {}
        for thread_id, thread_emails in threads.items():
            if len(thread_emails) > 1:
                # Sort by date, newest first
                thread_emails.sort(
                    key=lambda e: e.metadata.get("sent_date") or "",
                    reverse=True
                )
                
                master = thread_emails[0]  # Most recent
                older = thread_emails[1:]   # Suppress these
                
                group = DuplicateGroup(
                    group_id=f"thread_{thread_id[:8]}",
                    master_doc_id=master.doc_id,
                    duplicate_doc_ids=[e.doc_id for e in older],
                    duplicate_type="email_thread",
                    similarity_score=1.0,  # Same thread = logically duplicate
                )
                groups[group.group_id] = group
        
        return groups
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for SimHash."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\w+', text)
        
        # Generate word n-grams (for better locality sensitivity)
        ngrams = []
        for i in range(len(words) - 2):
            ngram = " ".join(words[i:i+3])
            ngrams.append(ngram)
        
        return ngrams
```

---

## 5. Relevance Scoring & Prioritization

### Relevance Scoring Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RELEVANCE SCORING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

Discovery Request: "All documents relating to safety testing of Product X"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: KEYWORD FILTERING (Fast, Cheap)                                     │
│                                                                              │
│  Boolean Query:                                                              │
│  (safety OR test* OR QA OR quality) AND ("Product X" OR "PX-2000")          │
│                                                                              │
│  Result: 500,000 → 150,000 documents (70% filtered out)                     │
│  Cost: ~$0 (Elasticsearch)                                                   │
│  Time: Seconds                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: ML CLASSIFICATION (Medium Speed, Low Cost)                          │
│                                                                              │
│  Fine-tuned classifier on discovery-request-specific training data          │
│  Model: BERT-base, trained on 1000 example documents                        │
│                                                                              │
│  Output: Relevance probability (0.0 - 1.0)                                  │
│                                                                              │
│  Result: 150,000 → 40,000 documents (score > 0.5)                           │
│  Cost: ~$0.001 per doc = $150                                               │
│  Time: Hours (batch)                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: LLM ANALYSIS (Slow, Expensive, High Accuracy)                       │
│                                                                              │
│  For borderline cases (score 0.3 - 0.7) or high-priority docs              │
│                                                                              │
│  GPT-4 prompt:                                                               │
│  "Is this document relevant to: safety testing of Product X?                │
│   Consider: direct mentions, indirect references, related topics            │
│   Document text: {text}                                                     │
│   Respond: RELEVANT / NOT_RELEVANT / UNCERTAIN + explanation"              │
│                                                                              │
│  Result: 40,000 → 35,000 relevant                                           │
│  Cost: ~$0.02 per doc = $800                                                │
│  Time: Days (rate-limited)                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONTINUOUS ACTIVE LEARNING                                                  │
│                                                                              │
│  • Human reviewers code documents (relevant/not relevant)                   │
│  • Feedback improves ML model in real-time                                  │
│  • System re-ranks remaining documents                                      │
│  • Prioritize docs most likely to be relevant but not yet reviewed         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology-Assisted Review (TAR) Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RelevanceLabel(Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNCERTAIN = "uncertain"
    NOT_REVIEWED = "not_reviewed"

@dataclass
class ScoredDocument:
    doc_id: str
    relevance_score: float  # 0.0 - 1.0
    relevance_label: RelevanceLabel
    score_source: str  # "keyword", "ml", "llm", "human"
    confidence: float


class TAREngine:
    """
    Technology-Assisted Review (TAR) / Predictive Coding.
    
    Implements CAL (Continuous Active Learning) for iterative
    relevance scoring with human feedback.
    """
    
    def __init__(
        self,
        vectorizer_config: dict = None,
        llm_client = None,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.model = LogisticRegression(class_weight='balanced')
        self.llm = llm_client
        self.is_trained = False
        
        # Store labeled examples for retraining
        self.labeled_docs: List[Tuple[str, RelevanceLabel]] = []
    
    async def initialize_from_request(
        self,
        discovery_request: str,
        seed_documents: List[dict] = None,
    ):
        """
        Initialize TAR from discovery request text.
        
        Creates initial keyword query and optionally uses
        seed documents for first model training.
        """
        self.discovery_request = discovery_request
        
        # Extract key terms from request
        self.key_terms = await self._extract_key_terms(discovery_request)
        
        # Build initial keyword query
        self.keyword_query = self._build_keyword_query(self.key_terms)
        
        # If seed documents provided, train initial model
        if seed_documents:
            await self._train_on_seeds(seed_documents)
    
    async def score_documents(
        self,
        documents: List[ExtractedDocument],
        use_tiers: bool = True,
    ) -> List[ScoredDocument]:
        """
        Score documents for relevance using tiered approach.
        """
        scored = []
        
        for doc in documents:
            # Tier 1: Keyword match
            keyword_score = self._keyword_score(doc)
            
            if keyword_score < 0.1:
                # Very unlikely relevant, skip expensive scoring
                scored.append(ScoredDocument(
                    doc_id=doc.doc_id,
                    relevance_score=keyword_score,
                    relevance_label=RelevanceLabel.NOT_REVIEWED,
                    score_source="keyword",
                    confidence=0.8,
                ))
                continue
            
            # Tier 2: ML classification (if model trained)
            if self.is_trained:
                ml_score = await self._ml_score(doc)
                
                # High confidence? Use ML score
                if ml_score > 0.8 or ml_score < 0.2:
                    scored.append(ScoredDocument(
                        doc_id=doc.doc_id,
                        relevance_score=ml_score,
                        relevance_label=self._score_to_label(ml_score),
                        score_source="ml",
                        confidence=abs(ml_score - 0.5) * 2,
                    ))
                    continue
            
            # Tier 3: LLM for uncertain cases (expensive)
            if use_tiers and self.llm:
                llm_result = await self._llm_score(doc)
                scored.append(ScoredDocument(
                    doc_id=doc.doc_id,
                    relevance_score=llm_result.score,
                    relevance_label=llm_result.label,
                    score_source="llm",
                    confidence=llm_result.confidence,
                ))
            else:
                # Default to keyword score
                scored.append(ScoredDocument(
                    doc_id=doc.doc_id,
                    relevance_score=keyword_score,
                    relevance_label=RelevanceLabel.NOT_REVIEWED,
                    score_source="keyword",
                    confidence=0.5,
                ))
        
        return scored
    
    def _keyword_score(self, doc: ExtractedDocument) -> float:
        """
        Score based on keyword matches.
        
        Fast, cheap, but low precision.
        """
        text = doc.text_content.lower()
        
        matches = 0
        total_weight = 0
        
        for term, weight in self.key_terms.items():
            if term.lower() in text:
                matches += weight
            total_weight += weight
        
        return matches / total_weight if total_weight > 0 else 0
    
    async def _ml_score(self, doc: ExtractedDocument) -> float:
        """
        Score using trained ML model.
        """
        # Vectorize document
        vector = self.vectorizer.transform([doc.text_content])
        
        # Get probability of relevance
        proba = self.model.predict_proba(vector)[0]
        
        # Return probability of "relevant" class
        relevant_idx = list(self.model.classes_).index(RelevanceLabel.RELEVANT.value)
        
        return proba[relevant_idx]
    
    async def _llm_score(self, doc: ExtractedDocument) -> 'LLMRelevanceResult':
        """
        Score using LLM for complex/uncertain cases.
        """
        prompt = f"""Analyze this document for relevance to the following discovery request.

DISCOVERY REQUEST:
{self.discovery_request}

DOCUMENT TEXT:
{doc.text_content[:4000]}  # Truncate for token limits

INSTRUCTIONS:
1. Determine if this document is RELEVANT, NOT_RELEVANT, or UNCERTAIN
2. Consider direct mentions, indirect references, and contextually related content
3. Provide brief reasoning

Respond in JSON format:
{{
    "relevance": "RELEVANT" | "NOT_RELEVANT" | "UNCERTAIN",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_phrases": ["phrases that indicate relevance or irrelevance"]
}}
"""
        
        response = await self.llm.chat(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response)
        
        label_map = {
            "RELEVANT": RelevanceLabel.RELEVANT,
            "NOT_RELEVANT": RelevanceLabel.NOT_RELEVANT,
            "UNCERTAIN": RelevanceLabel.UNCERTAIN,
        }
        
        return LLMRelevanceResult(
            label=label_map.get(result["relevance"], RelevanceLabel.UNCERTAIN),
            score=1.0 if result["relevance"] == "RELEVANT" else 0.0,
            confidence=result.get("confidence", 0.5),
            reasoning=result.get("reasoning", ""),
        )
    
    async def incorporate_feedback(
        self,
        doc_id: str,
        human_label: RelevanceLabel,
        reviewer_id: str,
    ):
        """
        Incorporate human review feedback for active learning.
        
        This is key to TAR: human labels improve the model,
        model prioritizes next documents for review.
        """
        # Get document
        doc = await self.db.get_document(doc_id)
        
        # Store labeled example
        self.labeled_docs.append((doc.text_content, human_label))
        
        # Retrain model periodically
        if len(self.labeled_docs) % 50 == 0:
            await self._retrain_model()
        
        # Re-rank remaining unreviewed documents
        await self._update_review_queue()
    
    async def _retrain_model(self):
        """Retrain ML model with new labeled data."""
        if len(self.labeled_docs) < 20:
            return  # Need minimum samples
        
        texts = [text for text, _ in self.labeled_docs]
        labels = [label.value for _, label in self.labeled_docs]
        
        # Fit vectorizer on all text
        X = self.vectorizer.fit_transform(texts)
        
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
    
    async def get_next_review_batch(
        self,
        batch_size: int = 50,
        strategy: str = "uncertainty",
    ) -> List[str]:
        """
        Get next batch of documents for human review.
        
        Strategies:
        - uncertainty: Documents model is least certain about
        - relevant_first: Highest predicted relevance
        - diverse: Cluster-based sampling for coverage
        """
        # Get unreviewed documents
        unreviewed = await self.db.get_unreviewed_documents()
        
        if strategy == "uncertainty":
            # Score and sort by uncertainty (closest to 0.5)
            scored = await self.score_documents(unreviewed)
            scored.sort(key=lambda s: abs(s.relevance_score - 0.5))
            return [s.doc_id for s in scored[:batch_size]]
        
        elif strategy == "relevant_first":
            scored = await self.score_documents(unreviewed)
            scored.sort(key=lambda s: s.relevance_score, reverse=True)
            return [s.doc_id for s in scored[:batch_size]]
        
        elif strategy == "diverse":
            return await self._diverse_sampling(unreviewed, batch_size)
    
    async def calculate_recall_estimate(self) -> dict:
        """
        Estimate recall (% of relevant docs found) for defensibility.
        
        Uses statistical sampling to estimate remaining relevant docs.
        """
        # Random sample of unreviewed documents
        sample = await self.db.get_random_unreviewed(n=100)
        
        # Have humans review the sample
        # (In practice, this is done offline)
        
        # Calculate estimates
        reviewed_relevant = await self.db.count_relevant_reviewed()
        total_relevant_estimate = reviewed_relevant / self.recall_rate
        
        return {
            "reviewed_relevant": reviewed_relevant,
            "estimated_total_relevant": total_relevant_estimate,
            "estimated_recall": reviewed_relevant / total_relevant_estimate,
            "confidence_interval": self._calculate_confidence_interval(),
        }
```

### Relevance Scoring Prompt Engineering

```python
class RelevancePromptBuilder:
    """
    Build effective prompts for LLM relevance scoring.
    """
    
    SYSTEM_PROMPT = """You are a legal document analyst assisting with e-discovery review.

Your task is to determine if documents are relevant to a discovery request.

IMPORTANT GUIDELINES:
1. "Relevant" means the document contains information that could be useful to either party in litigation
2. Consider both direct and indirect relevance
3. Err on the side of inclusion - it's better to flag something for human review than miss it
4. Look for:
   - Direct mentions of key terms, products, people, or events
   - Discussions of related topics that provide context
   - Documents that might lead to other relevant evidence
   - Timeline information related to the matter

5. Do NOT consider:
   - Whether the document helps or hurts either party (that's not your job)
   - Privilege (that's a separate analysis)
   - Admissibility (that's a legal question)

You are helping prioritize documents for human review, not making final determinations."""

    def build_relevance_prompt(
        self,
        discovery_request: str,
        document_text: str,
        document_metadata: dict,
        examples: List[dict] = None,
    ) -> str:
        """
        Build prompt for relevance determination.
        """
        prompt_parts = []
        
        # Add discovery request context
        prompt_parts.append(f"""
DISCOVERY REQUEST:
{discovery_request}

KEY CONCEPTS TO LOOK FOR:
{self._extract_key_concepts(discovery_request)}
""")
        
        # Add few-shot examples if available
        if examples:
            prompt_parts.append("\nEXAMPLES OF RELEVANCE DETERMINATIONS:")
            for ex in examples[:3]:
                prompt_parts.append(f"""
---
Document excerpt: "{ex['excerpt'][:500]}..."
Determination: {ex['label']}
Reasoning: {ex['reasoning']}
---
""")
        
        # Add document to analyze
        prompt_parts.append(f"""
DOCUMENT TO ANALYZE:
Type: {document_metadata.get('file_type', 'Unknown')}
Date: {document_metadata.get('date', 'Unknown')}
Author: {document_metadata.get('author', 'Unknown')}
Subject: {document_metadata.get('subject', 'N/A')}

TEXT:
{document_text[:6000]}

{"[Document truncated due to length]" if len(document_text) > 6000 else ""}

ANALYSIS REQUIRED:
1. Is this document RELEVANT, NOT_RELEVANT, or UNCERTAIN to the discovery request?
2. What specific content supports your determination?
3. What is your confidence level (0-100%)?

Respond in JSON format:
{{
    "determination": "RELEVANT" | "NOT_RELEVANT" | "UNCERTAIN",
    "confidence": 0-100,
    "supporting_content": ["quote or description of relevant content"],
    "reasoning": "brief explanation"
}}
""")
        
        return "\n".join(prompt_parts)
```

---

## 6. Privilege Review Automation

### Privilege Detection Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRIVILEGE REVIEW AUTOMATION                               │
└─────────────────────────────────────────────────────────────────────────────┘

TYPES OF PRIVILEGE:
───────────────────
1. ATTORNEY-CLIENT PRIVILEGE
   - Communications between client and attorney
   - For purpose of obtaining legal advice
   - Intended to be confidential

2. WORK PRODUCT DOCTRINE
   - Materials prepared in anticipation of litigation
   - By or for attorney
   - Includes mental impressions, legal theories

3. JOINT DEFENSE / COMMON INTEREST
   - Communications with co-defendants
   - Shared legal interest

DETECTION APPROACH:

┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PRIVILEGE INDICATORS                                               │
│                                                                              │
│  Check for:                                                                  │
│  • Attorney names/email addresses in participants                           │
│  • Law firm domains (@lawfirm.com)                                         │
│  • Legal terms: "privileged", "attorney-client", "legal advice"            │
│  • "Privileged and Confidential" header/footer                             │
│                                                                              │
│  Output: privilege_indicator_score (0-1)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: ATTORNEY IDENTIFICATION                                            │
│                                                                              │
│  Maintain lists:                                                             │
│  • Known attorneys (from client intake)                                     │
│  • Law firm email domains                                                   │
│  • Bar registration lookups (external API)                                  │
│                                                                              │
│  Match against participants (from, to, cc, bcc)                            │
│  Output: attorney_involvement (boolean + details)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONTENT ANALYSIS (LLM)                                             │
│                                                                              │
│  For high-indicator documents:                                               │
│  • Analyze if communication is seeking/providing legal advice               │
│  • Check if work product (litigation preparation)                          │
│  • Identify potential waiver (third-party disclosure)                      │
│                                                                              │
│  Output: privilege_classification + reasoning                               │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: HUMAN REVIEW QUEUE                                                 │
│                                                                              │
│  • AI can FLAG but not DECIDE privilege                                     │
│  • All potentially privileged docs require attorney review                  │
│  • Generate privilege log entries for withheld docs                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Privilege Detection Implementation

```python
from dataclasses import dataclass
from typing import List, Set, Optional
from enum import Enum

class PrivilegeType(Enum):
    ATTORNEY_CLIENT = "attorney_client"
    WORK_PRODUCT = "work_product"
    JOINT_DEFENSE = "joint_defense"
    NOT_PRIVILEGED = "not_privileged"
    NEEDS_REVIEW = "needs_review"

@dataclass
class PrivilegeAnalysis:
    doc_id: str
    privilege_type: PrivilegeType
    confidence: float
    
    # Supporting evidence
    attorney_participants: List[str]
    privilege_indicators: List[str]
    content_analysis: str
    
    # Risk flags
    potential_waiver: bool
    waiver_reason: Optional[str]
    
    # For privilege log
    privilege_basis: str  # Legal basis for privilege claim
    redaction_needed: bool


class PrivilegeDetector:
    """
    Automated privilege detection with human-in-the-loop.
    """
    
    # Known attorney email patterns
    ATTORNEY_PATTERNS = [
        r'@.*law\.com$',
        r'@.*legal\.com$',
        r'@.*llp\.com$',
        r'@.*attorney.*\.com$',
    ]
    
    # Privilege indicator phrases
    PRIVILEGE_INDICATORS = [
        "attorney-client privileged",
        "privileged and confidential",
        "attorney work product",
        "legal advice",
        "litigation hold",
        "in anticipation of litigation",
        "do not forward",
        "confidential communication",
    ]
    
    # Phrases that might indicate waiver
    WAIVER_INDICATORS = [
        "forwarded to",
        "cc'd external",
        "feel free to share",
        "posted to",
        "published",
    ]
    
    def __init__(
        self,
        attorney_list: List[dict],  # Known attorneys
        law_firm_domains: List[str],
        llm_client,
    ):
        self.attorneys = {a['email'].lower(): a for a in attorney_list}
        self.law_firm_domains = set(d.lower() for d in law_firm_domains)
        self.llm = llm_client
    
    async def analyze_document(
        self,
        doc: ExtractedDocument,
    ) -> PrivilegeAnalysis:
        """
        Analyze document for privilege.
        """
        # Stage 1: Check indicators
        indicators = self._find_privilege_indicators(doc)
        indicator_score = len(indicators) / 10  # Normalize
        
        # Stage 2: Check for attorney involvement
        attorney_participants = self._find_attorney_participants(doc)
        
        # Stage 3: Content analysis (if indicators present)
        content_analysis = ""
        privilege_type = PrivilegeType.NOT_PRIVILEGED
        confidence = 0.0
        
        if indicator_score > 0.1 or attorney_participants:
            # Use LLM for deeper analysis
            llm_result = await self._llm_privilege_analysis(doc, indicators, attorney_participants)
            content_analysis = llm_result.get("analysis", "")
            privilege_type = self._map_privilege_type(llm_result.get("privilege_type"))
            confidence = llm_result.get("confidence", 0.5)
        
        # Stage 4: Check for potential waiver
        waiver_check = self._check_waiver(doc)
        
        return PrivilegeAnalysis(
            doc_id=doc.doc_id,
            privilege_type=privilege_type if confidence > 0.7 else PrivilegeType.NEEDS_REVIEW,
            confidence=confidence,
            attorney_participants=attorney_participants,
            privilege_indicators=indicators,
            content_analysis=content_analysis,
            potential_waiver=waiver_check.has_waiver_risk,
            waiver_reason=waiver_check.reason,
            privilege_basis=self._generate_privilege_basis(privilege_type, llm_result),
            redaction_needed=self._needs_redaction(privilege_type, doc),
        )
    
    def _find_privilege_indicators(self, doc: ExtractedDocument) -> List[str]:
        """Find privilege-indicating phrases in document."""
        text = doc.text_content.lower()
        found = []
        
        for indicator in self.PRIVILEGE_INDICATORS:
            if indicator.lower() in text:
                found.append(indicator)
        
        return found
    
    def _find_attorney_participants(self, doc: ExtractedDocument) -> List[str]:
        """Find attorneys in document participants."""
        attorneys_found = []
        
        # Check email participants
        participants = []
        if doc.metadata.get("from"):
            participants.append(doc.metadata["from"])
        participants.extend(doc.metadata.get("to", []))
        participants.extend(doc.metadata.get("cc", []))
        
        for participant in participants:
            email = participant.lower() if isinstance(participant, str) else participant.get("email", "").lower()
            
            # Check known attorney list
            if email in self.attorneys:
                attorneys_found.append(email)
                continue
            
            # Check law firm domains
            domain = email.split("@")[-1] if "@" in email else ""
            if domain in self.law_firm_domains:
                attorneys_found.append(email)
                continue
            
            # Check attorney patterns
            import re
            for pattern in self.ATTORNEY_PATTERNS:
                if re.match(pattern, email):
                    attorneys_found.append(email)
                    break
        
        return attorneys_found
    
    async def _llm_privilege_analysis(
        self,
        doc: ExtractedDocument,
        indicators: List[str],
        attorneys: List[str],
    ) -> dict:
        """
        Use LLM for detailed privilege analysis.
        """
        prompt = f"""Analyze this document for attorney-client privilege and work product protection.

CONTEXT:
- Privilege indicators found: {indicators}
- Potential attorneys in participants: {attorneys}

DOCUMENT:
Type: {doc.metadata.get('file_type')}
From: {doc.metadata.get('from')}
To: {doc.metadata.get('to')}
Subject: {doc.metadata.get('subject')}
Date: {doc.metadata.get('date')}

Text:
{doc.text_content[:4000]}

ANALYSIS REQUIRED:
1. Is this communication between a client and their attorney?
2. Is it for the purpose of seeking or providing legal advice?
3. Was it intended to be confidential?
4. Is this work product prepared in anticipation of litigation?
5. Are there any third parties that might waive privilege?

Respond in JSON:
{{
    "privilege_type": "attorney_client" | "work_product" | "joint_defense" | "not_privileged",
    "confidence": 0.0-1.0,
    "analysis": "detailed reasoning",
    "is_seeking_legal_advice": true/false,
    "is_confidential": true/false,
    "third_party_risk": true/false,
    "privilege_basis": "legal basis for privilege claim if applicable"
}}
"""
        
        response = await self.llm.chat(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        
        return json.loads(response)
    
    def _check_waiver(self, doc: ExtractedDocument) -> 'WaiverCheck':
        """Check for potential privilege waiver."""
        text = doc.text_content.lower()
        
        for indicator in self.WAIVER_INDICATORS:
            if indicator in text:
                return WaiverCheck(
                    has_waiver_risk=True,
                    reason=f"Document contains '{indicator}' which may indicate disclosure to third party"
                )
        
        # Check for external recipients
        to_list = doc.metadata.get("to", [])
        cc_list = doc.metadata.get("cc", [])
        
        for recipient in to_list + cc_list:
            email = recipient if isinstance(recipient, str) else recipient.get("email", "")
            domain = email.split("@")[-1] if "@" in email else ""
            
            # If recipient is not from client or law firm, potential waiver
            if domain and domain not in self.law_firm_domains:
                # Check if it's a known client domain
                if not self._is_client_domain(domain):
                    return WaiverCheck(
                        has_waiver_risk=True,
                        reason=f"External recipient ({email}) may waive privilege"
                    )
        
        return WaiverCheck(has_waiver_risk=False, reason=None)


class PrivilegeLogGenerator:
    """
    Generate privilege log entries for withheld documents.
    
    Privilege log format required by courts:
    - Document date
    - Document type
    - Author(s)
    - Recipient(s)
    - Subject matter (without revealing privileged content)
    - Privilege claimed
    """
    
    async def generate_log_entry(
        self,
        doc: ExtractedDocument,
        analysis: PrivilegeAnalysis,
    ) -> dict:
        """Generate privilege log entry for a document."""
        
        # Redact subject line if it reveals privileged content
        subject = doc.metadata.get("subject", "")
        redacted_subject = await self._redact_if_needed(subject)
        
        return {
            "bates_number": doc.metadata.get("bates_number"),
            "date": doc.metadata.get("date"),
            "document_type": self._get_doc_type_description(doc),
            "author": self._format_participants(doc.metadata.get("from")),
            "recipients": self._format_participants(doc.metadata.get("to", [])),
            "cc": self._format_participants(doc.metadata.get("cc", [])),
            "subject_matter": redacted_subject,
            "privilege_claimed": analysis.privilege_type.value,
            "privilege_basis": analysis.privilege_basis,
        }
    
    async def _redact_if_needed(self, text: str) -> str:
        """Redact text if it reveals privileged content."""
        # Use LLM to determine if subject reveals privileged content
        prompt = f"""Does this subject line reveal the substance of legal advice? 
Subject: "{text}"

If yes, provide a generic description that doesn't reveal privileged content.
If no, return the original subject.

Respond in JSON: {{"needs_redaction": true/false, "redacted": "..."}}
"""
        
        response = await self.llm.chat([{"role": "user", "content": prompt}])
        result = json.loads(response)
        
        if result.get("needs_redaction"):
            return result.get("redacted", "[REDACTED]")
        return text
```

---

## 7. Cost Optimization Strategy

### Tiered Processing Cost Analysis

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COST OPTIMIZATION STRATEGY                                │
└─────────────────────────────────────────────────────────────────────────────┘

PROCESSING TIER COSTS (per document):
─────────────────────────────────────

┌───────────────────┬────────────┬────────────┬───────────────┬─────────────┐
│      STAGE        │   COST     │   TIME     │   ACCURACY    │   USE FOR   │
├───────────────────┼────────────┼────────────┼───────────────┼─────────────┤
│ Metadata/Hash     │  $0.0001   │   10ms     │    100%       │ All docs    │
│ Keyword filter    │  $0.0005   │   50ms     │    60-70%     │ All docs    │
│ ML classification │  $0.001    │  100ms     │    85-90%     │ 30% of docs │
│ LLM analysis      │  $0.02     │  2000ms    │    95%+       │ 5% of docs  │
│ Human review      │  $2.00     │  3 min     │    99%        │ 10% of docs │
└───────────────────┴────────────┴────────────┴───────────────┴─────────────┘

EXAMPLE: 1 Million Documents
────────────────────────────

WITHOUT OPTIMIZATION (brute force LLM):
• 1M docs × $0.02 = $20,000 in LLM costs
• 1M docs × 2s = 23 days processing time

WITH TIERED OPTIMIZATION:
• Tier 1 (all): 1M × $0.0005 = $500
• Tier 2 (30%): 300K × $0.001 = $300
• Tier 3 (5%): 50K × $0.02 = $1,000
• Human review (10%): 100K × $2 = $200,000

Total AI cost: ~$1,800 (vs $20,000)
Human review: $200,000 (vs $2,000,000 if reviewing all)

OVERALL: ~90% cost reduction
```

### Cost Optimization Implementation

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class ProcessingTier(Enum):
    METADATA_ONLY = "metadata_only"
    KEYWORD_FILTER = "keyword_filter"
    ML_CLASSIFICATION = "ml_classification"
    LLM_ANALYSIS = "llm_analysis"
    HUMAN_REVIEW = "human_review"

@dataclass
class ProcessingCost:
    tier: ProcessingTier
    cost_per_doc: float
    time_per_doc_ms: int
    accuracy: float

TIER_COSTS = {
    ProcessingTier.METADATA_ONLY: ProcessingCost(
        tier=ProcessingTier.METADATA_ONLY,
        cost_per_doc=0.0001,
        time_per_doc_ms=10,
        accuracy=1.0,  # Not accuracy, just extraction
    ),
    ProcessingTier.KEYWORD_FILTER: ProcessingCost(
        tier=ProcessingTier.KEYWORD_FILTER,
        cost_per_doc=0.0005,
        time_per_doc_ms=50,
        accuracy=0.65,
    ),
    ProcessingTier.ML_CLASSIFICATION: ProcessingCost(
        tier=ProcessingTier.ML_CLASSIFICATION,
        cost_per_doc=0.001,
        time_per_doc_ms=100,
        accuracy=0.88,
    ),
    ProcessingTier.LLM_ANALYSIS: ProcessingCost(
        tier=ProcessingTier.LLM_ANALYSIS,
        cost_per_doc=0.02,
        time_per_doc_ms=2000,
        accuracy=0.95,
    ),
}


class CostOptimizer:
    """
    Optimize processing costs while maintaining quality.
    """
    
    def __init__(
        self,
        budget_limit: float = None,
        accuracy_target: float = 0.95,
        deadline_hours: float = None,
    ):
        self.budget_limit = budget_limit
        self.accuracy_target = accuracy_target
        self.deadline_hours = deadline_hours
    
    def create_processing_plan(
        self,
        total_documents: int,
        estimated_relevant_rate: float = 0.10,
    ) -> 'ProcessingPlan':
        """
        Create cost-optimized processing plan.
        """
        # Calculate document flow through tiers
        tiers = []
        
        # Tier 1: All documents get metadata extraction
        tier1_docs = total_documents
        tiers.append({
            "tier": ProcessingTier.METADATA_ONLY,
            "docs": tier1_docs,
            "cost": tier1_docs * TIER_COSTS[ProcessingTier.METADATA_ONLY].cost_per_doc,
        })
        
        # Tier 2: Keyword filtering
        # Assume 70% pass keyword filter
        tier2_docs = int(tier1_docs * 0.7)
        tiers.append({
            "tier": ProcessingTier.KEYWORD_FILTER,
            "docs": tier1_docs,  # All docs get keyword scoring
            "cost": tier1_docs * TIER_COSTS[ProcessingTier.KEYWORD_FILTER].cost_per_doc,
            "output_docs": tier2_docs,  # 70% pass
        })
        
        # Tier 3: ML classification on keyword hits
        # Assume 40% pass ML threshold
        tier3_docs = int(tier2_docs * 0.4)
        tiers.append({
            "tier": ProcessingTier.ML_CLASSIFICATION,
            "docs": tier2_docs,
            "cost": tier2_docs * TIER_COSTS[ProcessingTier.ML_CLASSIFICATION].cost_per_doc,
            "output_docs": tier3_docs,
        })
        
        # Tier 4: LLM for borderline cases (ML score 0.3-0.7)
        borderline_docs = int(tier2_docs * 0.2)  # 20% are borderline
        tiers.append({
            "tier": ProcessingTier.LLM_ANALYSIS,
            "docs": borderline_docs,
            "cost": borderline_docs * TIER_COSTS[ProcessingTier.LLM_ANALYSIS].cost_per_doc,
        })
        
        # Calculate totals
        total_cost = sum(t["cost"] for t in tiers)
        total_time_hours = self._calculate_processing_time(tiers)
        
        # Estimate documents needing human review
        human_review_docs = int(tier3_docs * 1.2)  # Include LLM-flagged
        human_review_cost = human_review_docs * 2.0  # $2 per doc
        
        return ProcessingPlan(
            tiers=tiers,
            total_ai_cost=total_cost,
            total_human_review_cost=human_review_cost,
            estimated_processing_time_hours=total_time_hours,
            estimated_documents_for_review=human_review_docs,
            estimated_accuracy=self._estimate_accuracy(tiers),
        )
    
    def _calculate_processing_time(self, tiers: List[dict]) -> float:
        """Calculate total processing time with parallelization."""
        total_ms = 0
        
        for tier in tiers:
            tier_cost = TIER_COSTS[tier["tier"]]
            # Assume 100 parallel workers for batch processing
            parallel_factor = 100
            tier_time = (tier["docs"] * tier_cost.time_per_doc_ms) / parallel_factor
            total_ms += tier_time
        
        return total_ms / (1000 * 60 * 60)  # Convert to hours


class BatchProcessor:
    """
    Process documents in cost-optimized batches.
    """
    
    def __init__(
        self,
        keyword_engine,
        ml_engine,
        llm_client,
        cost_tracker,
    ):
        self.keyword = keyword_engine
        self.ml = ml_engine
        self.llm = llm_client
        self.cost_tracker = cost_tracker
    
    async def process_batch(
        self,
        documents: List[ExtractedDocument],
        processing_plan: ProcessingPlan,
    ) -> List[ScoredDocument]:
        """
        Process documents according to cost-optimized plan.
        """
        results = []
        
        # Stage 1: Keyword filtering (all docs)
        keyword_scores = await self.keyword.batch_score(documents)
        
        # Split by keyword score
        low_score = []   # Score < 0.1 - skip expensive processing
        medium_score = []  # 0.1 - 0.5 - ML classification
        high_score = []    # > 0.5 - likely relevant
        
        for doc, score in zip(documents, keyword_scores):
            if score < 0.1:
                # Very unlikely relevant - skip
                results.append(ScoredDocument(
                    doc_id=doc.doc_id,
                    relevance_score=score,
                    score_source="keyword_filter",
                    confidence=0.7,
                ))
                low_score.append(doc)
            elif score < 0.5:
                medium_score.append((doc, score))
            else:
                high_score.append((doc, score))
        
        # Track cost
        await self.cost_tracker.record(
            tier="keyword",
            docs=len(documents),
            cost=len(documents) * 0.0005,
        )
        
        # Stage 2: ML classification for medium scores
        if medium_score:
            ml_docs = [d for d, _ in medium_score]
            ml_scores = await self.ml.batch_score(ml_docs)
            
            for (doc, kw_score), ml_score in zip(medium_score, ml_scores):
                combined_score = (kw_score * 0.3 + ml_score * 0.7)
                
                if ml_score > 0.7 or ml_score < 0.3:
                    # ML is confident
                    results.append(ScoredDocument(
                        doc_id=doc.doc_id,
                        relevance_score=combined_score,
                        score_source="ml_classification",
                        confidence=abs(ml_score - 0.5) * 2,
                    ))
                else:
                    # ML uncertain - escalate to LLM
                    high_score.append((doc, combined_score))
            
            await self.cost_tracker.record(
                tier="ml",
                docs=len(ml_docs),
                cost=len(ml_docs) * 0.001,
            )
        
        # Stage 3: LLM for high-score and uncertain docs
        # Batch to reduce API calls
        if high_score:
            llm_batch_size = 10  # Process 10 at a time
            for i in range(0, len(high_score), llm_batch_size):
                batch = high_score[i:i+llm_batch_size]
                llm_results = await self._batch_llm_score(batch)
                results.extend(llm_results)
            
            await self.cost_tracker.record(
                tier="llm",
                docs=len(high_score),
                cost=len(high_score) * 0.02,
            )
        
        return results
    
    async def _batch_llm_score(
        self,
        doc_scores: List[Tuple[ExtractedDocument, float]],
    ) -> List[ScoredDocument]:
        """
        Batch LLM scoring to reduce API calls.
        
        Strategy: Include multiple documents in single prompt
        when they're short enough.
        """
        # Estimate tokens per document
        short_docs = []
        long_docs = []
        
        for doc, score in doc_scores:
            tokens = len(doc.text_content) // 4  # Rough estimate
            if tokens < 1000:
                short_docs.append((doc, score))
            else:
                long_docs.append((doc, score))
        
        results = []
        
        # Process short docs in batches
        if short_docs:
            batch_results = await self._multi_doc_llm_call(short_docs)
            results.extend(batch_results)
        
        # Process long docs individually
        for doc, score in long_docs:
            result = await self._single_doc_llm_call(doc, score)
            results.append(result)
        
        return results
```

---

## 8. Batch Processing Architecture

### Distributed Processing System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         Job Submission
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         JOB SCHEDULER                                        │
│                                                                              │
│  • Break job into tasks                                                     │
│  • Estimate resources needed                                                │
│  • Schedule based on priority and capacity                                  │
│  • Track progress and handle failures                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  TASK QUEUE     │   │  TASK QUEUE     │   │  TASK QUEUE     │
│  (Ingestion)    │   │  (Analysis)     │   │  (Export)       │
│                 │   │                 │   │                 │
│  SQS/Redis      │   │  SQS/Redis      │   │  SQS/Redis      │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  WORKER POOL    │   │  WORKER POOL    │   │  WORKER POOL    │
│  (Ingestion)    │   │  (Analysis)     │   │  (Export)       │
│                 │   │                 │   │                 │
│  20 workers     │   │  50 workers     │   │  10 workers     │
│  c6i.xlarge     │   │  c6i.2xlarge    │   │  c6i.xlarge     │
│                 │   │  + GPU for ML   │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘

Auto-scaling based on:
• Queue depth (scale up when > 10K tasks)
• Processing time (scale up when behind SLA)
• Time of day (scale down nights/weekends)
```

### Job and Task Management

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime
import uuid

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class ProcessingJob:
    """Top-level processing job for a matter."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    matter_id: str = ""
    job_type: str = ""  # "ingestion", "analysis", "production"
    
    status: JobStatus = JobStatus.PENDING
    priority: int = 5  # 1-10, higher = more urgent
    
    # Progress tracking
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Configuration
    config: Dict = field(default_factory=dict)
    
    @property
    def progress_percent(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


@dataclass
class ProcessingTask:
    """Individual processing task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str = ""
    task_type: str = ""
    
    status: TaskStatus = TaskStatus.QUEUED
    priority: int = 5
    
    # What to process
    payload: Dict = field(default_factory=dict)
    
    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    
    # Worker info
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobScheduler:
    """
    Schedule and manage batch processing jobs.
    """
    
    def __init__(
        self,
        task_queues: Dict[str, 'TaskQueue'],
        db: 'Database',
        metrics: 'MetricsCollector',
    ):
        self.queues = task_queues
        self.db = db
        self.metrics = metrics
    
    async def create_ingestion_job(
        self,
        matter_id: str,
        source_paths: List[str],
        options: Dict = None,
    ) -> ProcessingJob:
        """
        Create job to ingest documents from sources.
        """
        job = ProcessingJob(
            matter_id=matter_id,
            job_type="ingestion",
            config=options or {},
        )
        
        # Enumerate all files to process
        all_files = []
        for source_path in source_paths:
            files = await self._enumerate_files(source_path)
            all_files.extend(files)
        
        job.total_tasks = len(all_files)
        
        # Create tasks for each file
        tasks = []
        for file_info in all_files:
            task = ProcessingTask(
                job_id=job.job_id,
                task_type="ingest_document",
                payload={
                    "file_path": file_info.path,
                    "file_type": file_info.type,
                    "file_size": file_info.size,
                },
                priority=self._calculate_task_priority(file_info),
            )
            tasks.append(task)
        
        # Save job and tasks
        await self.db.save_job(job)
        await self.db.save_tasks(tasks)
        
        # Queue tasks
        for task in tasks:
            await self.queues["ingestion"].enqueue(task)
        
        return job
    
    async def create_analysis_job(
        self,
        matter_id: str,
        analysis_types: List[str],  # ["relevance", "privilege", "clustering"]
    ) -> ProcessingJob:
        """
        Create job to analyze documents.
        """
        job = ProcessingJob(
            matter_id=matter_id,
            job_type="analysis",
            config={"analysis_types": analysis_types},
        )
        
        # Get documents to analyze
        documents = await self.db.get_documents(matter_id, status="ingested")
        job.total_tasks = len(documents) * len(analysis_types)
        
        # Create tasks
        tasks = []
        for doc in documents:
            for analysis_type in analysis_types:
                task = ProcessingTask(
                    job_id=job.job_id,
                    task_type=f"analyze_{analysis_type}",
                    payload={
                        "doc_id": doc.doc_id,
                        "analysis_type": analysis_type,
                    },
                    priority=job.priority,
                )
                tasks.append(task)
        
        await self.db.save_job(job)
        await self.db.save_tasks(tasks)
        
        for task in tasks:
            await self.queues["analysis"].enqueue(task)
        
        return job


class TaskWorker:
    """
    Worker that processes tasks from queue.
    """
    
    def __init__(
        self,
        worker_id: str,
        queue: 'TaskQueue',
        processors: Dict[str, 'TaskProcessor'],
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.processors = processors
        self.running = False
    
    async def run(self):
        """Main worker loop."""
        self.running = True
        
        while self.running:
            try:
                # Get task from queue (blocks if empty)
                task = await self.queue.dequeue(timeout=30)
                
                if task:
                    await self._process_task(task)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(5)
    
    async def _process_task(self, task: ProcessingTask):
        """Process a single task."""
        # Update task status
        task.status = TaskStatus.RUNNING
        task.worker_id = self.worker_id
        task.started_at = datetime.utcnow()
        task.attempts += 1
        await self.db.update_task(task)
        
        try:
            # Get processor for task type
            processor = self.processors.get(task.task_type)
            if not processor:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Execute task
            result = await processor.process(task.payload)
            
            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update job progress
            await self.db.increment_job_progress(task.job_id)
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.last_error = str(e)
            
            if task.attempts >= task.max_attempts:
                task.status = TaskStatus.FAILED
                await self.db.increment_job_failures(task.job_id)
            else:
                # Retry with backoff
                task.status = TaskStatus.RETRY
                await self.queue.enqueue(
                    task,
                    delay_seconds=self._calculate_backoff(task.attempts),
                )
        
        finally:
            await self.db.update_task(task)
    
    def _calculate_backoff(self, attempts: int) -> int:
        """Exponential backoff for retries."""
        return min(300, 2 ** attempts * 10)  # Max 5 minutes
```

---

## 9. Production & Export System

### Production Formats

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION FORMAT REQUIREMENTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

COMMON PRODUCTION FORMATS:
──────────────────────────

1. NATIVE PRODUCTION
   • Original file format (DOCX, XLSX, PDF, MSG)
   • Preserves metadata
   • Used for: Spreadsheets, databases, complex documents

2. TIFF/PDF PRODUCTION (most common)
   • Convert all docs to TIFF images or PDF
   • Single-page TIFFs: 300 DPI, Group IV compression
   • Bates numbered (unique page identifier)
   • Used for: Emails, word docs, most documents

3. LOAD FILE FORMATS
   • Concordance (.dat) - Tab-delimited metadata
   • Relativity (.opt/.dat) - Industry standard
   • Summation (.dii) - Legacy format
   • EDRM XML - Open standard

LOAD FILE CONTENTS:
───────────────────
• Bates number range (start/end)
• Document metadata (date, author, subject)
• File paths (to images and natives)
• Text file path (extracted text)
• Family relationships (parent/child)
• Custodian information
• Confidentiality designations

EXAMPLE LOAD FILE (.dat):
─────────────────────────
þBATES_BEGþBATES_ENDþCUSTODIANþDATE_SENTþFROMþTOþSUBJECTþ
þABC000001þABC000003þJohn SmithþÞ2023-01-15þjsmith@coþmjones@coþRE: Contractþ
þABC000004þABC000004þJohn SmithþÞ2023-01-16þjsmith@coþlegal@coþPRIVILEGEDþ
```

### Production Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, BinaryIO
from enum import Enum
from pathlib import Path
import io

class ProductionFormat(Enum):
    NATIVE = "native"
    TIFF = "tiff"
    PDF = "pdf"
    
class LoadFileFormat(Enum):
    CONCORDANCE = "concordance"
    RELATIVITY = "relativity"
    EDRM_XML = "edrm_xml"

@dataclass
class ProductionConfig:
    """Configuration for document production."""
    format: ProductionFormat
    load_file_format: LoadFileFormat
    
    # Bates numbering
    bates_prefix: str  # e.g., "ABC"
    bates_start: int
    bates_digits: int = 7  # ABC0000001
    
    # Image settings (for TIFF/PDF)
    dpi: int = 300
    color_mode: str = "bitonal"  # bitonal, grayscale, color
    
    # Metadata fields to include
    metadata_fields: List[str] = None
    
    # Confidentiality
    apply_confidentiality_stamp: bool = True
    confidentiality_text: str = "CONFIDENTIAL"
    
    # Redaction
    apply_redactions: bool = True
    redaction_color: str = "black"

@dataclass
class ProductionDocument:
    """Document prepared for production."""
    doc_id: str
    bates_begin: str
    bates_end: str
    page_count: int
    
    # File paths (relative to production root)
    image_path: Optional[str]  # Path to TIFF/PDF
    native_path: Optional[str]  # Path to native file
    text_path: Optional[str]   # Path to extracted text
    
    # Metadata for load file
    metadata: Dict


class ProductionEngine:
    """
    Generate document productions in court-required formats.
    """
    
    def __init__(
        self,
        storage: 'DocumentStorage',
        converter: 'DocumentConverter',
        config: ProductionConfig,
    ):
        self.storage = storage
        self.converter = converter
        self.config = config
        self.current_bates = config.bates_start
    
    async def create_production(
        self,
        matter_id: str,
        document_ids: List[str],
        output_path: str,
    ) -> 'ProductionResult':
        """
        Create production from selected documents.
        """
        production_docs = []
        
        # Create directory structure
        await self._create_directories(output_path)
        
        for doc_id in document_ids:
            # Get document
            doc = await self.storage.get_document(doc_id)
            
            # Convert to production format
            prod_doc = await self._produce_document(doc, output_path)
            production_docs.append(prod_doc)
        
        # Generate load file
        load_file_path = await self._generate_load_file(
            production_docs,
            output_path,
        )
        
        # Generate production summary
        summary = await self._generate_summary(production_docs)
        
        return ProductionResult(
            production_docs=production_docs,
            load_file_path=load_file_path,
            summary=summary,
            bates_range=(
                self._format_bates(self.config.bates_start),
                self._format_bates(self.current_bates - 1),
            ),
        )
    
    async def _produce_document(
        self,
        doc: ExtractedDocument,
        output_path: str,
    ) -> ProductionDocument:
        """
        Produce a single document.
        """
        bates_begin = self._format_bates(self.current_bates)
        
        if self.config.format == ProductionFormat.NATIVE:
            # Copy native file with Bates-numbered name
            native_path = await self._copy_native(doc, output_path, bates_begin)
            image_path = None
            page_count = 1
            
        elif self.config.format in [ProductionFormat.TIFF, ProductionFormat.PDF]:
            # Convert to images
            images = await self._convert_to_images(doc)
            
            # Apply Bates stamps to each page
            stamped_images = []
            for i, image in enumerate(images):
                bates = self._format_bates(self.current_bates + i)
                stamped = await self._apply_bates_stamp(image, bates)
                
                # Apply confidentiality stamp if configured
                if self.config.apply_confidentiality_stamp:
                    stamped = await self._apply_confidentiality_stamp(stamped)
                
                stamped_images.append(stamped)
            
            # Save images
            image_path = await self._save_images(
                stamped_images,
                output_path,
                bates_begin,
            )
            native_path = None
            page_count = len(images)
        
        bates_end = self._format_bates(self.current_bates + page_count - 1)
        self.current_bates += page_count
        
        # Save extracted text
        text_path = await self._save_text(doc, output_path, bates_begin)
        
        return ProductionDocument(
            doc_id=doc.doc_id,
            bates_begin=bates_begin,
            bates_end=bates_end,
            page_count=page_count,
            image_path=image_path,
            native_path=native_path,
            text_path=text_path,
            metadata=self._extract_metadata(doc),
        )
    
    async def _convert_to_images(
        self,
        doc: ExtractedDocument,
    ) -> List[bytes]:
        """
        Convert document to page images.
        """
        if doc.file_type == FileType.PDF:
            return await self.converter.pdf_to_tiff(
                doc.content,
                dpi=self.config.dpi,
            )
        
        elif doc.file_type in [FileType.OFFICE_DOCX, FileType.OFFICE_DOC]:
            # Convert Word to PDF first, then to TIFF
            pdf_bytes = await self.converter.word_to_pdf(doc.content)
            return await self.converter.pdf_to_tiff(pdf_bytes, dpi=self.config.dpi)
        
        elif doc.file_type in [FileType.EMAIL_MSG, FileType.EMAIL_EML]:
            # Render email to PDF, then to TIFF
            pdf_bytes = await self.converter.email_to_pdf(doc)
            return await self.converter.pdf_to_tiff(pdf_bytes, dpi=self.config.dpi)
        
        elif doc.file_type == FileType.IMAGE:
            # Already an image, just convert to TIFF if needed
            return [await self.converter.image_to_tiff(doc.content)]
        
        else:
            raise ValueError(f"Unsupported file type for imaging: {doc.file_type}")
    
    async def _apply_bates_stamp(
        self,
        image: bytes,
        bates: str,
    ) -> bytes:
        """
        Apply Bates number stamp to image.
        """
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.open(io.BytesIO(image))
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Position: bottom right corner
        text_width = draw.textlength(bates, font=font)
        x = img.width - text_width - 50
        y = img.height - 50
        
        # Draw with white background for readability
        bbox = draw.textbbox((x, y), bates, font=font)
        draw.rectangle(bbox, fill="white")
        draw.text((x, y), bates, fill="black", font=font)
        
        # Convert back to bytes
        output = io.BytesIO()
        img.save(output, format="TIFF", compression="group4")
        return output.getvalue()
    
    async def _generate_load_file(
        self,
        documents: List[ProductionDocument],
        output_path: str,
    ) -> str:
        """
        Generate load file for review platform import.
        """
        if self.config.load_file_format == LoadFileFormat.CONCORDANCE:
            return await self._generate_concordance_dat(documents, output_path)
        
        elif self.config.load_file_format == LoadFileFormat.RELATIVITY:
            return await self._generate_relativity_load(documents, output_path)
        
        elif self.config.load_file_format == LoadFileFormat.EDRM_XML:
            return await self._generate_edrm_xml(documents, output_path)
    
    async def _generate_concordance_dat(
        self,
        documents: List[ProductionDocument],
        output_path: str,
    ) -> str:
        """
        Generate Concordance .dat load file.
        
        Format: þ-delimited (ASCII 254)
        """
        DELIMITER = "þ"  # Concordance delimiter
        NEWLINE = "®"    # Concordance newline in field
        
        # Define fields
        fields = [
            "BATES_BEGIN",
            "BATES_END",
            "PAGE_COUNT",
            "CUSTODIAN",
            "DATE_SENT",
            "DATE_RECEIVED",
            "FROM",
            "TO",
            "CC",
            "BCC",
            "SUBJECT",
            "FILE_NAME",
            "FILE_TYPE",
            "IMAGE_PATH",
            "NATIVE_PATH",
            "TEXT_PATH",
        ]
        
        lines = []
        
        # Header row
        lines.append(DELIMITER + DELIMITER.join(fields) + DELIMITER)
        
        # Data rows
        for doc in documents:
            values = [
                doc.bates_begin,
                doc.bates_end,
                str(doc.page_count),
                doc.metadata.get("custodian", ""),
                doc.metadata.get("date_sent", ""),
                doc.metadata.get("date_received", ""),
                doc.metadata.get("from", ""),
                self._format_recipients(doc.metadata.get("to", [])),
                self._format_recipients(doc.metadata.get("cc", [])),
                self._format_recipients(doc.metadata.get("bcc", [])),
                doc.metadata.get("subject", "").replace("\n", NEWLINE),
                doc.metadata.get("file_name", ""),
                doc.metadata.get("file_type", ""),
                doc.image_path or "",
                doc.native_path or "",
                doc.text_path or "",
            ]
            
            lines.append(DELIMITER + DELIMITER.join(values) + DELIMITER)
        
        # Write file
        dat_path = f"{output_path}/loadfile.dat"
        content = "\n".join(lines)
        
        # Write with proper encoding (usually ANSI for Concordance)
        with open(dat_path, "w", encoding="cp1252") as f:
            f.write(content)
        
        # Also generate .opt file for images
        await self._generate_opt_file(documents, output_path)
        
        return dat_path
    
    async def _generate_opt_file(
        self,
        documents: List[ProductionDocument],
        output_path: str,
    ):
        """
        Generate .opt image cross-reference file.
        
        Format: BatesNumber,VolumeLabel,ImagePath,DocBreak,FolderBreak,PageCount
        """
        lines = []
        
        for doc in documents:
            if doc.image_path:
                # Single-page format
                for i in range(doc.page_count):
                    bates = self._format_bates(
                        int(doc.bates_begin.replace(self.config.bates_prefix, "")) + i
                    )
                    page_path = doc.image_path.replace(".tif", f"_{i+1:04d}.tif")
                    
                    # Y = document break on first page
                    doc_break = "Y" if i == 0 else ""
                    
                    lines.append(f"{bates},VOL001,{page_path},{doc_break},,")
        
        opt_path = f"{output_path}/loadfile.opt"
        with open(opt_path, "w") as f:
            f.write("\n".join(lines))
    
    def _format_bates(self, number: int) -> str:
        """Format Bates number with prefix and zero-padding."""
        return f"{self.config.bates_prefix}{number:0{self.config.bates_digits}d}"
```

---

## 10. Scale & Performance

### Performance Targets

| Operation | Target | SLA |
|-----------|--------|-----|
| Document ingestion | 100K docs/hour | 50K minimum |
| Metadata extraction | 500 docs/second | 200 minimum |
| Full-text search | < 2 seconds | < 5 seconds |
| Relevance scoring | 50K docs/hour | 20K minimum |
| Production export | 10K docs/hour | 5K minimum |
| Total processing | 1M docs in 24 hours | 48 hours max |

### Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SCALING ARCHITECTURE                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              Load Balancer
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            ┌─────────────┐               ┌─────────────┐
            │   API       │               │   API       │
            │  Cluster    │               │  Cluster    │
            │  (Web UI)   │               │  (Batch)    │
            └─────────────┘               └─────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MESSAGE QUEUES                                       │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Ingest Q   │  │  Analysis Q │  │  Export Q   │  │  Priority Q │        │
│  │  (SQS)      │  │  (SQS)      │  │  (SQS)      │  │  (SQS)      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKER POOLS (Auto-scaling)                          │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  Ingestion Workers  │  │  Analysis Workers   │  │  Export Workers     │ │
│  │                     │  │                     │  │                     │ │
│  │  Min: 5             │  │  Min: 10            │  │  Min: 2             │ │
│  │  Max: 50            │  │  Max: 100           │  │  Max: 20            │ │
│  │  Type: c6i.xlarge   │  │  Type: g4dn.xlarge  │  │  Type: c6i.2xlarge  │ │
│  │                     │  │  (GPU for ML)       │  │                     │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                        │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │    PostgreSQL     │  │   Elasticsearch   │  │      Redis        │       │
│  │    (RDS r6g.xl)   │  │    (3-node)       │  │    (Cluster)      │       │
│  │                   │  │                   │  │                   │       │
│  │  • Metadata       │  │  • Full-text      │  │  • Cache          │       │
│  │  • Review state   │  │  • Faceted search │  │  • Job state      │       │
│  │  • Audit log      │  │  • Analytics      │  │  • Rate limiting  │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                              │
│  ┌───────────────────┐  ┌───────────────────┐                              │
│  │        S3         │  │    Turbopuffer    │                              │
│  │                   │  │                   │                              │
│  │  • Native files   │  │  • Embeddings     │                              │
│  │  • Images         │  │  • Semantic       │                              │
│  │  • Text           │  │    search         │                              │
│  └───────────────────┘  └───────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Quality Control & Defensibility

### Defensibility Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEFENSIBILITY FRAMEWORK                                   │
└─────────────────────────────────────────────────────────────────────────────┘

WHY DEFENSIBILITY MATTERS:
──────────────────────────
• Opposing counsel can challenge your discovery process
• Court can sanction parties for inadequate discovery
• "Reasonableness" is the legal standard
• Must be able to explain and justify methodology

KEY DEFENSIBILITY ELEMENTS:

1. DOCUMENTED METHODOLOGY
   • Written protocol before processing begins
   • Search term validation
   • TAR model training documentation
   • Quality control sampling plan

2. VALIDATION & TESTING
   • Random sample review for recall estimation
   • Elusion testing (search for what you missed)
   • Statistical analysis of results

3. COMPLETE AUDIT TRAIL
   • Chain of custody for all documents
   • Every search and filter logged
   • All reviewer decisions recorded
   • System changes documented

4. QUALITY CONTROL
   • Second-level review of sample
   • Consistency checking across reviewers
   • Error correction procedures
```

### Quality Control Implementation

```python
from dataclasses import dataclass
from typing import List, Dict
import random
import math

@dataclass
class QualityControlResult:
    """Results of quality control sampling."""
    sample_size: int
    agreed_count: int
    disagreed_count: int
    agreement_rate: float
    estimated_error_rate: float
    confidence_interval: tuple
    
    # Detailed breakdown
    false_positives: int  # Marked relevant, actually not
    false_negatives: int  # Marked not relevant, actually is
    
    # Statistical metrics
    precision: float
    recall_estimate: float
    f1_score: float


class QualityController:
    """
    Quality control for TAR/predictive coding defensibility.
    """
    
    def __init__(self, db, confidence_level: float = 0.95):
        self.db = db
        self.confidence_level = confidence_level
    
    async def calculate_sample_size(
        self,
        population_size: int,
        expected_error_rate: float = 0.05,
        margin_of_error: float = 0.02,
    ) -> int:
        """
        Calculate statistically valid sample size.
        
        Uses standard sample size formula for proportions.
        """
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(self.confidence_level, 1.96)
        
        # Sample size for infinite population
        p = expected_error_rate
        n_infinite = (z ** 2 * p * (1 - p)) / (margin_of_error ** 2)
        
        # Adjust for finite population
        n_finite = n_infinite / (1 + (n_infinite - 1) / population_size)
        
        return math.ceil(n_finite)
    
    async def create_qc_sample(
        self,
        matter_id: str,
        population: str = "reviewed",  # "reviewed" or "all"
    ) -> List[str]:
        """
        Create random sample for quality control review.
        """
        if population == "reviewed":
            docs = await self.db.get_reviewed_documents(matter_id)
        else:
            docs = await self.db.get_all_documents(matter_id)
        
        sample_size = await self.calculate_sample_size(len(docs))
        
        # Random sample
        sample_ids = random.sample([d.doc_id for d in docs], sample_size)
        
        # Record sample for audit
        await self.db.record_qc_sample(matter_id, sample_ids)
        
        return sample_ids
    
    async def calculate_qc_metrics(
        self,
        matter_id: str,
        qc_reviews: List[Dict],  # [{doc_id, original_label, qc_label}]
    ) -> QualityControlResult:
        """
        Calculate quality control metrics from sample review.
        """
        agreed = 0
        disagreed = 0
        false_positives = 0
        false_negatives = 0
        
        for review in qc_reviews:
            if review['original_label'] == review['qc_label']:
                agreed += 1
            else:
                disagreed += 1
                
                if review['original_label'] == 'relevant' and review['qc_label'] == 'not_relevant':
                    false_positives += 1
                elif review['original_label'] == 'not_relevant' and review['qc_label'] == 'relevant':
                    false_negatives += 1
        
        total = agreed + disagreed
        agreement_rate = agreed / total if total > 0 else 0
        error_rate = disagreed / total if total > 0 else 0
        
        # Calculate confidence interval for error rate
        ci = self._calculate_confidence_interval(error_rate, total)
        
        # Precision and recall estimates
        # (simplified - real calculation would use full confusion matrix)
        precision = 1 - (false_positives / total) if total > 0 else 0
        recall_estimate = 1 - (false_negatives / total) if total > 0 else 0
        
        f1 = 2 * (precision * recall_estimate) / (precision + recall_estimate) if (precision + recall_estimate) > 0 else 0
        
        return QualityControlResult(
            sample_size=total,
            agreed_count=agreed,
            disagreed_count=disagreed,
            agreement_rate=agreement_rate,
            estimated_error_rate=error_rate,
            confidence_interval=ci,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall_estimate=recall_estimate,
            f1_score=f1,
        )
    
    async def elusion_test(
        self,
        matter_id: str,
        sample_size: int = 300,
    ) -> Dict:
        """
        Elusion test: Sample documents marked "not relevant" to estimate
        how many relevant documents were missed.
        
        This is the key defensibility metric for TAR.
        """
        # Get documents marked not relevant
        not_relevant_docs = await self.db.get_documents_by_label(
            matter_id,
            label="not_relevant"
        )
        
        # Random sample
        sample = random.sample(
            [d.doc_id for d in not_relevant_docs],
            min(sample_size, len(not_relevant_docs))
        )
        
        # These will be reviewed by senior attorney
        await self.db.create_elusion_review(matter_id, sample)
        
        return {
            "sample_size": len(sample),
            "population_size": len(not_relevant_docs),
            "sample_doc_ids": sample,
            "status": "pending_review",
        }
    
    async def calculate_elusion_result(
        self,
        matter_id: str,
        elusion_reviews: List[Dict],  # [{doc_id, is_actually_relevant}]
    ) -> Dict:
        """
        Calculate recall estimate from elusion test results.
        """
        actually_relevant = sum(1 for r in elusion_reviews if r['is_actually_relevant'])
        sample_size = len(elusion_reviews)
        
        # Elusion rate = proportion of "not relevant" that were actually relevant
        elusion_rate = actually_relevant / sample_size if sample_size > 0 else 0
        
        # Get counts for recall calculation
        total_not_relevant = await self.db.count_documents_by_label(matter_id, "not_relevant")
        total_relevant = await self.db.count_documents_by_label(matter_id, "relevant")
        
        # Estimate missed relevant documents
        estimated_missed = int(elusion_rate * total_not_relevant)
        
        # Calculate recall
        total_actually_relevant = total_relevant + estimated_missed
        recall = total_relevant / total_actually_relevant if total_actually_relevant > 0 else 0
        
        # Confidence interval
        ci = self._calculate_confidence_interval(elusion_rate, sample_size)
        
        return {
            "sample_size": sample_size,
            "actually_relevant_in_sample": actually_relevant,
            "elusion_rate": elusion_rate,
            "elusion_rate_ci": ci,
            "estimated_missed_relevant": estimated_missed,
            "recall_estimate": recall,
            "is_acceptable": recall >= 0.75,  # Common threshold
        }
    
    def _calculate_confidence_interval(
        self,
        proportion: float,
        sample_size: int,
    ) -> tuple:
        """Calculate confidence interval for proportion."""
        z = 1.96  # 95% confidence
        
        se = math.sqrt((proportion * (1 - proportion)) / sample_size)
        margin = z * se
        
        lower = max(0, proportion - margin)
        upper = min(1, proportion + margin)
        
        return (lower, upper)
```

---

## 12. Interview Discussion Points

### Questions They'll Ask

**Q: How do you handle 2 million documents in a reasonable timeframe?**

> **A:** Tiered processing with parallel batch architecture. Tier 1 (metadata/keyword) processes everything fast and cheap, filtering out 70%. Tier 2 (ML classification) handles the medium-score documents. Tier 3 (LLM) only for the uncertain 5%. Distributed workers auto-scale based on queue depth. Target: 1M docs in 24 hours with 100 parallel workers.

**Q: How do you ensure the TAR process is defensible in court?**

> **A:** Four pillars: (1) Documented methodology before starting - search protocol, TAR training plan; (2) Validation through random sampling and elusion testing - prove recall is acceptable (>75%); (3) Complete audit trail - every search, every decision logged; (4) Quality control - second-level review of samples, inter-reviewer consistency. Courts look for "reasonableness," not perfection.

**Q: How do you balance cost optimization with accuracy?**

> **A:** Tiered approach where each tier is more expensive but more accurate. Cheap keyword filtering (free, 65% accurate) eliminates obvious non-relevant docs. ML classification ($0.001/doc, 88% accurate) handles the bulk. LLM ($0.02/doc, 95% accurate) only for borderline cases. Result: 90% cost reduction vs. brute-force LLM while maintaining accuracy on the documents that matter.

**Q: How do you handle privilege review at scale?**

> **A:** Two-stage: (1) AI flags potential privilege based on attorney participation, privilege indicators, and content analysis; (2) Flagged documents go to attorney review queue - AI never makes final privilege determination. Generate privilege log entries automatically for withheld documents. Check for waiver risks (third-party recipients).

**Q: How do you handle near-duplicate detection?**

> **A:** Multi-level: (1) Exact duplicates via MD5/SHA256 hash - O(1) lookup; (2) Near-duplicates via SimHash - locality-sensitive hashing that produces similar hashes for similar documents, then compare Hamming distance; (3) Email threading - group conversations, suppress earlier messages that are quoted in later ones; (4) Family deduplication - dedupe attachments across email families.

### Questions to Ask Them

1. **"What's your typical matter size, and what are your current processing SLAs?"**

2. **"How do you handle client-specific relevance criteria - do you fine-tune models per matter or use few-shot learning?"**

3. **"What's your approach when opposing counsel challenges your TAR methodology?"**

4. **"Do you support continuous active learning during review, or is it a batch process?"**

5. **"How do you handle production formats for different receiving parties with different requirements?"**

---

## Quick Reference: Key Metrics

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Recall** | > 75% | Court-acceptable threshold |
| **Precision** | > 50% | Reduce human review burden |
| **Processing Speed** | 100K/hour | Meet discovery deadlines |
| **Cost per Doc** | < $0.10 | Client budget constraints |
| **Dedup Rate** | 30-50% | Reduce review volume |
| **QC Agreement** | > 90% | Defensibility |

---

*This system design covers enterprise-scale e-discovery processing. Emphasize the cost optimization strategy (tiered processing) and defensibility framework (elusion testing, audit trails) as key differentiators.*
