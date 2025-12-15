# System Design: Multi-Tenant SaaS Architecture for Law Firms

## Table of Contents
1. [Why Attorney-Client Privilege Makes This Critical](#1-why-attorney-client-privilege-makes-this-critical)
2. [Tenant Isolation Strategies](#2-tenant-isolation-strategies)
3. [Recommended Architecture](#3-recommended-architecture)
4. [Database Layer Design](#4-database-layer-design)
5. [Application Layer Isolation](#5-application-layer-isolation)
6. [Encryption & Key Management](#6-encryption--key-management)
7. [Audit Logging System](#7-audit-logging-system)
8. [Resource Isolation & Quotas](#8-resource-isolation--quotas)
9. [Compliance Framework](#9-compliance-framework)
10. [Scale & Operations](#10-scale--operations)
11. [Disaster Recovery & Data Sovereignty](#11-disaster-recovery--data-sovereignty)
12. [Interview Discussion Points](#12-interview-discussion-points)

---

## 1. Why Attorney-Client Privilege Makes This Critical

### The Legal Stakes

Attorney-client privilege is one of the oldest and most sacred legal protections. **A single data leak can waive privilege for an entire case or client relationship.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRIVILEGE WAIVER SCENARIOS                                │
└─────────────────────────────────────────────────────────────────────────────┘

SCENARIO 1: Cross-Tenant Data Leak
──────────────────────────────────
Law Firm A is defending BigCorp in a lawsuit.
Law Firm B is the opposing counsel (plaintiff's firm).

If Law Firm B accidentally sees ANY of Firm A's case documents through
a software bug, BigCorp could argue:
• Privilege was waived through "voluntary disclosure to third party"
• All related privileged communications may now be discoverable
• Potential malpractice claim against Firm A

SCENARIO 2: Search Index Contamination
──────────────────────────────────────
Global search index accidentally includes Firm A's documents.
Firm B's attorney searches for "merger agreement" and sees a snippet
from Firm A's confidential M&A deal.

Even if Firm B doesn't USE the information:
• The disclosure itself may trigger waiver analysis
• Firm A must now disclose the breach to their client
• Client may need to notify opposing counsel
• Potential bar complaint and malpractice exposure

SCENARIO 3: AI Model Contamination
─────────────────────────────────
Training or fine-tuning an AI model on Firm A's documents,
then using that model to serve Firm B.

• Model might "memorize" and regurgitate privileged content
• Even statistical patterns could leak sensitive information
• Unclear legal territory but extreme risk
```

### The Business Stakes

| Breach Impact | Consequence |
|---------------|-------------|
| **Malpractice Liability** | Firms can be sued for breaching client confidentiality |
| **Bar Discipline** | Attorneys can lose their license |
| **Client Loss** | One breach = lose the client forever |
| **Reputation Damage** | Law is a relationship business; word spreads fast |
| **Regulatory Fines** | HIPAA (medical records), state bar rules |

### Design Principle: Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DEFENSE IN DEPTH LAYERS                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 1: NETWORK ISOLATION                                          │
    │  • Tenant-aware load balancing                                       │
    │  • VPC/subnet separation for large tenants                          │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 2: APPLICATION ISOLATION                                      │
    │  • Tenant context in every request                                   │
    │  • Middleware enforces tenant boundaries                            │
    │  • No cross-tenant API calls possible                               │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 3: DATABASE ISOLATION                                         │
    │  • Schema-per-tenant (PostgreSQL)                                    │
    │  • Index-per-case (Vector DB)                                        │
    │  • Row-level security as backup                                      │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 4: ENCRYPTION ISOLATION                                       │
    │  • Per-tenant encryption keys                                        │
    │  • Key rotation per tenant                                           │
    │  • Tenant can't decrypt other tenant's data                         │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Layer 5: AUDIT & DETECTION                                          │
    │  • Log all data access                                               │
    │  • Anomaly detection for cross-tenant access attempts               │
    │  • Immutable audit trail                                             │
    └─────────────────────────────────────────────────────────────────────┘

    If ONE layer fails, others still protect the data.
```

---

## 2. Tenant Isolation Strategies

### Comparison of Approaches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TENANCY ISOLATION MODELS                            │
└─────────────────────────────────────────────────────────────────────────────┘

MODEL 1: SHARED DATABASE + TENANT_ID COLUMN
───────────────────────────────────────────
┌─────────────────────────────────────────┐
│           SHARED DATABASE               │
│  ┌───────────────────────────────────┐  │
│  │ cases table                       │  │
│  │ ─────────────────────────────     │  │
│  │ id │ tenant_id │ client_name │... │  │
│  │ 1  │ firm_a    │ BigCorp     │    │  │
│  │ 2  │ firm_b    │ MegaCorp    │    │  │  ← ALL DATA IN SAME TABLE
│  │ 3  │ firm_a    │ SmallCo     │    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

✗ ONE BUG in WHERE clause = data leak
✗ Accidental JOIN without tenant filter = leak
✗ Hard to prove isolation for compliance
✗ Can't give tenant their data easily (export)
✓ Simple to implement
✓ Efficient resource usage


MODEL 2: SCHEMA-PER-TENANT (RECOMMENDED FOR MOST)
─────────────────────────────────────────────────
┌─────────────────────────────────────────┐
│           SHARED DATABASE               │
│  ┌─────────────┐  ┌─────────────┐      │
│  │ firm_a      │  │ firm_b      │      │
│  │ (schema)    │  │ (schema)    │      │
│  │ ┌─────────┐ │  │ ┌─────────┐ │      │
│  │ │ cases   │ │  │ │ cases   │ │      │
│  │ │ docs    │ │  │ │ docs    │ │      │
│  │ │ users   │ │  │ │ users   │ │      │
│  │ └─────────┘ │  │ └─────────┘ │      │
│  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────┘

✓ Strong isolation (can't accidentally query wrong schema)
✓ Easy to export tenant data (pg_dump schema)
✓ Can have schema-specific customizations
✓ Row-level security as additional layer
✗ More complex migrations (N schemas)
✗ Connection pooling complexity
✗ Schema count limits (thousands OK, millions not)


MODEL 3: DATABASE-PER-TENANT (HIGHEST ISOLATION)
────────────────────────────────────────────────
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  firm_a_db  │  │  firm_b_db  │  │  firm_c_db  │
│  ┌───────┐  │  │  ┌───────┐  │  │  ┌───────┐  │
│  │ cases │  │  │  │ cases │  │  │  │ cases │  │
│  │ docs  │  │  │  │ docs  │  │  │  │ docs  │  │
│  └───────┘  │  │  └───────┘  │  │  └───────┘  │
└─────────────┘  └─────────────┘  └─────────────┘

✓ MAXIMUM isolation (physically separate)
✓ Easy compliance proof
✓ Per-tenant backup/restore
✓ Can offer dedicated resources to big tenants
✗ Expensive (many DB instances)
✗ Operational complexity
✗ Hard to do cross-tenant analytics (if ever needed)


MODEL 4: HYBRID (Eve's Approach)
────────────────────────────────
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  PostgreSQL: Schema-per-tenant          Vector DB: Index-per-case │
│  ┌─────────────┬─────────────┐         ┌────────────────────────┐│
│  │ tenant_acme │ tenant_baker│         │ tenant_acme_case_001  ││
│  │ (schema)    │ (schema)    │         │ tenant_acme_case_002  ││
│  │             │             │         │ tenant_baker_case_001 ││
│  └─────────────┴─────────────┘         └────────────────────────┘│
│                                                                   │
│  Object Storage: Prefix-per-tenant                                │
│  s3://bucket/tenants/acme/cases/001/doc.pdf                      │
│  s3://bucket/tenants/baker/cases/001/doc.pdf                     │
└──────────────────────────────────────────────────────────────────┘

✓ Right isolation level for each data type
✓ Vector DB per-case = absolute search isolation
✓ Cost-effective (shared infra where safe)
✓ Strongest isolation where it matters most
```

### Decision Matrix

| Factor | Shared+TenantID | Schema-per-Tenant | DB-per-Tenant |
|--------|-----------------|-------------------|---------------|
| **Isolation Strength** | ⚠️ Weak | ✅ Strong | ✅✅ Strongest |
| **Compliance Proof** | ❌ Hard | ✅ Good | ✅✅ Easy |
| **Operational Cost** | ✅ Low | ⚠️ Medium | ❌ High |
| **Scale (# tenants)** | ✅ Millions | ✅ Thousands | ⚠️ Hundreds |
| **Data Export** | ⚠️ Complex | ✅ Easy | ✅ Easy |
| **Customization** | ❌ Hard | ✅ Possible | ✅✅ Full |
| **For Legal SaaS** | ❌ No | ✅ Yes | ✅ Enterprise |

**Recommendation for Eve Legal: Schema-per-tenant + Index-per-case**

---

## 3. Recommended Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TENANT LEGAL SAAS ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              INTERNET
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            EDGE LAYER                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ CloudFlare  │  │    WAF      │  │  DDoS       │  │   SSL/TLS   │        │
│  │    CDN      │  │             │  │  Protection │  │ Termination │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTHENTICATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AUTH SERVICE                                 │   │
│  │  • JWT validation          • Tenant ID extraction                   │   │
│  │  • SSO/SAML integration    • MFA enforcement                        │   │
│  │  • Session management      • IP allowlist (optional)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      API GATEWAY + TENANT ROUTING                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TENANT-AWARE GATEWAY                            │   │
│  │                                                                      │   │
│  │  1. Extract tenant from JWT/subdomain                               │   │
│  │  2. Validate tenant is active                                       │   │
│  │  3. Set tenant context for request                                  │   │
│  │  4. Apply tenant-specific rate limits                               │   │
│  │  5. Route to appropriate service                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│     STANDARD TENANTS        │   │    ENTERPRISE TENANTS       │
│     (Shared Compute)        │   │    (Dedicated Resources)    │
│  ┌───────────────────────┐  │   │  ┌───────────────────────┐  │
│  │   API Service Pool    │  │   │  │  Dedicated API Pods   │  │
│  │   (Kubernetes)        │  │   │  │  (Isolated Namespace) │  │
│  └───────────────────────┘  │   │  └───────────────────────┘  │
└─────────────────────────────┘   └─────────────────────────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                          │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │    PostgreSQL       │  │    Turbopuffer      │  │       S3            │ │
│  │    (RDS Multi-AZ)   │  │    (Vector DB)      │  │  (Object Storage)   │ │
│  │                     │  │                     │  │                     │ │
│  │  Schema-per-tenant  │  │  Index-per-case     │  │  Prefix-per-tenant  │ │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────────┐  │ │
│  │  │ tenant_acme   │  │  │  │ acme_case_001 │  │  │  │ /acme/...     │  │ │
│  │  │ tenant_baker  │  │  │  │ acme_case_002 │  │  │  │ /baker/...    │  │ │
│  │  │ tenant_cohen  │  │  │  │ baker_case_001│  │  │  │ /cohen/...    │  │ │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  └───────────────┘  │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │       Redis         │  │        KMS          │  │    Audit Log        │ │
│  │    (Cache/Queue)    │  │   (Key Management)  │  │  (Immutable Store)  │ │
│  │                     │  │                     │  │                     │ │
│  │  Tenant-prefixed    │  │  Key-per-tenant     │  │  Append-only        │ │
│  │  cache keys         │  │  + Key-per-case     │  │  CloudWatch/S3      │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow with Tenant Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REQUEST FLOW WITH TENANT ISOLATION                        │
└─────────────────────────────────────────────────────────────────────────────┘

  User Request: GET /api/cases/123
  Headers: Authorization: Bearer <JWT>
                    │
                    ▼
  ┌─────────────────────────────────────────────────────┐
  │  1. AUTH SERVICE                                     │
  │     • Validate JWT signature                        │
  │     • Extract: user_id=456, tenant_id=acme          │
  │     • Check: user belongs to tenant                 │
  │     • Check: tenant is active, not suspended        │
  └─────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────────────┐
  │  2. TENANT CONTEXT MIDDLEWARE                        │
  │     • Create TenantContext(tenant_id="acme")        │
  │     • Attach to request (thread-local/async ctx)   │
  │     • Set database schema: "tenant_acme"            │
  │     • Set S3 prefix: "tenants/acme/"                │
  │     • Set cache prefix: "acme:"                     │
  │     • Log: tenant_id in all log entries            │
  └─────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────────────┐
  │  3. BUSINESS LOGIC                                   │
  │     • Query: SELECT * FROM cases WHERE id=123       │
  │     • Automatically scoped to tenant_acme schema   │
  │     • If case 123 doesn't exist in acme → 404      │
  │     • NO WAY to accidentally query other tenant    │
  └─────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────────────┐
  │  4. AUDIT LOG                                        │
  │     • Log: user=456, tenant=acme, action=view_case │
  │     • Log: resource=case_123, ip=1.2.3.4           │
  │     • Immutable, append-only storage               │
  └─────────────────────────────────────────────────────┘
                    │
                    ▼
  Response: 200 OK + Case Data
```

---

## 4. Database Layer Design

### PostgreSQL Schema-Per-Tenant

```sql
-- ============================================================
-- TENANT MANAGEMENT (in public schema)
-- ============================================================

CREATE SCHEMA IF NOT EXISTS admin;

CREATE TABLE admin.tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(50) UNIQUE NOT NULL,  -- Used in schema name
    name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',  -- active, suspended, churned
    
    -- Subscription info
    plan VARCHAR(50) DEFAULT 'standard',  -- standard, professional, enterprise
    max_users INTEGER DEFAULT 10,
    max_cases INTEGER DEFAULT 1000,
    max_storage_gb INTEGER DEFAULT 100,
    
    -- Security settings
    encryption_key_id VARCHAR(100),  -- Reference to KMS key
    mfa_required BOOLEAN DEFAULT true,
    ip_allowlist JSONB DEFAULT '[]',
    sso_config JSONB,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE admin.tenant_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES admin.tenants(id),
    user_id UUID NOT NULL,  -- References central user service
    role VARCHAR(50) NOT NULL,  -- admin, attorney, paralegal, viewer
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(tenant_id, user_id)
);

-- ============================================================
-- TENANT SCHEMA TEMPLATE
-- (Created for each new tenant)
-- ============================================================

-- Function to create tenant schema
CREATE OR REPLACE FUNCTION admin.create_tenant_schema(tenant_slug VARCHAR)
RETURNS VOID AS $$
DECLARE
    schema_name VARCHAR := 'tenant_' || tenant_slug;
BEGIN
    -- Create schema
    EXECUTE format('CREATE SCHEMA IF NOT EXISTS %I', schema_name);
    
    -- Create tables in tenant schema
    EXECUTE format('
        CREATE TABLE %I.cases (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            external_id VARCHAR(100),
            client_name VARCHAR(255) NOT NULL,
            case_type VARCHAR(50),
            status VARCHAR(50) DEFAULT ''active'',
            
            -- No tenant_id needed! Schema IS the tenant boundary
            
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )', schema_name);
    
    EXECUTE format('
        CREATE TABLE %I.documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            case_id UUID REFERENCES %I.cases(id) ON DELETE CASCADE,
            filename VARCHAR(500),
            storage_path VARCHAR(1000),  -- S3 path within tenant prefix
            file_hash VARCHAR(64),
            
            created_at TIMESTAMPTZ DEFAULT NOW()
        )', schema_name, schema_name);
    
    EXECUTE format('
        CREATE TABLE %I.case_notes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            case_id UUID REFERENCES %I.cases(id) ON DELETE CASCADE,
            author_id UUID NOT NULL,
            content TEXT,
            
            created_at TIMESTAMPTZ DEFAULT NOW()
        )', schema_name, schema_name);
    
    -- Create indexes
    EXECUTE format('CREATE INDEX ON %I.cases(status)', schema_name);
    EXECUTE format('CREATE INDEX ON %I.documents(case_id)', schema_name);
    
    -- Row-level security as defense in depth
    EXECUTE format('ALTER TABLE %I.cases ENABLE ROW LEVEL SECURITY', schema_name);
    
    RAISE NOTICE 'Created tenant schema: %', schema_name;
END;
$$ LANGUAGE plpgsql;

-- Create a new tenant
SELECT admin.create_tenant_schema('acme_law');
SELECT admin.create_tenant_schema('baker_firm');
```

### Connection Management

```python
from contextlib import contextmanager
from contextvars import ContextVar
import psycopg2
from psycopg2 import pool

# Thread-safe tenant context
current_tenant: ContextVar[str] = ContextVar('current_tenant', default=None)

class TenantAwareConnectionPool:
    """
    Connection pool that enforces tenant isolation.
    """
    
    def __init__(self, dsn: str, min_conn: int = 10, max_conn: int = 100):
        self.pool = pool.ThreadedConnectionPool(min_conn, max_conn, dsn)
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection with tenant schema already set.
        """
        tenant_id = current_tenant.get()
        if not tenant_id:
            raise SecurityError("No tenant context set - refusing database access")
        
        conn = self.pool.getconn()
        try:
            # CRITICAL: Set schema search path to tenant's schema ONLY
            schema_name = f"tenant_{tenant_id}"
            
            with conn.cursor() as cur:
                # Set search_path to ONLY the tenant schema
                # This means unqualified table names resolve to tenant schema
                cur.execute(
                    "SET search_path TO %s",
                    (schema_name,)
                )
                
                # Also set application context for RLS policies
                cur.execute(
                    "SET app.current_tenant = %s",
                    (tenant_id,)
                )
            
            yield conn
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise
        finally:
            # Reset search path before returning to pool
            with conn.cursor() as cur:
                cur.execute("RESET search_path")
                cur.execute("RESET app.current_tenant")
            self.pool.putconn(conn)
    
    def execute(self, query: str, params: tuple = None):
        """
        Execute query in tenant context.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description:  # SELECT query
                    return cur.fetchall()
                return cur.rowcount


class TenantContextMiddleware:
    """
    FastAPI middleware to set tenant context.
    """
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract tenant from JWT (already validated by auth middleware)
            tenant_id = scope.get("state", {}).get("tenant_id")
            
            if tenant_id:
                # Set tenant context for this request
                token = current_tenant.set(tenant_id)
                try:
                    await self.app(scope, receive, send)
                finally:
                    current_tenant.reset(token)
            else:
                # No tenant context - only allow public endpoints
                await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
```

### Row-Level Security (Defense in Depth)

```sql
-- Even with schema isolation, add RLS as backup
-- This protects against bugs where wrong schema is set

-- Enable RLS on sensitive tables
ALTER TABLE tenant_acme.cases ENABLE ROW LEVEL SECURITY;

-- Create policy that checks app.current_tenant
CREATE POLICY tenant_isolation ON tenant_acme.cases
    FOR ALL
    USING (
        -- The schema name must match the current_tenant setting
        current_setting('app.current_tenant', true) = 'acme'
    );

-- This means even if someone somehow sets search_path wrong,
-- they still can't see data without the correct app.current_tenant
```

---

## 5. Application Layer Isolation

### Tenant-Aware Service Layer

```python
from dataclasses import dataclass
from typing import Optional
from contextvars import ContextVar

# Global tenant context (thread/async safe)
_tenant_context: ContextVar[Optional['TenantContext']] = ContextVar(
    'tenant_context', 
    default=None
)

@dataclass
class TenantContext:
    """
    Immutable tenant context attached to each request.
    """
    tenant_id: str
    tenant_slug: str
    user_id: str
    user_role: str
    
    # Computed paths
    @property
    def db_schema(self) -> str:
        return f"tenant_{self.tenant_slug}"
    
    @property
    def s3_prefix(self) -> str:
        return f"tenants/{self.tenant_slug}/"
    
    @property
    def cache_prefix(self) -> str:
        return f"{self.tenant_slug}:"
    
    @property
    def vector_namespace_prefix(self) -> str:
        return f"tenant_{self.tenant_slug}_"
    
    def vector_namespace(self, case_id: str) -> str:
        return f"{self.vector_namespace_prefix}case_{case_id}"


def get_tenant_context() -> TenantContext:
    """
    Get current tenant context or raise error.
    
    This should be called by ALL data access code.
    """
    ctx = _tenant_context.get()
    if ctx is None:
        raise SecurityError(
            "No tenant context available. "
            "This is a security violation - all data access must have tenant context."
        )
    return ctx


def set_tenant_context(ctx: TenantContext):
    """Set tenant context for current request."""
    return _tenant_context.set(ctx)


class TenantAwareRepository:
    """
    Base repository that enforces tenant isolation.
    """
    
    def __init__(self, db_pool: TenantAwareConnectionPool):
        self.db = db_pool
    
    @property
    def tenant(self) -> TenantContext:
        return get_tenant_context()
    
    def execute(self, query: str, params: tuple = None):
        """Execute query in tenant context."""
        # Connection automatically uses tenant's schema
        return self.db.execute(query, params)


class CaseRepository(TenantAwareRepository):
    """
    Repository for case data.
    """
    
    def get_by_id(self, case_id: str) -> Optional[dict]:
        """Get case by ID within current tenant."""
        # No need to filter by tenant_id - schema isolation handles it
        result = self.execute(
            "SELECT * FROM cases WHERE id = %s",
            (case_id,)
        )
        return result[0] if result else None
    
    def create(self, data: dict) -> dict:
        """Create case in current tenant."""
        result = self.execute(
            """
            INSERT INTO cases (client_name, case_type, status)
            VALUES (%s, %s, %s)
            RETURNING *
            """,
            (data['client_name'], data['case_type'], data.get('status', 'active'))
        )
        
        # Audit log
        self.audit_log(
            action="create_case",
            resource_type="case",
            resource_id=result[0]['id']
        )
        
        return result[0]
    
    def audit_log(self, action: str, resource_type: str, resource_id: str):
        """Log action for compliance."""
        # This goes to immutable audit store
        AuditLogger.log(
            tenant_id=self.tenant.tenant_id,
            user_id=self.tenant.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id
        )
```

### S3 Storage Isolation

```python
import boto3
from botocore.exceptions import ClientError

class TenantAwareStorage:
    """
    S3 storage with tenant prefix isolation.
    """
    
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = boto3.client('s3')
    
    @property
    def tenant(self) -> TenantContext:
        return get_tenant_context()
    
    def _tenant_key(self, key: str) -> str:
        """
        Prefix key with tenant path.
        
        Input: "cases/123/document.pdf"
        Output: "tenants/acme/cases/123/document.pdf"
        """
        # Prevent path traversal attacks
        if '..' in key or key.startswith('/'):
            raise SecurityError(f"Invalid key: {key}")
        
        return f"{self.tenant.s3_prefix}{key}"
    
    def upload(self, key: str, data: bytes, metadata: dict = None) -> str:
        """Upload file to tenant's prefix."""
        tenant_key = self._tenant_key(key)
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=tenant_key,
            Body=data,
            Metadata=metadata or {},
            ServerSideEncryption='aws:kms',
            SSEKMSKeyId=self.tenant.encryption_key_id  # Tenant-specific key
        )
        
        return tenant_key
    
    def download(self, key: str) -> bytes:
        """Download file from tenant's prefix."""
        tenant_key = self._tenant_key(key)
        
        # Extra safety: verify key starts with tenant prefix
        if not tenant_key.startswith(self.tenant.s3_prefix):
            raise SecurityError(
                f"Key {tenant_key} does not belong to tenant {self.tenant.tenant_id}"
            )
        
        response = self.s3.get_object(
            Bucket=self.bucket,
            Key=tenant_key
        )
        
        return response['Body'].read()
    
    def list_objects(self, prefix: str = "") -> list:
        """List objects within tenant's prefix."""
        tenant_prefix = self._tenant_key(prefix)
        
        response = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=tenant_prefix
        )
        
        # Strip tenant prefix from returned keys
        objects = []
        for obj in response.get('Contents', []):
            relative_key = obj['Key'].replace(self.tenant.s3_prefix, '', 1)
            objects.append({
                'key': relative_key,
                'size': obj['Size'],
                'last_modified': obj['LastModified']
            })
        
        return objects
    
    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for tenant's file."""
        tenant_key = self._tenant_key(key)
        
        return self.s3.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': self.bucket,
                'Key': tenant_key
            },
            ExpiresIn=expires_in
        )
```

### Vector Store Isolation (Per-Case)

```python
class TenantAwareVectorStore:
    """
    Vector store with per-case namespace isolation.
    
    This is critical: search results must NEVER leak across tenants or cases.
    """
    
    def __init__(self, client: TurbopufferClient):
        self.client = client
    
    @property
    def tenant(self) -> TenantContext:
        return get_tenant_context()
    
    def _namespace(self, case_id: str) -> str:
        """
        Generate namespace for a specific case.
        
        Format: tenant_{slug}_case_{case_id}
        
        This ensures:
        1. Complete isolation between tenants
        2. Complete isolation between cases within a tenant
        3. Easy deletion when case is closed
        """
        return self.tenant.vector_namespace(case_id)
    
    def index_document(self, case_id: str, chunks: list[dict]):
        """Index document chunks for a case."""
        namespace = self._namespace(case_id)
        
        vectors = [
            {
                'id': chunk['id'],
                'vector': chunk['embedding'],
                'attributes': chunk['metadata']
            }
            for chunk in chunks
        ]
        
        self.client.upsert(
            namespace=namespace,
            vectors=vectors
        )
    
    def search(
        self, 
        case_id: str, 
        query_vector: list[float],
        top_k: int = 10
    ) -> list[dict]:
        """
        Search within a specific case.
        
        CRITICAL: Search is scoped to case namespace.
        There is NO WAY to search across cases or tenants.
        """
        namespace = self._namespace(case_id)
        
        results = self.client.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k
        )
        
        return results
    
    def search_across_tenant_cases(
        self,
        case_ids: list[str],
        query_vector: list[float],
        top_k_per_case: int = 5
    ) -> dict[str, list]:
        """
        Search across multiple cases within the SAME tenant.
        
        Use case: "Find all documents mentioning 'indemnification' across my cases"
        
        This is safe because:
        1. case_ids are already filtered to current tenant (in business logic)
        2. Each namespace includes tenant prefix
        3. We search each case separately, then merge
        """
        results = {}
        
        for case_id in case_ids:
            namespace = self._namespace(case_id)
            case_results = self.client.query(
                namespace=namespace,
                vector=query_vector,
                top_k=top_k_per_case
            )
            results[case_id] = case_results
        
        return results
    
    def delete_case(self, case_id: str):
        """
        Delete all vectors for a case.
        
        Called when case is closed/archived.
        """
        namespace = self._namespace(case_id)
        self.client.delete_namespace(namespace)
```

---

## 6. Encryption & Key Management

### Per-Tenant Encryption Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENCRYPTION KEY HIERARCHY                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │      AWS KMS ROOT KEY           │
                    │      (AWS Managed)              │
                    └─────────────────┬───────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │  Tenant Acme    │ │  Tenant Baker   │ │  Tenant Cohen   │
          │  Master Key     │ │  Master Key     │ │  Master Key     │
          │  (CMK in KMS)   │ │  (CMK in KMS)   │ │  (CMK in KMS)   │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   │                   │                   │
         ┌─────────┴─────────┐         │                   │
         ▼                   ▼         ▼                   ▼
   ┌───────────┐      ┌───────────┐
   │ Data Key  │      │ Data Key  │    ... (similar structure)
   │ (S3 Docs) │      │ (DB)      │
   └───────────┘      └───────────┘

Benefits:
• Tenant Acme's key cannot decrypt Tenant Baker's data
• Key rotation per tenant (compliance requirement)
• Key deletion = cryptographic shredding of tenant data
• Audit trail per key
```

### Key Management Implementation

```python
import boto3
from dataclasses import dataclass
from typing import Optional

@dataclass
class TenantKeyInfo:
    tenant_id: str
    kms_key_id: str  # AWS KMS CMK ARN
    kms_key_alias: str
    created_at: str
    last_rotated: str
    
class TenantKeyManager:
    """
    Manage per-tenant encryption keys in AWS KMS.
    """
    
    def __init__(self):
        self.kms = boto3.client('kms')
    
    def create_tenant_key(self, tenant_id: str, tenant_name: str) -> TenantKeyInfo:
        """
        Create a new CMK for a tenant.
        """
        # Create the key
        response = self.kms.create_key(
            Description=f"Master key for tenant: {tenant_name}",
            KeyUsage='ENCRYPT_DECRYPT',
            Origin='AWS_KMS',
            Tags=[
                {'TagKey': 'tenant_id', 'TagValue': tenant_id},
                {'TagKey': 'purpose', 'TagValue': 'tenant_data_encryption'},
            ],
            # Enable automatic key rotation (annual)
            # For more frequent rotation, use custom rotation
        )
        
        key_id = response['KeyMetadata']['KeyId']
        key_arn = response['KeyMetadata']['Arn']
        
        # Create an alias for easier reference
        alias = f"alias/tenant/{tenant_id}"
        self.kms.create_alias(
            AliasName=alias,
            TargetKeyId=key_id
        )
        
        # Enable automatic rotation
        self.kms.enable_key_rotation(KeyId=key_id)
        
        return TenantKeyInfo(
            tenant_id=tenant_id,
            kms_key_id=key_arn,
            kms_key_alias=alias,
            created_at=response['KeyMetadata']['CreationDate'].isoformat(),
            last_rotated=response['KeyMetadata']['CreationDate'].isoformat()
        )
    
    def get_tenant_key(self, tenant_id: str) -> str:
        """Get KMS key ARN for tenant."""
        alias = f"alias/tenant/{tenant_id}"
        
        response = self.kms.describe_key(KeyId=alias)
        return response['KeyMetadata']['Arn']
    
    def encrypt_for_tenant(self, tenant_id: str, plaintext: bytes) -> bytes:
        """
        Encrypt data using tenant's key.
        """
        key_arn = self.get_tenant_key(tenant_id)
        
        response = self.kms.encrypt(
            KeyId=key_arn,
            Plaintext=plaintext,
            EncryptionContext={
                'tenant_id': tenant_id  # Bound to tenant context
            }
        )
        
        return response['CiphertextBlob']
    
    def decrypt_for_tenant(self, tenant_id: str, ciphertext: bytes) -> bytes:
        """
        Decrypt data using tenant's key.
        
        CRITICAL: Decryption will FAIL if wrong tenant_id is provided,
        because encryption context must match.
        """
        key_arn = self.get_tenant_key(tenant_id)
        
        response = self.kms.decrypt(
            KeyId=key_arn,
            CiphertextBlob=ciphertext,
            EncryptionContext={
                'tenant_id': tenant_id  # Must match encryption context
            }
        )
        
        return response['Plaintext']
    
    def schedule_key_deletion(self, tenant_id: str, days: int = 30):
        """
        Schedule tenant key for deletion (cryptographic shredding).
        
        Called when tenant churns. After deletion, their data
        is cryptographically unrecoverable.
        """
        key_arn = self.get_tenant_key(tenant_id)
        
        self.kms.schedule_key_deletion(
            KeyId=key_arn,
            PendingWindowInDays=days
        )
        
        # Also disable immediately to prevent new encryption
        self.kms.disable_key(KeyId=key_arn)
```

### Encryption Context for Additional Safety

```python
class SecureEncryptionContext:
    """
    Encryption contexts bind ciphertext to specific usage.
    
    Even if someone gets the key, they can't decrypt data
    encrypted with a different context.
    """
    
    @staticmethod
    def for_document(tenant_id: str, case_id: str, document_id: str) -> dict:
        """Context for encrypting a document."""
        return {
            'tenant_id': tenant_id,
            'case_id': case_id,
            'document_id': document_id,
            'data_type': 'document'
        }
    
    @staticmethod
    def for_database_field(tenant_id: str, table: str, field: str) -> dict:
        """Context for encrypting a database field."""
        return {
            'tenant_id': tenant_id,
            'table': table,
            'field': field,
            'data_type': 'database_field'
        }


# Usage example
class SecureDocumentStorage:
    def store_document(
        self, 
        case_id: str, 
        document_id: str, 
        content: bytes
    ) -> str:
        tenant = get_tenant_context()
        
        # Create specific encryption context
        context = SecureEncryptionContext.for_document(
            tenant_id=tenant.tenant_id,
            case_id=case_id,
            document_id=document_id
        )
        
        # Encrypt with context
        encrypted = self.kms.encrypt(
            KeyId=tenant.encryption_key_id,
            Plaintext=content,
            EncryptionContext=context
        )
        
        # Store encrypted blob
        return self.s3.upload(
            key=f"cases/{case_id}/documents/{document_id}",
            data=encrypted['CiphertextBlob'],
            metadata={'encryption_context': json.dumps(context)}
        )
```

---

## 7. Audit Logging System

### Comprehensive Audit Trail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIT LOGGING ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

   Application Events                     Audit Pipeline
         │                                     │
         ▼                                     ▼
┌─────────────────┐                  ┌─────────────────────┐
│  Audit Logger   │                  │  Kinesis Firehose   │
│  (async, non-   │─────────────────▶│  (buffered stream)  │
│   blocking)     │                  └──────────┬──────────┘
└─────────────────┘                             │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                          ┌─────────────────┐     ┌─────────────────┐
                          │    S3 Bucket    │     │  CloudWatch     │
                          │  (Long-term)    │     │  Logs Insights  │
                          │                 │     │  (Real-time)    │
                          │  • Immutable    │     │                 │
                          │  • 7-year       │     │  • Queries      │
                          │    retention    │     │  • Alerts       │
                          │  • Compliance   │     │  • Dashboards   │
                          └─────────────────┘     └─────────────────┘

Key Properties:
• IMMUTABLE: Logs cannot be modified after creation
• COMPLETE: Every data access is logged
• TAMPER-EVIDENT: Checksums detect any modification
• QUERYABLE: Can search for specific actions
• RETAINED: 7+ years for legal compliance
```

### Audit Log Schema

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
import json
import hashlib

@dataclass
class AuditEvent:
    """
    Comprehensive audit event for legal compliance.
    """
    # Identity
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Tenant context
    tenant_id: str = ""
    tenant_name: str = ""
    
    # Actor
    user_id: str = ""
    user_email: str = ""
    user_role: str = ""
    
    # Action
    action: str = ""  # e.g., "view_document", "download_file", "export_case"
    action_category: str = ""  # "read", "write", "delete", "export"
    
    # Resource
    resource_type: str = ""  # "case", "document", "user"
    resource_id: str = ""
    resource_name: str = ""
    
    # Request context
    ip_address: str = ""
    user_agent: str = ""
    request_id: str = ""
    api_endpoint: str = ""
    
    # Result
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    previous_hash: str = ""  # Hash of previous event (chain)
    
    def to_json(self) -> str:
        return json.dumps({
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'tenant_id': self.tenant_id,
            'tenant_name': self.tenant_name,
            'user_id': self.user_id,
            'user_email': self.user_email,
            'user_role': self.user_role,
            'action': self.action,
            'action_category': self.action_category,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'resource_name': self.resource_name,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id,
            'api_endpoint': self.api_endpoint,
            'success': self.success,
            'error_code': self.error_code,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'previous_hash': self.previous_hash,
        })
    
    def compute_hash(self) -> str:
        """Compute hash for integrity verification."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()


class AuditLogger:
    """
    High-performance audit logger.
    """
    
    def __init__(self, kinesis_client, stream_name: str):
        self.kinesis = kinesis_client
        self.stream_name = stream_name
        self._last_hash = ""
    
    async def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        metadata: dict = None
    ):
        """
        Log an audit event asynchronously.
        
        Non-blocking to avoid impacting application performance.
        """
        tenant = get_tenant_context()
        
        event = AuditEvent(
            tenant_id=tenant.tenant_id,
            tenant_name=tenant.tenant_slug,
            user_id=tenant.user_id,
            user_role=tenant.user_role,
            action=action,
            action_category=self._categorize_action(action),
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=get_request_ip(),
            user_agent=get_user_agent(),
            request_id=get_request_id(),
            success=success,
            metadata=metadata or {},
            previous_hash=self._last_hash,
        )
        
        # Update hash chain
        self._last_hash = event.compute_hash()
        
        # Send to Kinesis (async, buffered)
        await self._send_to_kinesis(event)
    
    async def _send_to_kinesis(self, event: AuditEvent):
        """Send event to Kinesis stream."""
        self.kinesis.put_record(
            StreamName=self.stream_name,
            Data=event.to_json(),
            PartitionKey=event.tenant_id  # Partition by tenant
        )
    
    def _categorize_action(self, action: str) -> str:
        """Categorize action for reporting."""
        if action.startswith(('view_', 'get_', 'list_', 'search_')):
            return 'read'
        elif action.startswith(('create_', 'update_', 'upload_')):
            return 'write'
        elif action.startswith(('delete_', 'remove_')):
            return 'delete'
        elif action.startswith(('export_', 'download_')):
            return 'export'
        return 'other'


# Decorator for automatic audit logging
def audit_logged(action: str, resource_type: str):
    """Decorator to automatically log function calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            resource_id = kwargs.get('id') or (args[1] if len(args) > 1 else 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                await AuditLogger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id),
                    success=True
                )
                return result
            except Exception as e:
                await AuditLogger.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=str(resource_id),
                    success=False,
                    metadata={'error': str(e)}
                )
                raise
        return wrapper
    return decorator


# Usage
class CaseService:
    @audit_logged(action="view_case", resource_type="case")
    async def get_case(self, id: str):
        return await self.repository.get_by_id(id)
    
    @audit_logged(action="export_case", resource_type="case")
    async def export_case(self, id: str, format: str):
        # This is a sensitive action - automatically logged
        return await self.exporter.export(id, format)
```

### Audit Query Examples

```sql
-- CloudWatch Logs Insights queries

-- 1. All document access for a specific case (for privilege review)
fields @timestamp, user_email, action, resource_id
| filter tenant_id = 'acme' 
| filter resource_type = 'document'
| filter metadata.case_id = 'case_123'
| sort @timestamp desc
| limit 1000

-- 2. All export actions (potential data exfiltration)
fields @timestamp, tenant_id, user_email, action, resource_type, resource_id
| filter action_category = 'export'
| sort @timestamp desc
| limit 100

-- 3. Failed access attempts (security monitoring)
fields @timestamp, tenant_id, user_email, action, error_message, ip_address
| filter success = false
| sort @timestamp desc
| limit 100

-- 4. Cross-tenant access attempts (should be 0!)
fields @timestamp, user_id, tenant_id, action, resource_type
| filter metadata.attempted_tenant != tenant_id
| sort @timestamp desc

-- 5. User activity report (for offboarding)
fields @timestamp, action, resource_type, resource_id
| filter tenant_id = 'acme'
| filter user_id = 'user_456'
| stats count() by action
```

---

## 8. Resource Isolation & Quotas

### The Noisy Neighbor Problem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NOISY NEIGHBOR SCENARIOS                                │
└─────────────────────────────────────────────────────────────────────────────┘

SCENARIO 1: CPU Hog
───────────────────
Tenant A runs a massive document processing job.
All API pods are busy processing Tenant A's requests.
Tenant B's simple "get case" request times out.

SCENARIO 2: Database Overload
─────────────────────────────
Tenant A runs expensive analytics query (full table scan).
PostgreSQL is busy, connection pool exhausted.
Tenant B can't save their document.

SCENARIO 3: Storage Abuse
─────────────────────────
Tenant A uploads 10TB of video files.
S3 costs spike, affecting pricing for everyone.
Or: S3 rate limits kick in, affecting all tenants.

SOLUTION: Resource Quotas + Isolation
─────────────────────────────────────
• Per-tenant rate limits
• Per-tenant resource quotas
• Priority queues
• Dedicated resources for enterprise tenants
```

### Rate Limiting Implementation

```python
from dataclasses import dataclass
from enum import Enum
import redis
import time

class RateLimitTier(Enum):
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Quotas per tenant tier."""
    
    # API rate limits (requests per minute)
    api_rpm: int
    
    # Heavy operation limits
    document_uploads_per_hour: int
    ai_queries_per_hour: int
    exports_per_day: int
    
    # Storage limits
    max_storage_gb: int
    max_file_size_mb: int
    
    # Concurrent limits
    max_concurrent_uploads: int
    max_concurrent_ai_queries: int


TIER_QUOTAS = {
    RateLimitTier.STANDARD: TenantQuotas(
        api_rpm=100,
        document_uploads_per_hour=50,
        ai_queries_per_hour=100,
        exports_per_day=10,
        max_storage_gb=100,
        max_file_size_mb=50,
        max_concurrent_uploads=5,
        max_concurrent_ai_queries=3,
    ),
    RateLimitTier.PROFESSIONAL: TenantQuotas(
        api_rpm=500,
        document_uploads_per_hour=200,
        ai_queries_per_hour=500,
        exports_per_day=50,
        max_storage_gb=500,
        max_file_size_mb=100,
        max_concurrent_uploads=10,
        max_concurrent_ai_queries=10,
    ),
    RateLimitTier.ENTERPRISE: TenantQuotas(
        api_rpm=2000,
        document_uploads_per_hour=1000,
        ai_queries_per_hour=2000,
        exports_per_day=500,
        max_storage_gb=5000,
        max_file_size_mb=500,
        max_concurrent_uploads=50,
        max_concurrent_ai_queries=50,
    ),
}


class TenantRateLimiter:
    """
    Per-tenant rate limiting using Redis.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def check_rate_limit(
        self, 
        tenant_id: str, 
        operation: str, 
        limit: int, 
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if operation is within rate limit.
        
        Returns: (allowed: bool, remaining: int)
        """
        key = f"ratelimit:{tenant_id}:{operation}"
        
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - window_seconds
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add new entry
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, window_seconds)
        
        results = pipe.execute()
        current_count = results[1]
        
        if current_count >= limit:
            return False, 0
        
        return True, limit - current_count - 1
    
    def check_concurrent_limit(
        self,
        tenant_id: str,
        operation: str,
        limit: int
    ) -> bool:
        """
        Check concurrent operation limit using Redis semaphore.
        """
        key = f"concurrent:{tenant_id}:{operation}"
        
        # Try to acquire semaphore
        current = self.redis.incr(key)
        
        if current > limit:
            # Over limit, release
            self.redis.decr(key)
            return False
        
        # Set expiry (auto-release after timeout)
        self.redis.expire(key, 300)  # 5 minute timeout
        return True
    
    def release_concurrent(self, tenant_id: str, operation: str):
        """Release concurrent operation slot."""
        key = f"concurrent:{tenant_id}:{operation}"
        self.redis.decr(key)


class RateLimitMiddleware:
    """
    Middleware that enforces tenant rate limits.
    """
    
    def __init__(self, limiter: TenantRateLimiter):
        self.limiter = limiter
    
    async def __call__(self, request, call_next):
        tenant = get_tenant_context()
        quotas = TIER_QUOTAS[tenant.tier]
        
        # Check API rate limit
        allowed, remaining = self.limiter.check_rate_limit(
            tenant_id=tenant.tenant_id,
            operation="api",
            limit=quotas.api_rpm,
            window_seconds=60
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "API rate limit exceeded. Please slow down.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(quotas.api_rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
```

### Resource Quotas Enforcement

```python
class StorageQuotaEnforcer:
    """
    Enforce storage quotas per tenant.
    """
    
    def __init__(self, db: Database, s3: S3Client):
        self.db = db
        self.s3 = s3
    
    async def check_upload_allowed(
        self, 
        tenant_id: str, 
        file_size_bytes: int
    ) -> tuple[bool, str]:
        """
        Check if tenant can upload a file of given size.
        """
        tenant = await self.db.get_tenant(tenant_id)
        quotas = TIER_QUOTAS[tenant.tier]
        
        # Check file size
        max_file_bytes = quotas.max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_file_bytes:
            return False, f"File exceeds maximum size of {quotas.max_file_size_mb}MB"
        
        # Check total storage
        current_usage = await self.get_storage_usage(tenant_id)
        max_storage_bytes = quotas.max_storage_gb * 1024 * 1024 * 1024
        
        if current_usage + file_size_bytes > max_storage_bytes:
            return False, f"Storage quota exceeded. Current: {current_usage / 1e9:.1f}GB, Max: {quotas.max_storage_gb}GB"
        
        return True, ""
    
    async def get_storage_usage(self, tenant_id: str) -> int:
        """Get current storage usage for tenant."""
        # Use cached value if available
        cache_key = f"storage_usage:{tenant_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            return int(cached)
        
        # Calculate from S3
        prefix = f"tenants/{tenant_id}/"
        total_size = 0
        
        paginator = self.s3.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                total_size += obj['Size']
        
        # Cache for 1 hour
        await self.redis.set(cache_key, str(total_size), ex=3600)
        
        return total_size
```

### Priority Queues for Fair Processing

```python
class TenantPriorityQueue:
    """
    Process jobs with tenant-aware priority.
    
    Ensures:
    1. Fair processing across tenants
    2. Enterprise tenants get higher priority
    3. No single tenant can starve others
    """
    
    PRIORITY_MAP = {
        RateLimitTier.ENTERPRISE: 3,
        RateLimitTier.PROFESSIONAL: 2,
        RateLimitTier.STANDARD: 1,
    }
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def enqueue(self, tenant_id: str, tier: RateLimitTier, job: dict):
        """
        Enqueue job with tenant-aware priority.
        """
        priority = self.PRIORITY_MAP[tier]
        queue_key = f"job_queue:priority_{priority}"
        
        # Also track per-tenant queue depth (for fairness)
        tenant_queue_key = f"job_queue:tenant:{tenant_id}"
        
        job_data = json.dumps({
            'tenant_id': tenant_id,
            'job': job,
            'enqueued_at': time.time()
        })
        
        pipe = self.redis.pipeline()
        pipe.lpush(queue_key, job_data)
        pipe.incr(tenant_queue_key)
        pipe.execute()
    
    def dequeue(self) -> Optional[dict]:
        """
        Dequeue next job, respecting priority and fairness.
        """
        # Try high priority first
        for priority in [3, 2, 1]:
            queue_key = f"job_queue:priority_{priority}"
            
            # BRPOP with timeout
            result = self.redis.brpop(queue_key, timeout=1)
            if result:
                job_data = json.loads(result[1])
                
                # Decrement tenant queue depth
                tenant_queue_key = f"job_queue:tenant:{job_data['tenant_id']}"
                self.redis.decr(tenant_queue_key)
                
                return job_data
        
        return None
```

---

## 9. Compliance Framework

### SOC 2 Type II Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SOC 2 TRUST SERVICES CRITERIA                            │
└─────────────────────────────────────────────────────────────────────────────┘

SECURITY (Common Criteria)
─────────────────────────
✓ Access controls (RBAC, MFA)
✓ Encryption at rest and in transit
✓ Network security (firewalls, VPCs)
✓ Vulnerability management
✓ Incident response procedures

Our Implementation:
• Per-tenant encryption keys (KMS)
• Schema-per-tenant database isolation
• WAF and DDoS protection
• Regular penetration testing
• 24/7 security monitoring


AVAILABILITY
────────────
✓ Uptime commitments (99.9%+)
✓ Disaster recovery
✓ Backup procedures
✓ Capacity planning

Our Implementation:
• Multi-AZ deployment
• Automated failover
• Daily backups with 30-day retention
• Auto-scaling based on load


CONFIDENTIALITY
──────────────
✓ Data classification
✓ Access restrictions
✓ Data retention/disposal
✓ Third-party agreements

Our Implementation:
• All data classified as confidential
• Tenant isolation prevents cross-access
• Per-tenant data deletion on churn
• DPA with all sub-processors


PROCESSING INTEGRITY
────────────────────
✓ Data accuracy
✓ Completeness
✓ Timely processing

Our Implementation:
• Input validation on all data
• Audit logging of all changes
• Monitoring and alerting


PRIVACY
───────
✓ Notice and consent
✓ Data minimization
✓ Right to access/delete

Our Implementation:
• Privacy policy in ToS
• Data export API for tenants
• Full data deletion on request
```

### HIPAA Compliance (For Medical Records)

```python
class HIPAAComplianceLayer:
    """
    Additional protections for PHI (Protected Health Information).
    
    Required when law firms handle medical records in PI cases.
    """
    
    # PHI data types that require extra protection
    PHI_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{9}\b',              # SSN without dashes
        r'\b[A-Z]{2}\d{6,8}\b',    # Driver's license (varies by state)
        r'\b\d{10}\b',             # MRN (Medical Record Number)
    ]
    
    def detect_phi(self, text: str) -> List[dict]:
        """Detect potential PHI in text."""
        findings = []
        for pattern in self.PHI_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                findings.append({
                    'type': 'potential_phi',
                    'pattern': pattern,
                    'location': match.span()
                })
        return findings
    
    def encrypt_phi_fields(self, record: dict, phi_fields: List[str]) -> dict:
        """
        Encrypt specific PHI fields with additional layer.
        
        For medical records, we use double encryption:
        1. Tenant key (standard)
        2. PHI-specific key (additional protection)
        """
        tenant = get_tenant_context()
        
        for field in phi_fields:
            if field in record and record[field]:
                # Encrypt with PHI-specific key
                encrypted = self.kms.encrypt(
                    KeyId=self.phi_key_arn,
                    Plaintext=record[field].encode(),
                    EncryptionContext={
                        'tenant_id': tenant.tenant_id,
                        'data_type': 'phi',
                        'field': field
                    }
                )
                record[field] = base64.b64encode(encrypted['CiphertextBlob']).decode()
        
        return record
    
    def audit_phi_access(self, record_type: str, record_id: str, fields_accessed: List[str]):
        """
        Enhanced audit logging for PHI access.
        
        HIPAA requires detailed logging of all PHI access.
        """
        AuditLogger.log(
            action="access_phi",
            resource_type=record_type,
            resource_id=record_id,
            metadata={
                'phi_fields_accessed': fields_accessed,
                'access_reason': 'case_processing',  # Could be from request
                'hipaa_event': True
            }
        )
```

### Compliance Reporting

```python
class ComplianceReporter:
    """
    Generate compliance reports for auditors and tenants.
    """
    
    async def generate_access_report(
        self, 
        tenant_id: str, 
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        Generate access report for compliance audit.
        """
        # Query audit logs
        events = await self.audit_store.query(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            'report_type': 'access_audit',
            'tenant_id': tenant_id,
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': len(events),
                'unique_users': len(set(e['user_id'] for e in events)),
                'events_by_category': self._count_by_category(events),
                'failed_access_attempts': len([e for e in events if not e['success']]),
            },
            'details': events
        }
    
    async def generate_data_inventory(self, tenant_id: str) -> dict:
        """
        Generate inventory of all tenant data.
        
        Required for data subject access requests.
        """
        return {
            'report_type': 'data_inventory',
            'tenant_id': tenant_id,
            'generated_at': datetime.utcnow().isoformat(),
            'data_stores': {
                'database': {
                    'schema': f'tenant_{tenant_id}',
                    'tables': await self._list_tables(tenant_id),
                    'row_counts': await self._count_rows(tenant_id),
                },
                'object_storage': {
                    'prefix': f'tenants/{tenant_id}/',
                    'total_objects': await self._count_s3_objects(tenant_id),
                    'total_size_gb': await self._get_s3_size(tenant_id),
                },
                'vector_store': {
                    'namespaces': await self._list_vector_namespaces(tenant_id),
                    'total_vectors': await self._count_vectors(tenant_id),
                }
            }
        }
```

---

## 10. Scale & Operations

### Scaling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SCALING BY TENANT TIER                                 │
└─────────────────────────────────────────────────────────────────────────────┘

STANDARD TENANTS (Shared Resources)
───────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SHARED KUBERNETES CLUSTER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  API Pod 1  │  │  API Pod 2  │  │  API Pod 3  │  │  API Pod N  │        │
│  │  (all std   │  │  (all std   │  │  (all std   │  │  (all std   │        │
│  │   tenants)  │  │   tenants)  │  │   tenants)  │  │   tenants)  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
│  Shared RDS (Multi-tenant schemas)                                           │
│  Shared Redis Cluster                                                        │
│  Shared S3 Bucket (prefix isolation)                                         │
└─────────────────────────────────────────────────────────────────────────────┘


ENTERPRISE TENANTS (Dedicated Resources)
────────────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEDICATED KUBERNETES NAMESPACE                            │
│                         (tenant: biglaw_llp)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ Dedicated   │  │ Dedicated   │  │ Dedicated   │                         │
│  │ API Pods    │  │ Workers     │  │ Cache       │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                                                              │
│  Option A: Dedicated RDS Instance                                            │
│  Option B: Dedicated Schema + Read Replica                                   │
│  Dedicated S3 Bucket (optional)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Database Scaling

```python
class TenantDatabaseRouter:
    """
    Route database connections based on tenant tier.
    """
    
    def __init__(self):
        # Shared database for standard tenants
        self.shared_pool = ConnectionPool(
            dsn=config.SHARED_DB_DSN,
            min_size=20,
            max_size=100
        )
        
        # Dedicated connections for enterprise tenants
        self.enterprise_pools: Dict[str, ConnectionPool] = {}
    
    def get_connection(self, tenant: TenantContext):
        """Get appropriate connection for tenant."""
        
        if tenant.tier == RateLimitTier.ENTERPRISE:
            # Check if tenant has dedicated database
            if tenant.tenant_id in self.enterprise_pools:
                return self.enterprise_pools[tenant.tenant_id].acquire()
            
            # Check if tenant has dedicated read replica
            if tenant.dedicated_read_replica:
                return self._get_read_replica_connection(tenant)
        
        # Default: shared pool
        return self.shared_pool.acquire()
    
    def provision_dedicated_database(self, tenant_id: str):
        """
        Provision dedicated RDS instance for enterprise tenant.
        
        Called when tenant upgrades to enterprise.
        """
        # 1. Create RDS instance via AWS API
        rds_instance = self.rds.create_db_instance(
            DBInstanceIdentifier=f"eve-{tenant_id}",
            DBInstanceClass="db.r5.large",
            Engine="postgres",
            # ... other config
        )
        
        # 2. Migrate tenant data
        self._migrate_tenant_data(tenant_id, rds_instance['Endpoint'])
        
        # 3. Create connection pool
        self.enterprise_pools[tenant_id] = ConnectionPool(
            dsn=f"postgresql://{rds_instance['Endpoint']}/eve",
            min_size=5,
            max_size=50
        )
        
        # 4. Update tenant routing
        self._update_tenant_routing(tenant_id, rds_instance['Endpoint'])
```

### Monitoring & Alerting

```yaml
# Prometheus alerts for multi-tenant system

groups:
  - name: tenant_isolation
    rules:
      # Alert if cross-tenant access is detected
      - alert: CrossTenantAccessAttempt
        expr: sum(rate(cross_tenant_access_attempts_total[5m])) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Cross-tenant access attempt detected"
          description: "Potential security breach - immediate investigation required"
      
      # Alert if tenant quota is near limit
      - alert: TenantStorageQuotaNearLimit
        expr: (tenant_storage_used_bytes / tenant_storage_quota_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tenant {{ $labels.tenant_id }} is near storage quota"
      
      # Alert if tenant is being rate limited excessively
      - alert: TenantExcessiveRateLimiting
        expr: sum(rate(tenant_rate_limit_exceeded_total[15m])) by (tenant_id) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Tenant {{ $labels.tenant_id }} hitting rate limits frequently"
      
      # Alert if encryption key access fails
      - alert: TenantKeyAccessFailure
        expr: sum(rate(kms_decrypt_errors_total[5m])) by (tenant_id) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Encryption key access failing for tenant {{ $labels.tenant_id }}"

  - name: tenant_health
    rules:
      # Per-tenant error rate
      - alert: TenantHighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (tenant_id)
          /
          sum(rate(http_requests_total[5m])) by (tenant_id) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate for tenant {{ $labels.tenant_id }}"
      
      # Per-tenant latency
      - alert: TenantHighLatency
        expr: |
          histogram_quantile(0.95, 
            sum(rate(http_request_duration_seconds_bucket[5m])) by (tenant_id, le)
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency for tenant {{ $labels.tenant_id }}"
```

---

## 11. Disaster Recovery & Data Sovereignty

### Backup Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKUP ARCHITECTURE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │          PRODUCTION DATA            │
                    └─────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │  PostgreSQL   │       │      S3       │       │  Vector DB    │
    │               │       │               │       │               │
    │ • Continuous  │       │ • Versioning  │       │ • Daily       │
    │   WAL archive │       │   enabled     │       │   exports     │
    │ • Daily snap  │       │ • Cross-region│       │ • To S3       │
    │ • 30-day ret  │       │   replication │       │               │
    └───────────────┘       └───────────────┘       └───────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────┐
                    │       DISASTER RECOVERY SITE        │
                    │           (us-west-2)               │
                    │                                     │
                    │  • Cross-region RDS replica         │
                    │  • S3 cross-region replication      │
                    │  • Standby Kubernetes cluster       │
                    │  • DNS failover configured          │
                    └─────────────────────────────────────┘

Recovery Objectives:
• RPO (Recovery Point Objective): 1 hour
• RTO (Recovery Time Objective): 4 hours
```

### Per-Tenant Backup & Restore

```python
class TenantBackupManager:
    """
    Manage per-tenant backups for:
    - Data portability (export for tenant)
    - Compliance (prove we can restore specific tenant)
    - Disaster recovery (restore single tenant without affecting others)
    """
    
    async def create_tenant_backup(self, tenant_id: str) -> str:
        """
        Create complete backup of tenant data.
        
        Returns backup ID for restore.
        """
        backup_id = f"backup_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # 1. Backup PostgreSQL schema
        schema_name = f"tenant_{tenant_id}"
        pg_dump_cmd = f"pg_dump -n {schema_name} -Fc -f /tmp/{backup_id}_db.dump"
        await self._run_command(pg_dump_cmd)
        
        # 2. Backup S3 objects
        s3_prefix = f"tenants/{tenant_id}/"
        await self._sync_s3_to_backup(s3_prefix, backup_id)
        
        # 3. Export vector store namespaces
        namespaces = await self._list_tenant_namespaces(tenant_id)
        for ns in namespaces:
            await self._export_namespace(ns, backup_id)
        
        # 4. Store backup metadata
        backup_metadata = {
            'backup_id': backup_id,
            'tenant_id': tenant_id,
            'created_at': datetime.utcnow().isoformat(),
            'components': ['database', 's3', 'vector_store'],
            'encryption_key_id': await self._get_tenant_key(tenant_id),
        }
        
        await self._store_backup_metadata(backup_metadata)
        
        return backup_id
    
    async def restore_tenant(self, backup_id: str, target_tenant_id: str = None):
        """
        Restore tenant from backup.
        
        Can restore to same tenant (disaster recovery) or 
        different tenant (data migration).
        """
        metadata = await self._get_backup_metadata(backup_id)
        source_tenant = metadata['tenant_id']
        target_tenant = target_tenant_id or source_tenant
        
        # 1. Restore PostgreSQL
        target_schema = f"tenant_{target_tenant}"
        await self._restore_pg_dump(backup_id, target_schema)
        
        # 2. Restore S3
        target_prefix = f"tenants/{target_tenant}/"
        await self._restore_s3(backup_id, target_prefix)
        
        # 3. Restore vector store
        await self._restore_vector_namespaces(backup_id, target_tenant)
        
        # 4. Log restoration for audit
        AuditLogger.log(
            action="restore_tenant",
            resource_type="tenant",
            resource_id=target_tenant,
            metadata={
                'backup_id': backup_id,
                'source_tenant': source_tenant
            }
        )
    
    async def export_for_tenant(self, tenant_id: str) -> str:
        """
        Export all tenant data in portable format.
        
        Used for:
        - Data subject access requests
        - Tenant moving to different system
        - Legal discovery
        """
        export_id = f"export_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        export_data = {
            'tenant_id': tenant_id,
            'exported_at': datetime.utcnow().isoformat(),
            'cases': await self._export_cases(tenant_id),
            'documents': await self._export_documents(tenant_id),
            'users': await self._export_users(tenant_id),
            'audit_logs': await self._export_audit_logs(tenant_id),
        }
        
        # Create encrypted export file
        export_path = f"exports/{export_id}.json.enc"
        await self._create_encrypted_export(export_data, export_path, tenant_id)
        
        return export_id
```

### Data Sovereignty

```python
class DataSovereigntyManager:
    """
    Handle data residency requirements.
    
    Some tenants (especially international firms) require
    data to stay in specific geographic regions.
    """
    
    REGION_CONFIG = {
        'us': {
            'primary_region': 'us-east-1',
            'dr_region': 'us-west-2',
            'rds_endpoint': 'eve-us.xxx.us-east-1.rds.amazonaws.com',
            's3_bucket': 'eve-data-us',
        },
        'eu': {
            'primary_region': 'eu-west-1',
            'dr_region': 'eu-central-1',
            'rds_endpoint': 'eve-eu.xxx.eu-west-1.rds.amazonaws.com',
            's3_bucket': 'eve-data-eu',
        },
        'uk': {
            'primary_region': 'eu-west-2',
            'dr_region': 'eu-west-1',
            'rds_endpoint': 'eve-uk.xxx.eu-west-2.rds.amazonaws.com',
            's3_bucket': 'eve-data-uk',
        }
    }
    
    def get_tenant_region_config(self, tenant_id: str) -> dict:
        """Get region configuration for tenant."""
        tenant = self.db.get_tenant(tenant_id)
        return self.REGION_CONFIG.get(tenant.data_region, self.REGION_CONFIG['us'])
    
    def route_request(self, tenant_id: str) -> str:
        """Route request to correct regional cluster."""
        config = self.get_tenant_region_config(tenant_id)
        return config['primary_region']
```

---

## 12. Interview Discussion Points

### Questions They'll Ask

**Q: Why schema-per-tenant instead of database-per-tenant?**

> **A:** Balance of isolation and operational efficiency. Schema-per-tenant gives strong isolation (can't accidentally query wrong schema), easy data export (pg_dump schema), and compliance-friendly architecture. Database-per-tenant would be operationally expensive (thousands of RDS instances) and harder to manage. We add row-level security as defense-in-depth, and vector store uses per-case indices for absolute search isolation where it matters most.

**Q: How do you prevent a developer from accidentally writing a cross-tenant query?**

> **A:** Multiple layers: (1) Connection pool sets `search_path` to tenant schema before returning connection — unqualified table names only resolve to tenant's tables; (2) Application context (`app.current_tenant`) is set, enabling row-level security as backup; (3) No tenant_id columns in application tables — the schema IS the tenant boundary; (4) Code review process that flags any raw SQL; (5) Automated security scanning for queries that don't use the tenant-aware ORM.

**Q: What happens if a tenant's encryption key is compromised?**

> **A:** We use AWS KMS with automatic key rotation. If compromised: (1) Immediately disable the compromised key; (2) Create new key; (3) Re-encrypt all tenant data with new key (background job); (4) Notify tenant per incident response policy; (5) Audit logs show what data was accessed. The encryption context binding means even with the key, attacker needs correct tenant_id to decrypt.

**Q: How do you handle tenant offboarding while maintaining compliance?**

> **A:** (1) Export all tenant data (for their records); (2) Verify export integrity; (3) 30-day grace period (in case they return); (4) After grace period: DROP SCHEMA CASCADE, delete S3 prefix, delete vector namespaces; (5) Schedule KMS key deletion (7-day minimum); (6) Retain audit logs for 7 years (compliance); (7) Generate deletion certificate for tenant's records.

**Q: How do you scale the system as you add more tenants?**

> **A:** Horizontal scaling at multiple layers: (1) API tier: auto-scaling Kubernetes pods; (2) Database: read replicas for query load, connection pooling (PgBouncer); (3) Enterprise tenants get dedicated resources (isolated namespace, optionally dedicated RDS); (4) Per-tenant rate limiting prevents any single tenant from overwhelming shared resources; (5) Vector store (Turbopuffer) is serverless, scales automatically per namespace.

### Questions to Ask Them

1. **"How do you handle the case where two law firms on your platform are opposing counsel on the same case? Any special handling beyond tenant isolation?"**

2. **"For your per-case vector indices, how do you handle cross-case search within a tenant (e.g., 'find similar clauses across all my contracts')?"**

3. **"What's your approach to tenant data migration when a firm is acquired or merges with another firm on your platform?"**

4. **"How do you handle schema migrations across hundreds of tenant schemas without downtime?"**

5. **"Do you offer any 'bring your own key' (BYOK) options for enterprise tenants who want to control their encryption keys?"**

---

## Quick Reference: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **DB Isolation** | Schema-per-tenant | Strong isolation, operationally manageable |
| **Vector Isolation** | Index-per-case | Absolute search isolation for privilege |
| **Storage Isolation** | Prefix-per-tenant | Simple, effective with IAM policies |
| **Encryption** | Key-per-tenant | Cryptographic tenant boundary |
| **Rate Limiting** | Per-tenant quotas | Prevent noisy neighbor |
| **Enterprise Option** | Dedicated resources | For high-value tenants |

---

*This system design demonstrates understanding of both technical multi-tenancy patterns AND the unique legal requirements around attorney-client privilege. Focus on the defense-in-depth approach — no single layer is sufficient, but together they create robust isolation.*
