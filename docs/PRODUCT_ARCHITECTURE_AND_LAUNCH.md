# Product Strategy, Architecture, and Roadmap

The LayerForge-X ecosystem is developed with a clear architectural vision: it is established as a **comprehensive platform for layered scene representation**, centered on the canonical **Depth-Aware Amodal Layer Graph (DALG)**.

## Core Value Proposition

In the LayerForge-X paradigm, the DALG manifest serves as the **authoritative system of record**. Auxiliary outputs, such as PNG stacks, PSD-compatible folder structures, and diagnostic metrics, are treated as derived projections of the underlying graph.

The system facilitates the following core capabilities:

- **Scene Asset Synthesis:** Transformation of a single RGB bitmap into an editable, multi-layered asset.
- **Ordered Scene Representation:** Preservation of depth hierarchy, occlusion evidence, and amodal extents.
- **Mode Flexibility:** Support for native, recursive (peeling), prompt-targeted, and external (Qwen) enrichment tracks.
- **Performance Evaluation:** Automated scoring using fidelity and editability-centric metrics.
- **Standardized Export:** Generation of a stable design manifest for seamless integration into downstream creative pipelines.

## System Interfaces and Roles

The current implementation provides the core engines necessary for the following product surfaces:

| Surface | Representative Anchor | Role in Ecosystem |
| --- | --- | --- |
| **Inference Engine** | `layerforge run`, `peel`, `extract`, `transparent`, `enrich-qwen` | Synthesizes candidate DALG representations |
| **Automated Evaluator** | `self_eval.py`, `editability.py`, `run_frontier_comparison.py` | Validates and selects optimal representations |
| **Canonical Scene Artifact** | `dalg_manifest.json`, `schemas/dalg.schema.json` | Establishes a stable graph contract across all modes |
| **API Interface** | `docs/api/openapi.yaml` | Defines the control-plane contract for DALG asset delivery |
| **Design Export** | `layerforge export-design` | Generates specialized manifests for design-tool ingestion |

## Multi-Plane Architecture

LayerForge-X is architected across four distinct planes to ensure scalability and modularity:

1. **Experience Plane:** Comprises the web-based editor, collaborative review tools, and SDK documentation.
2. **Control Plane:** Manages workspaces, project lifecycle, job orchestration, and asset delivery.
3. **Inference Plane:** Houses the core decomposition engines, including native pipelines, recursive peeling, and external model enrichment (Qwen).
4. **Data and Telemetry Plane:** Manages immutable DALG assets, metrics, audit trails, and performance history.

## The DALG Manifest as a Unified Standard

The DALG manifest is designed to unify diverse decomposition methods under a single schema. This ensures that downstream consumers (e.g., web editors, creative plugins) can process a consistent scene-graph structure regardless of whether the layers were generated via native inference, recursive peeling, or external model enrichment.

## Strategic Roadmap and Target Markets

The initial deployment of LayerForge-X targets high-volume creative operations and AI-driven product teams:

1. **SaaS Preview and Export:** Establishing a web-based portal for rapid asset generation and validation.
2. **Collaborative Workflow Integration:** Implementing team-based features, including recipe sharing and comparative reviews.
3. **Enterprise Governance:** Deploying advanced security controls, audit logging, and isolated compute environments.
4. **Public SDK Release:** Exposing the DALG contract for third-party plugin development once the core schema has reached maturity.

## Implementation Status

### Current Capabilities (v1.0-Final)
- Full support for native, peeling, and Qwen-enriched inference.
- Canonical DALG export and design-manifest generation.
- Automated benchmarking suite and qualitative figure generation.
- Lightweight local web UI for result inspection.

### Future Development (Post-v1.0)
- Real-time collaborative editor integration.
- Advanced video-to-DALG decomposition.
- Fine-grained control over generative inpainting within the DAG context.


- canonical DALG schema and DALG export command;
- contract-first OpenAPI document;
- native/Qwen/peeling/extract/transparent normalization into one design-manifest JSON;
- self-evaluation and editability metrics already wired into frontier review.

Planned next:

- control plane and tenant-scoped job model;
- browser graph inspector and review workflow;
- additional exports such as PSD or partner-tool manifests;
- enterprise deployment packaging once repeated demand exists.
