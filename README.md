# Radiology RAG System 🏥

Advanced Retrieval-Augmented Generation system for radiology, combining 99K+ peer-reviewed abstracts with 46K+ RadLex terminology entries. Powered by Modal.com serverless A100 GPU.

## Features

- 📚 **99,000+ literature abstracts** from major radiology journals
- 🔬 **46,000+ RadLex terms** for standardized medical definitions
- 🌩️ **Modal A100 GPU** for fast LLM inference (~90s per query)
- 🎯 **Inline citations** with clickable DOI links
- 💰 **Cost-efficient**: ~$0.006 per query
- 🤖 **Multi-agent workflow** with query analysis and reranking

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Chainlit** - Modern chat interface  
- **Modal** - Serverless GPU inference  
- **Sentence Transformers** - Local embeddings  
- **Qdrant** - Vector database

### 2. Setup Modal.com (for serverless GPU inference)

Modal.com provides serverless A100 GPU access for LLM inference. **No GPU hardware required locally!**

**Step 2.1: Install Modal**
```bash
pip install modal
```

**Step 2.2: Create Modal Account**
1. Visit https://modal.com
2. Sign up for free account
3. You get **$30 free credits** (enough for ~5,000 queries!)

**Step 2.3: Authenticate**
```bash
modal token new
```
This will open your browser to authenticate. Copy the token back to terminal.

**Step 2.4: Deploy GPU Function**
```bash
modal deploy src/modal_app.py
```

This deploys your LLM to Modal's A100 GPU infrastructure. You'll see:
```
✓ Created objects.
├── 🔨 Created mount /Users/eri/llm/src/modal_app.py
├── 🔨 Created radiology-rag::generate_answer
└── 🔨 Created App radiology-rag
✓ App deployed! 🎉
```

**Verify deployment:**
```bash
modal app list  # Should show 'radiology-rag'
```

### 3. Index Data (One-time)

**Fast indexing (Recommended for Apple Silicon):**
```bash
python scripts/index_data_fast.py  # 5-10 minutes with lightweight model
```

**Full quality indexing:**
```bash
python scripts/index_data.py  # 30-45 minutes with BGE-M3 model
```

**Important:** If you use fast indexing, update `src/config.py`:
```python
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLL_ABSTRACTS = "radiology_abstracts_fast"
COLL_RADLEX = "radlex_terms_fast"
```

### 4. Launch Web Interface

```bash
chainlit run app.py
```

**What happens:**
- Opens at `http://localhost:8000`  
- Chat interface loads with welcome message  
- Shows step-by-step RAG pipeline  
- Real-time progress indicators for Modal GPU  

**Test Chainlit installation:**
```bash
chainlit run chainlit_hello.py  # Quick test
```

**Alternative (Gradio - deprecated):**
```bash
python scripts/run_app_modal.py  # Opens at http://localhost:7860
```

---

## User Interface

The system uses [**Chainlit**](https://github.com/Chainlit/chainlit) for the web interface - a modern framework built specifically for conversational AI applications.

### Why Chainlit > Gradio?

| Feature | Chainlit | Gradio |
|---------|----------|--------|
| **Chat Interface** | ✅ Native, polished | ⚠️ Generic blocks |
| **Message History** | ✅ Per-session | ❌ No memory |
| **Step Visualization** | ✅ Built-in | ❌ Manual |
| **Streaming** | ✅ Real-time | ⚠️ Limited |
| **Citations** | ✅ Rich formatting | ⚠️ Plain text |
| **UX** | ✅ Chat-first | ⚠️ Form-like |
| **Production Ready** | ✅ Yes | ⚠️ Prototyping |
| **Customization** | ✅ Themes, CSS | ⚠️ Limited |

### Features in our app:

- 💬 **Real-time chat** with message history
- 🔍 **Step tracking** shows each stage of retrieval
- 📊 **Progress indicators** for Modal GPU calls
- 🎨 **Professional UI** with markdown support
- 🔗 **Clickable citations** with inline DOI links

---

## How It Works: Modal.com Architecture

This system uses a **hybrid architecture** to minimize costs and maximize performance:

### Local Processing (Your Computer)
- ✅ **Embeddings**: Generate query embeddings with sentence-transformers
- ✅ **Vector Search**: Search Qdrant database locally
- ✅ **Reranking**: Cross-encoder scoring on CPU
- 💰 **Cost**: Free (runs on your hardware)

### Remote Processing (Modal.com Serverless GPU)
- ✅ **LLM Inference**: Qwen 2.5-7B runs on A100 GPU
- ✅ **Auto-scaling**: GPU spins up only when needed
- ✅ **No idle costs**: Pay only for active inference
- 💰 **Cost**: ~$0.006 per query (~$0.60 for 100 queries)

### Why This Architecture?

**Embedding models** (384-1024 dims) are small and fast on CPU:
- all-MiniLM-L6-v2: ~80MB, runs in <1 second on CPU
- No need for expensive GPU for embedding/search

**LLMs** (7B parameters) are large and slow on CPU:
- Qwen 2.5-7B: ~14GB, would take 5-10 minutes on CPU
- A100 GPU: Same generation in 30-40 seconds

**Result**: Best of both worlds!
- Fast local search
- Fast GPU generation
- Minimal cloud costs

---

## Modal.com Pricing & Credits

**Free Credits:**
- New accounts: **$30 free credits**
- Enough for ~5,000 queries!

**Pricing:**
- A100 GPU: ~$0.00059/second
- Average query: ~40 seconds = **$0.006/query**
- Container idle (2 min): Included in query cost

**Monthly Cost Examples:**
```
 50 queries/month:  $0.30
100 queries/month:  $0.60
500 queries/month:  $3.00
```

**Cost Tracking:**
```bash
# Check remaining credits
modal profile current
```

**When Credits Run Out:**
- Add payment method for continued use
- Or switch to local CPU (very slow but free)

---

## Project Structure

```
llm/
├── src/
│   ├── config.py              # Configuration settings
│   ├── state.py               # State definitions
│   ├── utils.py               # Utility functions
│   ├── indexer.py             # Data indexing
│   ├── agent_modal.py         # Main RAG agent (Modal GPU)
│   ├── modal_app.py           # A100 GPU inference
│   ├── modal_app_h100.py      # H100 GPU (optional, faster)
│   ├── config_docker.py       # Docker Qdrant config (future)
│   └── indexer_docker.py      # Docker indexer (future)
├── scripts/
│   ├── index_data.py          # Initial index with BGE-M3
│   ├── index_data_fast.py     # Initial index with MiniLM (10x faster)
│   ├── add_papers.py          # Add new papers incrementally
│   ├── index_data_docker.py   # Docker indexing (future)
│   └── run_app_modal.py       # Gradio app (deprecated)
├── app.py                     # Chainlit app (recommended)
├── chainlit.md                # Welcome message for Chainlit
├── chainlit_hello.py          # Test Chainlit installation
├── .chainlit/                 # Chainlit configuration
│   └── config.toml
├── public/                    # Static assets for Chainlit
├── out_pubmed_multi/          # Literature CSVs
├── Radlex.csv                 # RadLex terminology
├── requirements.txt           # Dependencies
├── GPU_OPTIONS.md            # GPU comparison guide
└── QDRANT_DOCKER_SETUP.md    # Docker setup (future)
```

---

## Methods

### Data Sources

**Literature Corpus**
- **Source**: PubMed abstracts from 15 major radiology journals
- **Journals**: Radiology, AJR, European Radiology, Radiographics, etc.
- **Total**: ~99,000 abstracts with title, authors, year, journal, DOI, PMID
- **Format**: Individual CSV files per journal in `out_pubmed_multi/`

**Terminology Database**
- **Source**: RadLex ontology (Radiological Society of North America)
- **Total**: 46,022 standardized radiology terms
- **Fields**: Preferred labels, synonyms, definitions
- **Format**: Single CSV file `Radlex.csv`

### Embedding and Indexing

**Text Embedding**

**Default Model (High Quality):**
- **Model**: BAAI/bge-m3 (1024-dimensional multilingual embeddings)
- **Processing time**: ~30-45 minutes on Apple Silicon MPS
- **Quality**: Highest accuracy for medical terminology

**Fast Alternative (Recommended for Quick Setup):**
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional)
- **Processing time**: ~5-10 minutes on Apple Silicon MPS
- **Quality**: 85-90% of BGE-M3 quality, 10x faster
- **Use**: `python scripts/index_data_fast.py`

**Common Settings:**
- **Framework**: sentence-transformers
- **Chunking**: Abstracts stored as single semantic units (no splitting)
- **Batch processing**: 
  - CPU: 64 documents per batch
  - MPS (Apple Silicon): 16-128 (model dependent)
  - A100 GPU: 256 documents per batch

**Vector Database**
- **System**: Qdrant v1.7+
- **Storage**: Local filesystem (`~/rad_rag/qdrant/`)
- **Collections**: 
  - `radiology_abstracts` (99K vectors)
  - `radlex_terms` (46K vectors)
- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Distance metric**: Cosine similarity

### Retrieval Pipeline

**Stage 1: Dense Retrieval**
- **Method**: Semantic search using embedding model (BGE-M3 or MiniLM)
- **Query encoding**: Same model used for indexing
- **Initial retrieval**: TOP_K = 50 documents per collection
- **Similarity threshold**: 0.40 (cosine similarity)
- **Note**: Query embeddings must use the same model as indexing

**Stage 2: Query Analysis**
- **Modality detection**: Pattern matching for CT, MRI, X-ray, ultrasound, PET
- **Terminology detection**: Keyword analysis for definition requests
  - Keywords: "signs of", "findings", "what is", "define", etc.
- **Conditional retrieval**: RadLex terms included only if explicitly needed

**Stage 3: Cross-Encoder Reranking**
- **Model**: BAAI/bge-reranker-base
- **Input**: Query-document pairs from initial retrieval
- **Scoring**: Semantic relevance scores (0-1 range)
- **Final selection**: TOP_N = 12 highest-scoring documents
- **Purpose**: Improves precision by re-ranking semantic matches

### Answer Generation

**Large Language Model**
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Parameters**: 7 billion
- **Deployment**: Modal.com serverless A100 GPU
- **Inference time**: ~30-40 seconds per query
- **Context window**: 32K tokens
- **Max output**: 1536 tokens

**Prompt Engineering**
- **System role**: Expert radiology assistant
- **Context**: Concatenated abstracts with metadata (title, journal, year, DOI)
- **Instructions**: 
  - Use numbered citations [1], [2], etc.
  - Cite after every claim
  - Do not generate reference section (added programmatically)
- **Temperature**: 0.0 (deterministic outputs)
- **Sampling**: Greedy decoding (do_sample=False)

**Citation Processing**
- **LLM output**: Numbered citations [1], [2], [3]
- **Post-processing**: Replace with inline format `[[Author Year](DOI_URL)]`
- **Link generation**: `https://doi.org/{doi}` for each citation
- **Format**: Markdown-compatible clickable links
- **Cleanup**: Remove invalid citations, preserve paragraph spacing

### Multi-Agent Workflow

Implemented as a state machine with 5 sequential nodes:

1. **Query Analysis Node**
   - Detect imaging modality (CT, MRI, etc.)
   - Determine if terminology needed
   - Output: `modality`, `include_terminology` flags

2. **Abstract Retrieval Node**
   - Search literature collection
   - Return TOP_K=50 candidates
   - Output: `retrieved_abstracts`

3. **RadLex Retrieval Node** (conditional)
   - Execute only if `include_terminology=True`
   - Search terminology collection
   - Return TOP_K=5 terms
   - Output: `retrieved_radlex`

4. **Reranking Node**
   - Score all retrieved abstracts with cross-encoder
   - Select TOP_N=12 best matches
   - Output: `reranked_docs`

5. **Generation Node**
   - Build context from top documents
   - Call Modal GPU endpoint
   - Process citations
   - Output: `answer`, `confidence`, `citations`

**State Management**
- Framework: TypedDict with required fields
- Persistence: In-memory during query execution
- No conversation history (stateless queries)

### Infrastructure

**Local Processing**
- Embeddings: CPU or local GPU
- Vector search: Qdrant on localhost
- Reranking: CPU (lightweight model)

**Remote Processing (Modal.com)**
- **Platform**: Modal.com serverless GPU infrastructure
- **GPU**: A100 (40GB VRAM)
- **Model**: Qwen/Qwen2.5-7B-Instruct
- **Cold start**: ~70 seconds (first query after idle)
- **Warm container**: Kept alive 2 minutes between queries
- **Auto-scaling**: Spins up on demand, shuts down when idle
- **Deployment**: One-time `modal deploy src/modal_app.py`
- **Function lookup**: Agent connects via `modal.Function.from_name()`

**Performance Metrics**
- Average query time: 90 seconds
- Embedding time: ~1 second
- Retrieval time: ~2 seconds
- Reranking time: ~3 seconds
- LLM inference: ~30-40 seconds
- Post-processing: <1 second

**Cost Analysis**
- A100 runtime: ~40 seconds @ $0.00059/sec
- Container overhead: ~2 minutes idle @ $0.00059/sec
- Average cost: $0.006 per query
- Monthly (100 queries): $0.60

### Alternative Configurations

**GPU Options** (see `GPU_OPTIONS.md`)
- H100: 2.5x faster (~35s), 3x more expensive ($0.018/query)
- A100: Current default, good balance
- L40S: Similar speed, comparable cost

**Docker Deployment** (see `QDRANT_DOCKER_SETUP.md`)
- Future: Qdrant in Docker for production
- Files prepared: `config_docker.py`, `indexer_docker.py`

---

## Example Queries

**Imaging Findings**
```
CT signs of mesenteric ischemia in acute abdomen?
```

**Modality Comparison**
```
MRI vs CT for acute stroke diagnosis?
```

**Classification Systems**
```
Explain the BI-RADS classification system
```

**Diagnostic Criteria**
```
What are the ultrasound criteria for appendicitis?
```

---

## Configuration

### Local Settings (`src/config.py`)

```python
# === Model Selection ===
EMBED_MODEL = "BAAI/bge-m3"  # or "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Runs on Modal GPU

# === Retrieval Parameters ===
TOP_K = 50          # Initial retrieval count
TOP_N = 12          # Final documents after reranking
SIM_THRESH = 0.40   # Similarity threshold

# === Generation ===
MAX_NEW_TOK = 1536  # Max output tokens

# === Batch Sizes ===
ENCODING_BATCH_SIZE = 64  # Auto-detected by device

# === Database ===
QDRANT_PATH = "~/rad_rag/qdrant"
COLL_ABSTRACTS = "radiology_abstracts"
COLL_RADLEX = "radlex_terms"
```

### Modal GPU Settings (`src/modal_app.py`)

```python
@app.function(
    gpu="A100",                    # GPU type (A100, H100, L40S, T4)
    timeout=300,                   # Max function runtime
    container_idle_timeout=120,    # Keep warm for 2 minutes
)
```

**To use faster H100 GPU:**
```bash
# Deploy H100 version (2.5x faster, 3x more expensive)
modal deploy src/modal_app_h100.py

# Update src/agent_modal.py line 27:
_generate_answer_fn = modal.Function.from_name("radiology-rag-h100", "generate_answer")
```

See `GPU_OPTIONS.md` for detailed GPU comparison.

---

## Hardware Requirements

### Indexing (One-time)

**Fast indexing (all-MiniLM-L6-v2):**
- **Apple Silicon**: 5-10 minutes (recommended)
- **CPU**: 10-15 minutes
- **A100 GPU**: 2-3 minutes

**Full quality (BGE-M3):**
- **Apple Silicon MPS**: 30-45 minutes with batch size 16
- **CPU**: 90+ hours (not recommended!)
- **A100 GPU**: 10-15 minutes

### Inference (Per query)
- **Local**: CPU for embeddings/reranking
- **Remote**: Modal A100 GPU for LLM (serverless)

---

## Troubleshooting

**"Collections not found"**
```bash
python scripts/index_data_fast.py  # Run indexing first
```

**Modal errors**

*Authentication failed:*
```bash
modal token new  # Re-authenticate with Modal
```

*"App radiology-rag not found":*
```bash
modal deploy src/modal_app.py  # Deploy the function
modal app list                  # Verify deployment
```

*"No modal credits remaining":*
- Add payment method at https://modal.com/settings
- Or switch to local CPU inference (see below)

**Running without Modal (Local CPU only)**

If you want to avoid Modal entirely (very slow but free):
```bash
# This option doesn't exist yet - Modal is currently required
# Future: Local inference option coming soon
```
Currently, **Modal is required** for LLM inference. Local CPU would take 5-10 minutes per query.

**Out of memory during indexing**
```bash
# Use fast indexing with smaller model
python scripts/index_data_fast.py
```

**Indexing takes too long (Apple Silicon)**
```bash
# Use MPS mode with fast model (~5-10 minutes)
python scripts/index_data_fast.py

# Or use overnight CPU indexing with BGE-M3
python scripts/index_data.py  # Choose CPU mode
```

**Slow queries**
- Check Modal deployment: `modal app list`
- Verify GPU is active: Check Modal dashboard
- Consider H100 for faster inference

---

## Citation

If you use this system in your research:

```bibtex
@software{radiology_rag_2025,
  title={Radiology RAG System},
  author={Your Name},
  year={2025},
  note={Powered by Qwen2.5-7B, BGE-M3, Modal.com}
}
```

**Cite underlying models:**
- Qwen2.5: https://github.com/QwenLM/Qwen2.5
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- RadLex: https://www.rsna.org/en/practice-tools/data-tools-and-standards/radlex-radiology-lexicon

---

## License

Educational and research use. Ensure compliance with:
- Modal.com terms of service
- Model licenses (Qwen, BGE)
- Journal data usage policies
- RadLex terms of use

---

## Performance Tips

**For Apple Silicon Users:**
1. Use `index_data_fast.py` for quick setup (~5-10 min)
2. MPS works great with smaller embedding models
3. Avoid CPU-only mode (extremely slow - 90+ hours)

**For Production:**
1. Use BGE-M3 for highest quality retrieval
2. Index on A100 GPU (Modal, Colab, or cloud instance)
3. Or use overnight CPU indexing on a dedicated machine

**Model Comparison:**
| Model | Indexing Time (Mac) | Quality | Recommended For |
|-------|-------------------|---------|-----------------|
| **all-MiniLM-L6-v2** | 5-10 min | ⭐⭐⭐ | Quick setup, development |
| **BGE-M3** | 30-45 min | ⭐⭐⭐⭐⭐ | Production, best quality |

---

## Advanced Usage

### Adding New Papers (Incremental Indexing)

You can add new papers **without reindexing everything**:

**Step 1: Get new papers**
Download new CSVs with the same format:
- Required columns: `abstract`, `title`, `authors`, `journal`, `year`, `doi`, `pmid`

**Step 2: Add them incrementally**
```bash
# Add a single CSV
python scripts/add_papers.py new_journal.csv

# Add all CSVs from a folder
python scripts/add_papers.py ~/Downloads/new_papers/

# Add to existing folder
python scripts/add_papers.py out_pubmed_multi/
```

**Example output:**
```
📄 INCREMENTAL PAPER ADDITION
Found 2 CSV files:
   • new_journal_2024.csv
   • radiology_updates.csv

Total papers to add: 1,234
Current database size: 99,403 papers

⏱️  Total time: 45.2s
📊 Papers added: 1,234
Database size: 100,637 papers (+1,234)
```

**Benefits:**
- ✅ No full reindex needed (saves 5-45 minutes!)
- ✅ Add papers anytime
- ✅ No app restart needed
- ✅ Immediate availability in searches

**Note:** New papers use the same embedding model as your initial index.

---

### Switch to H100 GPU (Faster)

```bash
# Deploy H100 version
modal deploy src/modal_app_h100.py

# Update agent to use H100
# Edit src/agent_modal.py line 27:
_generate_answer_fn = modal.Function.from_name("radiology-rag-h100", "generate_answer")

# Restart app
python scripts/run_app_modal.py
```

**Trade-off**: 2.5x faster (~35s vs 90s), 3x more expensive ($0.018 vs $0.006)

### Monitor Modal Usage

```bash
# View apps
modal app list

# View function logs
modal app logs radiology-rag

# Check credits
modal profile current
```

### Stop/Restart Modal Functions

```bash
# Functions auto-scale down after idle
# No manual stopping needed!

# To redeploy after changes:
modal deploy src/modal_app.py
```

---

## Data Management

### CSV Format for Papers

To add your own papers, use CSV files with these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `abstract` | ✅ | Paper abstract text | "Background: ... Methods: ..." |
| `title` | ✅ | Paper title | "CT in acute stroke" |
| `authors` | ✅ | Authors (semicolon separated) | "Smith J;Doe A" |
| `journal` | ✅ | Journal name | "Radiology" |
| `year` | ✅ | Publication year | "2024" |
| `doi` | ✅ | DOI identifier | "10.1148/radiol.2024..." |
| `pmid` | Optional | PubMed ID | "38234567" |

### Data Sources

**Current corpus:**
- PubMed abstracts from 15 radiology journals
- See `out_pubmed_multi/` for examples

**To add more sources:**
1. Download abstracts in CSV format
2. Ensure columns match the format above
3. Run: `python scripts/add_papers.py your_file.csv`

**Potential sources:**
- PubMed (via API or export)
- Europe PMC
- ArXiv (medical imaging papers)
- Institutional repositories
- Journal websites

---

## Future Enhancements

- [ ] Multi-turn conversation support
- [ ] Document upload via web interface
- [ ] Docker production deployment
- [ ] Fine-tuned medical LLM
- [ ] Additional literature sources (ArXiv, Europe PMC)
- [ ] Image analysis integration
- [ ] OpenAI embeddings API support
- [ ] Local CPU inference option (no Modal required)
- [ ] Auto-deduplicate papers (by DOI/PMID)

---

## Contact

Questions? Issues? Open a GitHub issue or contact the maintainers.

**Current Stats**
- 📊 99,000+ abstracts indexed
- 🔬 46,000+ RadLex terms
- ⚡ 90-second query time
- 💰 $0.006 per query
