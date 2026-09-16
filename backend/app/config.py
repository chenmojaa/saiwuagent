from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _hd(name: str, default):
  """声明一个同时接受 `HD_<name>` 与 `<name>` 两种写法的环境变量字段。

  背景：本文件的注释、README、docs 以及 `.env.example` 一直把开关写成
  `HD_PLANNER_ENABLED` / `HD_USE_GRAPH` 这种带前缀形式，但 Settings 没有配
  env_prefix，pydantic-settings 默认只认「字段名大写」即 `PLANNER_ENABLED`。
  结果是照文档填的 `HD_*` 变量被**静默忽略**（实测 `HD_PLANNER_ENABLED=false`
  无效、`PLANNER_ENABLED=false` 有效）。

  用 AliasChoices 把两种名字都挂上：带前缀的优先（文档写法），无前缀的兜底
  （现有 .env 与 pydantic 原生写法），因此对既有配置完全向后兼容。
  """
  return Field(default=default,
               validation_alias=AliasChoices("HD_" + name, name))


class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
  )

  # ---- LLM ----
  llm_provider: str = "openai"
  llm_model: str = "gpt-4o-mini"
  llm_api_key: str = ""
  llm_api_base: str = ""

  # ---- Embedding ----
  embedding_provider: str = "openai"
  embedding_model: str = "text-embedding-3-small"
  embedding_api_key: str = ""
  embedding_api_base: str = "https://api.minimax.chat/v1"
  embedding_device: str = "cpu"

  # ---- Server ----
  host: str = "127.0.0.1"
  port: int = 8000
  log_level: str = "info"

  # ---- Storage (P2) ----
  data_dir: str = "./data"
  sqlite_path: str = "./data/notes.db"
  chroma_dir: str = "./data/chroma"
  notes_dir: str = "./data/notes"


  # ---- Agent tools (tool-calling for MCP + skills) ----
  tools_enabled: bool = _hd("TOOLS_ENABLED", True)          # master switch for tool-calling
  tools_max_steps: int = _hd("TOOLS_MAX_STEPS", 6)           # cap on tool-call iterations
  mcp_call_timeout_sec: float = _hd("MCP_CALL_TIMEOUT", 30.0)   # per-tool call budget
  mcp_init_timeout_sec: float = _hd("MCP_INIT_TIMEOUT", 10.0)   # per-server startup budget
  llm_max_retries: int = _hd("LLM_MAX_RETRIES", 3)          # transient failures retry budget (0 = no retry)
  # ---- Feishu (Lark) sync ----
  feishu_enabled: bool = False
  feishu_app_id: str = ""
  feishu_app_secret: str = ""
  feishu_api_base: str = "https://open.feishu.cn"
  feishu_space_ids: str = ""          # comma-separated list; empty = all visible spaces
  feishu_sync_interval_min: int = 15   # minutes between sync runs
  feishu_page_size: int = 50
  feishu_web_url: str = ""             # for constructing view URLs e.g. https://{tenant}.feishu.cn
  custom_models_allow_private: bool = _hd("CUSTOM_MODELS_ALLOW_PRIVATE", False)  # allow RFC1918 / loopback in /settings/custom-models (for local Ollama)

  # ---- Agent / Router ----
  use_graph: bool = _hd("USE_GRAPH", True)              # false = legacy direct-call path
  router_enabled: bool = _hd("ROUTER_ENABLED", True)    # false = always intent=chat
  router_model: str = _hd("ROUTER_MODEL", "")           # empty = use main LLM model
  router_base_url: str = _hd("ROUTER_BASE_URL", "")     # empty = use main base_url
  research_max_iter: int = _hd("RESEARCH_MAX_ITER", 3)
  research_target_chunks: int = _hd("RESEARCH_TARGET_CHUNKS", 8)
  planner_enabled: bool = _hd("PLANNER_ENABLED", True)          # task planning before research
  planner_max_steps: int = _hd("PLANNER_MAX_STEPS", 4)          # max sub-queries per plan
  parallel_plan_enabled: bool = _hd("PARALLEL_PLAN_ENABLED", True)   # run plan steps concurrently via ThreadPoolExecutor
  parallel_plan_max_workers: int = _hd("PARALLEL_PLAN_MAX_WORKERS", 4)  # thread pool size (capped at step count)
  memory_extraction_enabled: bool = _hd("MEMORY_EXTRACTION_ENABLED", True)  # auto fact extraction after each turn
  memory_max_facts: int = _hd("MEMORY_MAX_FACTS", 8)            # max recalled facts injected per turn
  ingest_provider: str = _hd("INGEST_PROVIDER", "")            # empty = use main LLM for metadata extraction

  # ---- 检索质量阈值 ----
  # 融合分（0.7*向量 + 0.3*关键词）低于此值的切片不进入引用。
  # 0.18 太宽松：实测「推荐合肥蜀山区野钓的地点」会召回一堆只是词面
  # 沾了"安徽/合肥"的文旅切片（0.24~0.41），被当成"来源"展示给用户。
  # 实测相关查询 min≈0.42、不相关 max≈0.41，0.45 能干净分开。
  retrieval_min_score: float = _hd("RETRIEVAL_MIN_SCORE", 0.45)
  # 单维度下限（max(向量分, 关键词分)）：至少要有一条路径给出信号。
  retrieval_min_dim_score: float = _hd("RETRIEVAL_MIN_DIM_SCORE", 0.18)

  # ---- Ingestion ----
  chunk_size: int = 500
  chunk_overlap: int = 80


@lru_cache
def get_settings() -> Settings:
  return Settings()


settings = get_settings()
