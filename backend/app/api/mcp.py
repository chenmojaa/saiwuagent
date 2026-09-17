'''MCP server configuration, presets, and lightweight connection checks.'''
from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

router = APIRouter(prefix='/mcp', tags=['mcp'])
_REGISTRY_LOCK = threading.RLock()
_MCP_DIR = Path(settings.data_dir) / 'mcp'
_MAX_SERVERS = 50
_MAX_ARGS = 100
_MAX_ENV = 100

MCP_PRESETS = [
  {
    'id': 'filesystem',
    'name': 'Filesystem',
    'name_zh': '本地文件',
    'description': '按目录授权读取和写入文件，适合让 Agent 处理项目文件。',
    'emoji': '📁',
    'category': '本地工具',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-filesystem', '.'],
    'env': {},
    'requirements': 'Node.js / npx',
  },
  {
    'id': 'fetch',
    'name': 'Fetch',
    'name_zh': '网页抓取',
    'description': '抓取网页内容并转换为模型可读文本，适合补充知识库。',
    'emoji': '🌐',
    'category': '联网工具',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-fetch'],
    'env': {},
    'requirements': 'Node.js / npx',
  },
  {
    # 用 API key 接入的模型本身没有联网能力（训练数据有截止时间），
    # 这个预设把「联网搜索」交给模型自己驱动：模型通过 mcp_invoke 打开真实
    # 浏览器去搜索引擎，再读页面正文，因此能拿到实时信息。
    # 与内置的 web_search 兜底的区别：兜底是 research 图里代码自动跑、模型
    # 全程无感知；这里是模型可主动、多轮调用的工具。
    # 默认有头模式（能看见浏览器窗口）；想跑无头加 '--headless' 即可。
    'id': 'playwright',
    'name': 'Playwright',
    'name_zh': '浏览器搜索',
    'description': '让 Agent 打开真实浏览器搜索并阅读网页，适合需要实时信息的场景。首次运行会下载 Chromium。',
    'emoji': '🔎',
    'category': '联网工具',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@playwright/mcp@latest'],
    'env': {},
    'requirements': 'Node.js 18+ / npx / 首次运行需下载 Chromium',
  },
  {
    'id': 'memory',
    'name': 'Memory',
    'name_zh': '持久记忆',
    'description': '保存跨会话的结构化记忆，让 Agent 记住用户偏好和关键事实。',
    'emoji': '🧠',
    'category': 'Agent 能力',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-memory'],
    'env': {},
    'requirements': 'Node.js / npx',
  },
  {
    'id': 'github',
    'name': 'GitHub',
    'name_zh': 'GitHub',
    'description': '让 Agent 查询仓库、Issue 和代码内容，需要 GitHub Token。',
    'emoji': '🐙',
    'category': '开发工具',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-github'],
    'env': {'GITHUB_PERSONAL_ACCESS_TOKEN': ''},
    'requirements': 'GitHub Token / Node.js / npx',
  },
  {
    'id': 'postgres',
    'name': 'Postgres',
    'name_zh': 'PostgreSQL',
    'description': '让 Agent 查询 PostgreSQL 数据库，建议只授予只读账号。',
    'emoji': '🐘',
    'category': '数据工具',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-postgres', 'postgresql://localhost/mydb'],
    'env': {},
    'requirements': 'Node.js / npx / PostgreSQL',
  },
  {
    'id': 'sequential-thinking',
    'name': 'Sequential Thinking',
    'name_zh': '顺序思考',
    'description': '为复杂任务增加结构化思考步骤，适合研究和多步分析。',
    'emoji': '🧩',
    'category': 'Agent 能力',
    'transport': 'stdio',
    'command': 'npx',
    'args': ['-y', '@modelcontextprotocol/server-sequential-thinking'],
    'env': {},
    'requirements': 'Node.js / npx',
  },
]


class MCPServerInput(BaseModel):
  name: str = Field(min_length=1, max_length=80)
  transport: Literal['stdio', 'http'] = 'stdio'
  command: str = ''
  args: list[str] = Field(default_factory=list)
  env: dict[str, str] = Field(default_factory=dict)
  url: str = ''
  description: str = Field(default='', max_length=300)
  enabled: bool = True


class MCPServerPatch(BaseModel):
  name: str | None = Field(default=None, min_length=1, max_length=80)
  transport: Literal['stdio', 'http'] | None = None
  command: str | None = None
  args: list[str] | None = None
  env: dict[str, str] | None = None
  url: str | None = None
  description: str | None = None
  enabled: bool | None = None


def _now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _registry_path() -> Path:
  return _MCP_DIR / 'servers.json'


def _read_servers() -> list[dict]:
  path = _registry_path()
  if not path.exists():
    return []
  try:
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
  except (OSError, json.JSONDecodeError):
    return []
  items = payload.get('servers', []) if isinstance(payload, dict) else []
  return [item for item in items if isinstance(item, dict) and isinstance(item.get('id'), str)]


def _write_servers(items: list[dict]) -> None:
  _MCP_DIR.mkdir(parents=True, exist_ok=True)
  path = _registry_path()
  temporary = path.with_suffix('.json.tmp')
  temporary.write_text(json.dumps({'version': 1, 'servers': items}, ensure_ascii=False, indent=2), encoding='utf-8')
  temporary.replace(path)


def _slug(value: str) -> str:
  cleaned = re.sub(r'[^a-zA-Z0-9_-]+', '-', value.strip()).strip('-_').lower()
  return cleaned or 'mcp'


def _validate_values(values: dict) -> dict:
  values = dict(values)
  name = str(values.get('name', '')).strip()
  if not name:
    raise ValueError('MCP 名称不能为空')
  if len(name) > 80:
    raise ValueError('MCP 名称不能超过 80 个字符')
  transport = values.get('transport', 'stdio')
  if transport not in ('stdio', 'http'):
    raise ValueError('transport 必须是 stdio 或 http')
  command = str(values.get('command', '')).strip()
  args = values.get('args') or []
  env = values.get('env') or {}
  url = str(values.get('url', '')).strip()
  if not isinstance(args, list) or len(args) > _MAX_ARGS:
    raise ValueError('启动参数格式不正确或数量过多')
  if any(not isinstance(argument, str) or len(argument) > 4000 for argument in args):
    raise ValueError('启动参数格式不正确')
  if not isinstance(env, dict) or len(env) > _MAX_ENV:
    raise ValueError('环境变量格式不正确或数量过多')
  for key, value in env.items():
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key) or not isinstance(value, str):
      raise ValueError('环境变量名称或值格式不正确')
  if transport == 'stdio' and not command:
    raise ValueError('stdio 模式需要填写启动命令')
  if transport == 'http':
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
      raise ValueError('http 模式需要填写有效的 HTTP(S) 地址')
  values['name'] = name
  values['command'] = command
  values['args'] = [str(argument) for argument in args]
  values['env'] = {str(key): str(value) for key, value in env.items()}
  values['url'] = url
  values['description'] = str(values.get('description', '')).strip()[:300]
  values['enabled'] = bool(values.get('enabled', True))
  return values


def _new_id(name: str) -> str:
  return f'{_slug(name)}-{uuid.uuid4().hex[:8]}'


def _find_server(server_id: str) -> dict | None:
  return next((item for item in _read_servers() if item.get('id') == server_id), None)


def _replace_server(server: dict) -> None:
  with _REGISTRY_LOCK:
    items = _read_servers()
    index = next((index for index, item in enumerate(items) if item.get('id') == server['id']), None)
    if index is None:
      items.append(server)
    else:
      items[index] = server
    _write_servers(items)


@router.get('/presets')
def list_presets():
  installed = {item.get('preset_id') for item in _read_servers()}
  return {'items': [{**preset, 'installed': preset['id'] in installed} for preset in MCP_PRESETS]}


@router.get('')
def list_servers():
  with _REGISTRY_LOCK:
    return {'items': _read_servers()}


@router.post('/presets/{preset_id}')
def install_preset(preset_id: str):
  preset = next((item for item in MCP_PRESETS if item['id'] == preset_id), None)
  if preset is None:
    raise HTTPException(status_code=404, detail='MCP 预设不存在')
  with _REGISTRY_LOCK:
    existing = next((item for item in _read_servers() if item.get('preset_id') == preset_id), None)
    if existing is not None:
      return {'item': existing, 'created': False}
    if len(_read_servers()) >= _MAX_SERVERS:
      raise HTTPException(status_code=400, detail='MCP 数量已达上限')
    values = _validate_values({
      'name': preset['name'],
      'transport': preset['transport'],
      'command': preset['command'],
      'args': preset['args'],
      'env': preset['env'],
      'url': '',
      'description': preset['description'],
      'enabled': False,
    })
    server = {
      'id': _new_id(values['name']),
      'preset_id': preset['id'],
      **values,
      'created_at': _now(),
      'updated_at': _now(),
      'last_test_at': None,
      'last_test_ok': None,
      'last_test_message': '尚未测试',
    }
    _replace_server(server)
    return {'item': server, 'created': True}


@router.post('')
def create_server(body: MCPServerInput):
  with _REGISTRY_LOCK:
    if len(_read_servers()) >= _MAX_SERVERS:
      raise HTTPException(status_code=400, detail='MCP 数量已达上限')
    values = _validate_values(body.model_dump())
    server = {
      'id': _new_id(values['name']),
      'preset_id': '',
      **values,
      'created_at': _now(),
      'updated_at': _now(),
      'last_test_at': None,
      'last_test_ok': None,
      'last_test_message': '尚未测试',
    }
    _write_servers([*_read_servers(), server])
    return server


@router.patch('/{server_id}')
def update_server(server_id: str, body: MCPServerPatch):
  with _REGISTRY_LOCK:
    existing = _find_server(server_id)
    if existing is None:
      raise HTTPException(status_code=404, detail='MCP 不存在')
    values = _validate_values({**existing, **body.model_dump(exclude_unset=True)})
    updated = {**existing, **values, 'id': server_id, 'updated_at': _now()}
    _replace_server(updated)
    return updated


@router.post('/{server_id}/test')
def test_server(server_id: str):
  with _REGISTRY_LOCK:
    existing = _find_server(server_id)
    if existing is None:
      raise HTTPException(status_code=404, detail='MCP 不存在')
    transport = existing.get('transport', 'stdio')
    if transport == 'stdio':
      command = str(existing.get('command', '')).strip()
      resolved = shutil.which(command)
      if not resolved and Path(command).is_file():
        resolved = str(Path(command).resolve())
      ok = bool(resolved)
      message = f'启动器可用：{resolved}' if ok else f'未找到启动命令：{command}'
    else:
      parsed = urlparse(str(existing.get('url', '')))
      ok = parsed.scheme in ('http', 'https') and bool(parsed.netloc)
      message = 'HTTP(S) 地址格式有效（仅做配置检查）' if ok else 'HTTP(S) 地址无效'
    tested = {
      **existing,
      'last_test_at': _now(),
      'last_test_ok': ok,
      'last_test_message': message,
      'updated_at': _now(),
    }
    _replace_server(tested)
    return {'ok': ok, 'message': message, 'item': tested}


@router.delete('/{server_id}')
def delete_server(server_id: str):
  with _REGISTRY_LOCK:
    items = _read_servers()
    remaining = [item for item in items if item.get('id') != server_id]
    if len(remaining) == len(items):
      raise HTTPException(status_code=404, detail='MCP 不存在')
    _write_servers(remaining)
  return {'ok': True}

@router.get('/calls')
def list_calls(limit: int = 100, server_id: str | None = None, session_id: str | None = None, status: str | None = None):
  from app.storage.db import list_mcp_calls
  rows = list_mcp_calls(limit=limit, server_id=server_id, session_id=session_id, status=status)
  return {'calls': rows, 'count': len(rows)}

@router.delete('/calls')
def clear_calls(server_id: str | None = None):
  from app.storage.db import clear_mcp_calls
  n = clear_mcp_calls(server_id=server_id)
  return {'ok': True, 'cleared': n}

@router.get('/calls/stats')
def call_stats(server_id: str | None = None):
  from app.storage.db import list_mcp_calls
  rows = list_mcp_calls(limit=500, server_id=server_id)
  total = len(rows)
  by_status = {}
  by_server = {}
  latencies = []
  for r in rows:
    by_status[r['status']] = by_status.get(r['status'], 0) + 1
    by_server[r['server_id']] = by_server.get(r['server_id'], 0) + 1
    if r['latency_ms'] > 0: latencies.append(r['latency_ms'])
  avg = int(sum(latencies) / len(latencies)) if latencies else 0
  p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
  return {'total': total, 'by_status': by_status, 'by_server': by_server, 'avg_latency_ms': avg, 'p95_latency_ms': p95}

