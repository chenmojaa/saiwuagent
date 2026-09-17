# -*- coding: utf-8 -*-
"""联网结果质量判据（公共模块）。

单独抽出来是因为两边都要用：
  * ``agent/nodes/verify.py`` —— 判断「能不能拿这批网页去裁决」
  * ``tools/web_search.py``  —— 判断「要不要升级到浏览器搜索」

放这里而不是 verify.py，是为了避免 web_search -> verify -> web_search 的循环导入。
本模块**不依赖任何项目内模块**，是纯函数集合。
"""
from __future__ import annotations

import re

# 按标点/空格切段。中文没有空格，靠这些分隔符先粗切。
_TERM_SPLIT_RE = re.compile(r"[\s,，、;；:：/|\\()（）\[\]【】\"'“”]+")
_ASCII_WORD_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_CJK_ONLY_RE = re.compile(r"[^\u4e00-\u9fff]")

# 相关性阈值。实测标定（2026-09-17，见 tools/browser_search.py 的说明）：
#   Bing HTML 抓取跑偏时命中率 0.00~0.17；真实相关时 ≥ 0.33。
# 取 0.25 落在中间。偏宽松是刻意的 —— 挡住有效结果 = 功能直接失效，
# 放过边缘材料只是让上层多一道判断，代价小得多。
DEFAULT_MIN_RATIO = 0.25


def query_terms(query: str) -> list[str]:
  """从查询里抽出用于相关性判断的词。

  中文用「按标点/空格切段 + 段内取 2-gram」的粗粒度方案。不需要精确分词 ——
  目标只是判断「网页到底有没有在讲这个问题」，2-gram 已经够：真讲这个问题的
  页面必然会命中大部分 2-gram。
  """
  terms: list[str] = []
  for seg in _TERM_SPLIT_RE.split(query or ""):
    seg = seg.strip()
    if not seg:
      continue
    if _ASCII_WORD_RE.match(seg):
      if len(seg) >= 2:
        terms.append(seg.lower())
      continue
    cjk = _CJK_ONLY_RE.sub("", seg)
    if len(cjk) >= 2:
      terms.extend(cjk[i:i + 2] for i in range(len(cjk) - 1))
    elif cjk:
      terms.append(cjk)
    # 数字（年份、金额）往往是判定关键，单独保留
    terms.extend(re.findall(r"\d{2,}", seg))
  return list(dict.fromkeys(terms))


def relevance_ratio(query: str, chunks: list[dict],
                    text_chars: int = 400) -> float | None:
  """查询词在网页标题+正文里的命中比例。抽不出词时返回 None。"""
  terms = query_terms(query)
  if not terms:
    return None
  hay = " ".join(
    "%s %s" % (c.get("title") or "", str(c.get("text") or "")[:text_chars])
    for c in (chunks or [])
  ).lower()
  if not hay.strip():
    return 0.0
  return sum(1 for t in terms if t in hay) / len(terms)


def is_relevant(query: str, chunks: list[dict],
                min_ratio: float = DEFAULT_MIN_RATIO) -> bool:
  """这批网页是否真的在讲这个问题。

  抽不出词（纯符号查询）时返回 True —— 判断不了就不拦，避免误伤。
  """
  ratio = relevance_ratio(query, chunks)
  if ratio is None:
    return True
  return ratio >= min_ratio
