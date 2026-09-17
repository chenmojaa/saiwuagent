"""URL fetcher with SSRF guard."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import trafilatura

_log = logging.getLogger(__name__)


# Hosts that are NEVER allowed regardless of allow_private: link-local (cloud
# metadata), multicast, reserved, and the well-known AWS / GCP metadata IPs.
_ALWAYS_BLOCKED_NAMES = {"localhost", "metadata.google.internal"}

# RFC 2544 网络设备基准测试保留段。代理软件（Clash / Surge / sing-box 等）的
# fake-IP 模式默认就占用 198.18.0.0/16：本机 DNS 会把**所有**域名解析到这一段，
# 再由代理按域名转发到真实服务器。
#
# 问题是 Python 的 ipaddress 把 198.18.0.0/15 归进了 is_private，于是每个公网
# 域名在这里都会被判成「内网地址」而拒绝。实测表现（2026-09-17）：Bing 搜索页
# 能抓到（httpx 走系统代理，不过这道校验），但**每一条结果的正文都抓不到**，
# web_search 只能降级用搜索摘要。
#
# 该网段是 IANA 保留段，现实中不会有内网服务监听它，所以按公网放行是安全的；
# 其余内网段（10/8、172.16/12、192.168/16、127/8）的拦截完全不变。
_FAKE_IP_NETWORKS = (ipaddress.ip_network("198.18.0.0/15"),)


def _is_fake_ip(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    return any(ip in network for network in _FAKE_IP_NETWORKS)


def _ip_to_block(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address", allow_private: bool) -> bool:
    # Always blocked categories — these are SSRF amplifiers regardless of policy.
    if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    # 代理 fake-IP：不是内网，是代理占位地址，见 _FAKE_IP_NETWORKS 的说明。
    if _is_fake_ip(ip):
        return False
    if ip.is_loopback or ip.is_private:
        return not allow_private
    return False


def _resolve_and_check(host: str, allow_private: bool) -> bool:
    """Return True if the host is blocked under the given policy."""
    if not host:
        return True
    if host.lower() in _ALWAYS_BLOCKED_NAMES:
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        # Refuse to connect when DNS fails open — fail closed.
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if _ip_to_block(ip, allow_private):
            return True
    return False


def _check_url(url: str) -> None:
    """Backward-compatible internal alias (allow_private=False)."""
    check_url(url, allow_private=False)


def check_url(url: str, allow_private: bool = False) -> None:
    """Raise ValueError if the URL targets a blocked host.

    ``allow_private=False`` (default) rejects loopback + RFC1918 in addition
    to the always-blocked link-local / multicast / reserved ranges. Set
    ``allow_private=True`` only for code paths the operator has explicitly
    opted in (e.g. local Ollama at 127.0.0.1). Link-local (169.254/16) and
    cloud-metadata IPs are ALWAYS rejected — this guard cannot be turned off
    from the request body, only by editing this function.
    """
    scheme = ""
    host = ""
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        host = parsed.hostname or ""
    except Exception:
        host = ""
    if scheme not in ("http", "https"):
        raise ValueError(f"URL scheme must be http or https: {scheme or '<empty>'}")
    if _resolve_and_check(host, allow_private):
        raise ValueError(f"URL host blocked by SSRF policy: {host or '<empty>'}")


def fetch_url(url: str) -> dict:
  """Fetch URL and extract main text. Returns {title, content, word_count}."""
  _check_url(url)
  downloaded = trafilatura.fetch_url(url)
  if not downloaded:
    raise ValueError(f"无法抓取 URL: {url}")

  text = trafilatura.extract(
    downloaded,
    include_comments=False,
    include_tables=False,
    no_fallback=False,
  )
  if not text:
    raise ValueError(f"未提取到正文: {url}")

  meta = trafilatura.extract_metadata(downloaded)
  title = (meta.title if meta and meta.title else url)[:200]

  return {
    "title": title,
    "content": text,
    "word_count": len(text),
  }
