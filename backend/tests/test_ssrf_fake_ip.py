"""Tests for the proxy fake-IP exemption in the SSRF guard.

背景（2026-09-17 实测）：代理软件（Clash / Surge / sing-box 等）的 fake-IP
模式会把**所有**域名解析到 198.18.0.0/15，而 Python 的 ipaddress 把这一段
归入 is_private，于是 fetch_url 拒绝每一个公网域名。表现很隐蔽：
Bing 搜索页能抓到（httpx 走系统代理，不过这道校验），但每条结果的正文都
抓不到，web_search 只能降级用搜索摘要。

这个文件锁定两件事，缺一不可：
  1. fake-IP 段按公网放行；
  2. 真正的内网 / 云元数据地址拦截**完全不受影响** —— 这是安全边界，
     修 bug 不能顺手把它削弱。
"""
from __future__ import annotations

import ipaddress
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _blocked(ip: str, allow_private: bool = False) -> bool:
    from app.tools.fetch_url import _ip_to_block
    return _ip_to_block(ipaddress.ip_address(ip), allow_private)


# ============ fake-IP 段放行 ============
def test_fake_ip_range_allowed():
    from app.tools.fetch_url import _is_fake_ip
    for ip in ("198.18.0.0", "198.18.0.1", "198.18.0.250",
               "198.18.255.255", "198.19.0.1", "198.19.255.254"):
        assert _is_fake_ip(ipaddress.ip_address(ip)), ip
    print("PASS test_fake_ip_range_allowed")


def test_fake_ip_boundaries_excluded():
    """198.18.0.0/15 的上下邻居不能被误判成 fake-IP。"""
    from app.tools.fetch_url import _is_fake_ip
    for ip in ("198.17.255.255", "198.20.0.1", "199.18.0.1"):
        assert not _is_fake_ip(ipaddress.ip_address(ip)), ip
    print("PASS test_fake_ip_boundaries_excluded")


def test_fake_ip_not_blocked():
    assert _blocked("198.18.0.250") is False
    assert _blocked("198.18.0.250", allow_private=True) is False
    assert _blocked("198.19.1.1") is False
    print("PASS test_fake_ip_not_blocked")


# ============ 安全边界未被削弱 ============
def test_loopback_still_blocked():
    assert _blocked("127.0.0.1") is True
    # allow_private 是给本地 Ollama 这类显式 opt-in 用的，行为保持不变。
    assert _blocked("127.0.0.1", allow_private=True) is False
    print("PASS test_loopback_still_blocked")


def test_rfc1918_still_blocked():
    for ip in ("10.0.0.1", "10.255.255.255", "172.16.0.1",
               "172.31.255.255", "192.168.1.1", "192.168.255.254"):
        assert _blocked(ip) is True, ip
    print("PASS test_rfc1918_still_blocked")


def test_cloud_metadata_always_blocked():
    """link-local / 云元数据即使 allow_private=True 也必须拒绝。"""
    for ip in ("169.254.169.254", "169.254.0.1", "0.0.0.0"):
        assert _blocked(ip, allow_private=False) is True, ip
        assert _blocked(ip, allow_private=True) is True, ip
    print("PASS test_cloud_metadata_always_blocked")


def test_public_ip_allowed():
    for ip in ("8.8.8.8", "1.1.1.1", "104.16.0.1"):
        assert _blocked(ip) is False, ip
    print("PASS test_public_ip_allowed")


# ============ 端到端：解析结果决定放行 ============
def _patch_dns(ips):
    """把 socket.getaddrinfo 固定成给定 IP 列表，返回还原函数。"""
    real = socket.getaddrinfo
    socket.getaddrinfo = lambda host, *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips
    ]
    return real


def test_host_resolving_to_fake_ip_is_allowed():
    """模拟代理环境：域名解析到 fake-IP 时必须放行。"""
    import app.tools.fetch_url as fu
    real = _patch_dns(["198.18.0.43"])
    try:
        fu.check_url("https://zh.wikipedia.org/wiki/x")
    finally:
        socket.getaddrinfo = real
    print("PASS test_host_resolving_to_fake_ip_is_allowed")


def test_host_resolving_to_private_is_still_rejected():
    import app.tools.fetch_url as fu
    real = _patch_dns(["192.168.1.10"])
    try:
        try:
            fu.check_url("https://evil.example.com/x")
            raise AssertionError("解析到 RFC1918 地址时必须拒绝")
        except ValueError:
            pass
    finally:
        socket.getaddrinfo = real
    print("PASS test_host_resolving_to_private_is_still_rejected")


def test_mixed_resolution_rejects():
    """fake-IP 与真实内网混合时保守拒绝（不能被 fake-IP 蒙混过关）。"""
    import app.tools.fetch_url as fu
    real = _patch_dns(["198.18.0.10", "192.168.1.10"])
    try:
        try:
            fu.check_url("https://mixed.example.com/x")
            raise AssertionError("混合解析含内网地址时必须拒绝")
        except ValueError:
            pass
    finally:
        socket.getaddrinfo = real
    print("PASS test_mixed_resolution_rejects")


def test_dns_failure_fails_closed():
    """DNS 解析失败必须 fail-closed，不能 fail-open。"""
    import app.tools.fetch_url as fu
    real = socket.getaddrinfo
    def boom(host, *a, **kw):
        raise socket.gaierror("name resolution failed")
    socket.getaddrinfo = boom
    try:
        try:
            fu.check_url("https://nonexistent.invalid/x")
            raise AssertionError("DNS 失败时必须拒绝")
        except ValueError:
            pass
    finally:
        socket.getaddrinfo = real
    print("PASS test_dns_failure_fails_closed")


def test_scheme_guard_unchanged():
    import app.tools.fetch_url as fu
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://example.com/"):
        try:
            fu.check_url(url)
            raise AssertionError("非 http(s) scheme 必须拒绝: " + url)
        except ValueError:
            pass
    print("PASS test_scheme_guard_unchanged")


def main() -> int:
    test_names = sorted([k for k in globals().keys() if k.startswith("test_")])
    passed = 0
    failed = 0
    for name in test_names:
        try:
            globals()[name]()
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print("=" * 40)
    print(f"Passed: {passed}/{len(test_names)}  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
