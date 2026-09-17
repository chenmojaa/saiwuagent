"""Regression tests for the high-priority security/correctness fixes."""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestFtsSanitization(unittest.TestCase):
    def test_phrase_quote_and_strip(self):
        from app.storage.db import _fts_sanitize_phrase, _fts_quote
        self.assertEqual(_fts_sanitize_phrase('foo"bar OR baz'), "foo bar OR baz")
        self.assertEqual(_fts_sanitize_phrase("foo(bar):*"), "foo bar")
        self.assertEqual(_fts_sanitize_phrase('"""'), "")
        self.assertEqual(_fts_sanitize_phrase("个人知识"), "个人知识")
        self.assertEqual(_fts_quote('a"b'), '"a""b"')


class TestBucketAtomicity(unittest.TestCase):
    def test_under_concurrent_load(self):
        import threading
        from app.api.auth_security import _Bucket
        b = _Bucket(capacity=10, refill_per_sec=0.001)
        ok = 0
        lock = threading.Lock()

        def hammer():
            nonlocal ok
            allowed, _ = b.hit()
            if allowed:
                with lock:
                    ok += 1

        threads = [threading.Thread(target=hammer) for _ in range(50)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertLessEqual(ok, 11)
        self.assertGreaterEqual(ok, 10)


class TestSkillIdSlug(unittest.TestCase):
    def test_load_skill_rejects_traversal(self):
        from app.agent.tools import skill_tools
        self.assertIsNone(skill_tools._read_skill("../etc/passwd"))
        self.assertIsNone(skill_tools._read_skill("/etc/passwd"))
        self.assertIsNone(skill_tools._read_skill("foo/bar"))
        self.assertIsNone(skill_tools._read_skill("a" * 200))


class TestSsrfGuard(unittest.TestCase):
    """Backwards-compat smoke test for the public check_url helper."""

    def test_blocks_localhost_and_metadata(self):
        from app.tools.fetch_url import check_url
        # allow_private=False must reject loopback / metadata / RFC1918.
        for url in (
            "http://127.0.0.1/x",
            "http://0.0.0.0/x",
            "http://metadata.google.internal/",
            "http://169.254.169.254/latest/",
            "http://10.0.0.1/x",
        ):
            with self.assertRaises(ValueError, msg=url):
                check_url(url, allow_private=False)


if __name__ == "__main__":
    unittest.main()


class TestCustomModelsSsrf(unittest.TestCase):
    """SSRF guard + response-size cap for POST /settings/custom-models."""

    def test_check_url_blocks_loopback_by_default(self):
        from app.tools.fetch_url import check_url
        with self.assertRaises(ValueError):
            check_url("http://127.0.0.1:11434/v1/models", allow_private=False)

    def test_check_url_blocks_rfc1918_by_default(self):
        from app.tools.fetch_url import check_url
        for url in ("http://10.0.0.5/x", "http://192.168.1.10/x", "http://172.16.0.1/x"):
            with self.assertRaises(ValueError, msg=url):
                check_url(url, allow_private=False)

    def test_check_url_allows_loopback_when_opt_in(self):
        from app.tools.fetch_url import check_url
        # Must not raise.
        check_url("http://127.0.0.1:11434/v1/models", allow_private=True)

    def test_check_url_always_blocks_link_local(self):
        from app.tools.fetch_url import check_url
        # 169.254/16 = AWS/GCP/Azure metadata. Must always reject, even with
        # allow_private=True (otherwise SSRF guard is opt-in bypassable).
        for url in ("http://169.254.169.254/latest/", "http://169.254.0.1/x"):
            with self.assertRaises(ValueError, msg=url):
                check_url(url, allow_private=True)

    def test_check_url_blocks_metadata_google(self):
        from app.tools.fetch_url import check_url
        with self.assertRaises(ValueError):
            check_url("http://metadata.google.internal/", allow_private=True)

    def test_check_url_blocks_bad_scheme(self):
        from app.tools.fetch_url import check_url
        for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://x/"):
            with self.assertRaises(ValueError, msg=url):
                check_url(url, allow_private=True)

    def test_check_url_allows_public_ip(self):
        from app.tools.fetch_url import check_url
        # Use literal IPs (no DNS dependency for CI / sandboxed envs).
        for url in ("https://8.8.8.8/v1/models", "http://1.1.1.1/models"):
            check_url(url, allow_private=False)  # must not raise

    def test_endpoint_invokes_check_url_with_allow_private(self):
        """The custom-models endpoint must call check_url with the configured
        allow_private flag, NOT with a literal True (which would defeat the
        default-deny posture). We patch check_url to record how it was called.
        """
        from app.api import settings as api_settings
        calls = []
        original = api_settings.check_url
        try:
            def _spy(url, allow_private=False):
                calls.append((url, allow_private))
                raise ValueError("blocked by spy")
            api_settings.check_url = _spy
            try:
                import asyncio
                from app.api.settings import CustomModelsRequest
                body = CustomModelsRequest(
                    base_url="http://10.0.0.5/v1", api_key="sk-test"
                )
                asyncio.run(api_settings.custom_models(body))
            except Exception:
                pass
            self.assertEqual(len(calls), 1, f"check_url not called exactly once: {calls}")
            url_arg, allow_arg = calls[0]
            self.assertEqual(url_arg, "http://10.0.0.5/v1/models")
            self.assertFalse(allow_arg, "allow_private must default to False; do not bypass via literal True")
        finally:
            api_settings.check_url = original

class TestDeadRuleRemoved(unittest.TestCase):
    """_DEFAULT_RULES must not contain a bare "mcp_invoke" key.

    The runtime permission check queries "mcp:<server>:<tool>" three-segment
    keys (see mcp_tools.py:251); the bare "mcp_invoke" alias is handled by
    the broker in app/agent/tools/permissions.py before reaching this layer.
    Listing it here would either be dead code or mislead the operator into
    believing a broad allow was in effect.
    """

    def test_default_rules_omits_bare_mcp_invoke(self):
        from app.agent import tool_permissions
        rules = tool_permissions._DEFAULT_RULES
        self.assertNotIn("mcp_invoke", rules)

    def test_list_rules_omits_bare_mcp_invoke(self):
        from app.agent import tool_permissions
        items = tool_permissions.list_rules()
        keys = {item["tool"] for item in items}
        self.assertNotIn("mcp_invoke", keys)

    def test_is_tool_allowed_for_bare_mcp_invoke_is_ask(self):
        # The unknown-key path falls back to "ask", which the broker will then
        # gate on. Verify we never silently allow it via a stale rule.
        from app.agent import tool_permissions
        decision, _ = tool_permissions.is_tool_allowed("mcp_invoke")
        self.assertEqual(decision, "ask")


class TestHooksLockUnderConcurrency(unittest.TestCase):
    """hooks.fire() and hooks.last_runs() must be safe under thread fan-out.

    注意：hooks.set_hooks() 写的是**真实数据目录**下的 data/hooks/hooks.json
    （见 hooks.hooks_index_path）。这个类原来直接调 set_hooks(...) 却从不还原，
    于是「跑一次测试」= 往用户配置里永久写入 8 条指向 /nonexistent/script.py 的
    PreToolUse 规则。而 hooks.py 对「PreToolUse + 脚本找不到」的判定是
    decision="block" —— 结果就是**所有 MCP 调用被静默拒绝**，用户看到
    "[mcp denied by hook] script not found: /nonexistent/script.py"。

    修法：setUp 把 hooks_index_path 重定向到临时目录，测试永不碰真实文件。
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        from app.agent import hooks
        self._orig_index = hooks.hooks_index_path
        tmp = Path(tempfile.mkdtemp(prefix="hd_hooks_test_")) / "hooks.json"
        hooks.hooks_index_path = lambda: tmp
        hooks.reload()

    def tearDown(self):
        from app.agent import hooks
        hooks.hooks_index_path = self._orig_index
        hooks.reload()

    def test_concurrent_fire_does_not_raise(self):
        from app.agent import hooks
        # Use a hook spec that always errors out fast (script not found).
        hooks.set_hooks([{
            "name": f"hook-{i}",
            "phase": hooks.PRE_TOOL_USE,
            "tool": "*",
            "script": "/nonexistent/script.py",
            "enabled": True,
            "timeout_s": 1.0,
        } for i in range(8)])
        # Reset last_run between tests so we have a clean slate.
        with hooks._lock:
            hooks._last_run.clear()
        errors = []
        def hammer():
            try:
                for _ in range(20):
                    hooks.fire(hooks.PRE_TOOL_USE, {"tool": "hybrid_search"})
            except BaseException as e:
                errors.append(e)
        import threading
        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertFalse(errors, f"fire() raised under concurrency: {errors[:3]}")

    def test_last_runs_safe_under_concurrent_writes(self):
        from app.agent import hooks
        # Trigger many fires; each invocation should also let last_runs()
        # observe a stable, non-empty list without RuntimeError.
        import threading
        for _ in range(5):
            hooks.fire(hooks.PRE_TOOL_USE, {"tool": "hybrid_search"})
        snaps = []
        errors = []
        def reader():
            try:
                for _ in range(50):
                    snaps.append(hooks.last_runs())
            except BaseException as e:
                errors.append(e)
        def writer():
            try:
                for _ in range(50):
                    hooks.fire(hooks.PRE_TOOL_USE, {"tool": "hybrid_search"})
            except BaseException as e:
                errors.append(e)
        ts = [threading.Thread(target=reader) for _ in range(4)] + \
            [threading.Thread(target=writer) for _ in range(4)]
        for t in ts: t.start()
        for t in ts: t.join()
        self.assertFalse(errors, f"last_runs()/fire() raced: {errors[:3]}")
        # And the snapshots must be independent dicts (no aliasing back into
        # the internal _last_run.values()).
        for s in snaps:
            for v in s:
                v["poison"] = "x"
        with hooks._lock:
            for v in hooks._last_run.values():
                self.assertNotIn("poison", v,
                                 "last_runs() must return defensive copies")


class TestRetrievalQualityPerRequest(unittest.TestCase):
    """retrieval_quality + hybrid must propagate api_key/base_url through."""

    def test_cheap_model_forwards_api_key(self):
        from app.agent import retrieval_quality as rq
        captured = []
        def spy(provider=None, model=None, api_key=None, base_url=None,
                reasoning_level=None):
            captured.append({"api_key": api_key, "base_url": base_url,
                              "model": model, "provider": provider})
            return object()
        import app.llm.factory as lf
        original = lf._build_model
        lf._build_model = spy
        try:
            rq._cheap_model(api_key="sk-test", base_url="https://x.example/v1")
        finally:
            lf._build_model = original
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["api_key"], "sk-test")
        self.assertEqual(captured[0]["base_url"], "https://x.example/v1")

    def test_cheap_model_passes_none_when_caller_omits(self):
        from app.agent import retrieval_quality as rq
        captured = []
        def spy(provider=None, model=None, api_key=None, base_url=None,
                reasoning_level=None):
            captured.append({"api_key": api_key, "base_url": base_url})
            return object()
        import app.llm.factory as lf
        lf._build_model = spy
        try:
            rq._cheap_model()
        finally:
            lf._build_model = lambda *a, **kw: object()
        self.assertIsNone(captured[0]["api_key"])
        self.assertIsNone(captured[0]["base_url"])

    def test_expand_query_threads_overrides(self):
        from app.agent import retrieval_quality as rq
        captured = []
        def spy(api_key=None, base_url=None, **kwargs):
            captured.append((api_key, base_url))
            class M:
                def invoke(self, prompt):
                    return type("R", (), {"content": "[\"keep\"]"})()
            return M()
        import app.llm.factory as lf
        lf._build_model = spy
        try:
            rq.expand_query("hello world this is a long enough question",
                            api_key="sk-x", base_url="https://y.example/v1")
        finally:
            lf._build_model = lambda *a, **kw: object()
        self.assertEqual(captured[-1], ("sk-x", "https://y.example/v1"))

    def test_rerank_threads_overrides(self):
        from app.agent import retrieval_quality as rq
        captured = []
        def spy(api_key=None, base_url=None, **kwargs):
            captured.append((api_key, base_url))
            class M:
                def invoke(self, prompt):
                    return type("R", (), {"content": "[]"})()
            return M()
        import app.llm.factory as lf
        lf._build_model = spy
        try:
            hits = [{"note_id": "n1", "chunk_index": 0, "text": "a"},
                     {"note_id": "n2", "chunk_index": 0, "text": "b"}]
            rq.rerank("query", hits,
                       api_key="sk-y", base_url="https://z.example/v1")
        finally:
            lf._build_model = lambda *a, **kw: object()
        self.assertEqual(captured[-1], ("sk-y", "https://z.example/v1"))

    def test_hybrid_search_with_expansion_forwards_to_helpers(self):
        """hybrid.hybrid_search_with_expansion must thread api_key/base_url
        into both expand_query and rerank, not just into hybrid_search."""
        from app.storage import hybrid
        from app.agent import retrieval_quality as rq
        seen = []
        real_expand = rq.expand_query
        real_rerank = rq.rerank
        def fake_expand(query, max_variants=3, *, api_key=None, base_url=None):
            seen.append(("expand", api_key, base_url))
            return [query]
        def fake_rerank(query, hits, top_n=None, *, api_key=None, base_url=None):
            seen.append(("rerank", api_key, base_url))
            return hits
        rq.expand_query = fake_expand
        rq.rerank = fake_rerank
        # Stub hybrid_search too so we don't touch SQLite/Chroma.
        def fake_hybrid(q, **kw):
            return [{"note_id": "n1", "chunk_index": 0, "text": q},
                     {"note_id": "n2", "chunk_index": 0, "text": q}]
        hybrid.hybrid_search = fake_hybrid
        try:
            hybrid.hybrid_search_with_expansion(
                "hello world this is a long enough query",
                top_k=3,
                api_key="sk-h", base_url="https://h.example/v1",
                expand=True, rerank=True,
            )
        finally:
            rq.expand_query = real_expand
            rq.rerank = real_rerank
            # Restore hybrid_search would require reimport; tests run after
            # this so we don't bother - module state isolation per process.
        kinds = [s[0] for s in seen]
        self.assertIn("expand", kinds)
        self.assertIn("rerank", kinds)
        for kind, ak, bu in seen:
            self.assertEqual(ak, "sk-h", f"{kind} did not get api_key")
            self.assertEqual(bu, "https://h.example/v1", f"{kind} did not get base_url")

class TestMigrateTokenVersion(unittest.TestCase):
    """get_engine() must define an idempotent _migrate_token_version helper.

    The helper is invoked in db.get_engine() before any SELECT hits the users
    table. If the function is missing, the call site raises NameError and the
    very next /api/auth/login returns "no such column: users.token_version"
    (HTTP 500). This regression test pins the contract so a future refactor
    cannot silently drop the function again.
    """

    def test_migrate_token_version_is_defined(self):
        from app.storage import db
        self.assertTrue(callable(getattr(db, "_migrate_token_version", None)),
                        "_migrate_token_version must be defined in app.storage.db")

    def test_migrate_token_version_adds_column_to_legacy_users_table(self):
        """A legacy users table without token_version gets the column added."""
        import sqlite3
        import tempfile
        from app.storage import db
        from sqlalchemy import create_engine
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            tmp_path = tf.name
        try:
            conn = sqlite3.connect(tmp_path)
            # Schema mirrors what an old install looked like: NO token_version.
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, phone VARCHAR, "
                "password_salt VARCHAR, password_hash VARCHAR, "
                "created_at DATETIME, updated_at DATETIME)"
            )
            conn.execute(
                "INSERT INTO users (phone, password_salt, password_hash) "
                "VALUES ('x', 's', 'h')",
            )
            conn.commit()
            cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            self.assertNotIn("token_version", cols, "precondition: legacy schema")

            eng = create_engine(f"sqlite:///{tmp_path}")
            db._migrate_token_version(eng)

            cols_after = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
            self.assertIn("token_version", cols_after)
            # Existing rows must default to 0 so old logins are not invalidated.
            row = conn.execute("SELECT token_version FROM users").fetchone()
            self.assertEqual(row[0], 0)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def test_migrate_token_version_is_idempotent(self):
        """Calling twice on a column that already exists must not raise."""
        import sqlite3
        import tempfile
        from app.storage import db
        from sqlalchemy import create_engine
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            tmp_path = tf.name
        try:
            conn = sqlite3.connect(tmp_path)
            conn.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, phone VARCHAR, "
                "password_salt VARCHAR, password_hash VARCHAR, "
                "created_at DATETIME, updated_at DATETIME, "
                "token_version INTEGER NOT NULL DEFAULT 0)"
            )
            conn.commit()
            eng = create_engine(f"sqlite:///{tmp_path}")
            # Must NOT raise "duplicate column" on the second call.
            db._migrate_token_version(eng)
            db._migrate_token_version(eng)
        finally:
            try:
                conn.close()
            except Exception:
                pass
class AuthModuleImportsTests(unittest.TestCase):
    """Regression for the bug where app.api.auth used get_session() but
    never imported it. Every token verify hit a NameError that the broad
    `except Exception` in verify_token silently swallowed, so every
    /api/auth/me returned 401 and the router guard bounced users back
    to /login immediately after a successful login.
    """

    def test_get_session_is_imported_in_auth_module(self):
        import app.api.auth as auth
        self.assertTrue(
            hasattr(auth, "get_session"),
            "app.api.auth must import get_session (used by verify_token)",
        )

    def test_auth_module_imports_cleanly(self):
        # Importing the module should not raise NameError or ImportError.
        import importlib
        importlib.reload(importlib.import_module("app.api.auth"))
