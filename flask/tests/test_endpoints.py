from __future__ import annotations

import time


def _seed(ts, tenant_id: str, items: dict):
    with ts.transcripts_lock:
        ts.transcriptd[tenant_id] = {k: {"transcript": v} for k, v in items.items()}


def test_session_post_mints_tenant_for_valid_source(client, ts):
    resp = client.post("/session", json={"source": "mic"})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["source"] == "mic"
    assert isinstance(body["tenant_id"], str) and len(body["tenant_id"]) > 0

    with ts.session_lock:
        entry = ts.latest_session_by_source["mic"]
    assert entry is not None
    tenant_id, created_ts = entry
    assert tenant_id == body["tenant_id"]
    assert abs(created_ts - time.time()) < 5


def test_session_post_rejects_unknown_source(client):
    resp = client.post("/session", json={"source": "totally-bogus"})
    assert resp.status_code == 400
    body = resp.get_json()
    assert "source must be one of" in body.get("error", "")


def test_transcripts_post_enqueues_and_returns_accepted(client, ts):
    """New REST endpoint: POST /transcripts returns 202 Accepted (async)."""
    payload = {
        "audio_b64": "AAAA",
        "chunk_id": "12345",
        "tenant_id": "tenant-x",
    }
    resp = client.post("/transcripts", json=payload)
    assert resp.status_code == 202
    body = resp.get_json()
    assert body == {"chunk_id": "12345", "tenant_id": "tenant-x", "status": "processing"}

    # Worker is disabled in tests; the item should still be on the queue.
    assert ts.audio_stack.qsize() == 1
    queued = ts.audio_stack.get_nowait()
    ts.audio_stack.task_done()
    assert queued == ("tenant-x", "12345", "AAAA")


def test_transcripts_post_rejects_missing_fields(client):
    resp = client.post("/transcripts", json={"chunk_id": "1"})
    assert resp.status_code == 400
    assert "Missing required fields" in resp.get_json().get("error", "")


def test_legacy_transcribe_still_returns_200(client, ts):
    """Deprecated /transcribe alias preserves the historical 200 status."""
    payload = {"audio_b64": "AAAA", "chunk_id": "12345", "tenant_id": "tenant-x"}
    resp = client.post("/transcribe", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"chunk_id": "12345", "tenant_id": "tenant-x", "status": "processing"}
    assert ts.audio_stack.qsize() == 1
    ts.audio_stack.get_nowait()
    ts.audio_stack.task_done()


def test_transcribe_rejects_missing_fields(client):
    resp = client.post("/transcripts", json={"chunk_id": "1"})
    assert resp.status_code == 400
    assert "Missing required fields" in resp.get_json().get("error", "")


def test_transcribe_rejects_empty_payload(client):
    resp = client.post("/transcripts", data="", content_type="application/json")
    assert resp.status_code in (400, 415)


def test_list_transcripts_filters_by_from_until(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b", "900": "c"})

    resp = client.get("/transcripts?tenant_id=t1&from=200&until=800")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "500" in body
    assert "100" not in body
    assert "900" not in body


def test_list_transcripts_rejects_non_integer_from(client, ts):
    _seed(ts, "t1", {"100": "a"})
    resp = client.get("/transcripts?tenant_id=t1&from=notanint")
    assert resp.status_code == 400


def test_get_transcript_returns_404_when_no_session(client):
    resp = client.get("/transcripts/123?source=mic")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body == {"error": "Transcript not found", "chunk_id": "123"}


def test_get_transcript_rejects_unknown_source(client):
    resp = client.get("/transcripts/123?source=microphone")
    assert resp.status_code == 400


def test_get_transcript_finds_seeded_entry(client, ts):
    _seed(ts, "t1", {"42": "hello world"})
    resp = client.get("/transcripts/42?tenant_id=t1")
    assert resp.status_code == 200
    assert resp.get_json() == {"chunk_id": "42", "transcript": "hello world"}


def test_get_transcript_returns_404_for_missing_chunk(client, ts):
    _seed(ts, "t1", {"42": "hello world"})
    resp = client.get("/transcripts/99999999?tenant_id=t1")
    assert resp.status_code == 404
    assert resp.get_json() == {
        "error": "Transcript not found",
        "chunk_id": "99999999",
    }


def test_delete_transcript_removes_entry(client, ts):
    _seed(ts, "t1", {"42": "hello world", "43": "keep me"})
    resp = client.delete("/transcripts/42?tenant_id=t1")
    assert resp.status_code == 200
    assert resp.get_json() == {"chunk_id": "42", "transcript": "hello world"}
    # 42 gone, 43 remains
    with ts.transcripts_lock:
        remaining = set(ts.transcriptd["t1"].keys())
    assert remaining == {"43"}


def test_delete_transcript_returns_204_when_chunk_absent(client, ts):
    _seed(ts, "t1", {"42": "hello world"})
    resp = client.delete("/transcripts/99999999?tenant_id=t1")
    assert resp.status_code == 204
    assert resp.data == b""
    # The seeded chunk must remain untouched.
    with ts.transcripts_lock:
        remaining = set(ts.transcriptd["t1"].keys())
    assert remaining == {"42"}


def test_delete_transcript_returns_204_when_tenant_empty(client):
    resp = client.delete("/transcripts/42?tenant_id=does-not-exist")
    assert resp.status_code == 204
    assert resp.data == b""


def test_delete_transcript_is_idempotent(client, ts):
    _seed(ts, "t1", {"42": "hello"})
    # First delete: 200 with body.
    resp = client.delete("/transcripts/42?tenant_id=t1")
    assert resp.status_code == 200
    # Second delete of the same id: 204 No Content (no error).
    resp = client.delete("/transcripts/42?tenant_id=t1")
    assert resp.status_code == 204
    assert resp.data == b""


def test_first_transcript_returns_lowest_chunk(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b", "900": "c"})
    resp = client.get("/transcripts/first?tenant_id=t1")
    assert resp.status_code == 200
    assert resp.get_json() == {"chunk_id": "100", "transcript": "a"}


def test_first_transcript_returns_204_when_empty(client):
    resp = client.get("/transcripts/first?tenant_id=does-not-exist")
    assert resp.status_code == 204
    assert resp.data == b""


def test_first_transcript_delete_returns_204_when_empty(client):
    resp = client.delete("/transcripts/first?tenant_id=does-not-exist")
    assert resp.status_code == 204
    assert resp.data == b""


def test_first_transcript_returns_204_when_from_filter_excludes_all(client, ts):
    _seed(ts, "t1", {"100": "a", "200": "b"})
    resp = client.get("/transcripts/first?tenant_id=t1&from=9999")
    assert resp.status_code == 204
    assert resp.data == b""


def test_first_transcript_delete_pops_lowest_chunk(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b"})
    resp = client.delete("/transcripts/first?tenant_id=t1")
    assert resp.status_code == 200
    assert resp.get_json() == {"chunk_id": "100", "transcript": "a"}
    with ts.transcripts_lock:
        remaining = set(ts.transcriptd["t1"].keys())
    assert remaining == {"500"}


def test_latest_transcript_returns_highest_chunk(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b", "900": "c"})
    resp = client.get("/transcripts/latest?tenant_id=t1")
    assert resp.status_code == 200
    assert resp.get_json() == {"chunk_id": "900", "transcript": "c"}


def test_latest_transcript_returns_204_when_empty(client):
    resp = client.get("/transcripts/latest?tenant_id=does-not-exist")
    assert resp.status_code == 204
    assert resp.data == b""


def test_latest_transcript_delete_returns_204_when_empty(client):
    resp = client.delete("/transcripts/latest?tenant_id=does-not-exist")
    assert resp.status_code == 204
    assert resp.data == b""


def test_transcripts_count_counts_within_range(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b", "900": "c"})
    resp = client.get("/transcripts/count?tenant_id=t1&from=0&until=1000")
    assert resp.status_code == 200
    assert resp.get_json() == {"size": 3}

    resp = client.get("/transcripts/count?tenant_id=t1&from=200&until=800")
    assert resp.get_json() == {"size": 1}


def test_legacy_list_transcripts_alias_still_works(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b"})
    resp = client.get("/list_transcripts?tenant_id=t1&from=0&until=1000")
    assert resp.status_code == 200
    assert set(resp.get_json().keys()) == {"100", "500"}


def test_legacy_transcripts_size_alias_still_works(client, ts):
    _seed(ts, "t1", {"100": "a", "500": "b"})
    resp = client.get("/transcripts_size?tenant_id=t1&from=0&until=1000")
    assert resp.status_code == 200
    assert resp.get_json() == {"size": 2}


def test_swagger_has_distinct_models(client):
    resp = client.get("/swagger.json")
    assert resp.status_code == 200
    spec = resp.get_json()
    definitions = spec.get("definitions") or spec.get("components", {}).get("schemas") or {}
    assert "Transcript" in definitions
    assert "TranscribeAck" in definitions
