from flask import Flask, request, jsonify, abort
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import numpy as np
import threading
import requests
import logging
import base64
import queue
import time
import uuid
import wave
import io
import os
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> list:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]

app = Flask(__name__)
api = Api(app, version='1.0', title='Transcription API',
          description='A simple Transcription API', doc='/swagger')

# CORS_ALLOWED_ORIGINS is a comma-separated list. Default is local-dev only.
# Use "*" explicitly if (and only if) you really want to allow any origin.
_cors_origins = _env_csv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5040,http://127.0.0.1:5040",
)
CORS(app, resources={r"/*": {"origins": _cors_origins}})
logger.info(f"CORS allowed origins: {_cors_origins}")

# We either use a local in-code model or access a whisper.cpp server.
use_whisper_server = _env_bool('WHISPER_SERVER_USE', False)
_legacy_model = os.getenv('WHISPER_MODEL')
model_fast_name = os.getenv('WHISPER_MODEL_FAST', _legacy_model or 'small')    # 244M
model_smart_name = os.getenv('WHISPER_MODEL_SMART', _legacy_model or 'medium')  # 769M
device = None
whisper_server = os.getenv('WHISPER_SERVER', 'http://localhost:8007').rstrip('/')

# Models are only loaded when we are NOT using the whisper.cpp server.
model_fast = None
model_smart = None

if use_whisper_server:
    logger.info(f"Whisper backend: server at {whisper_server}/inference")
else:
    import torch 
    import whisper  

    logger.info("TORCH CUDA: %s", torch.cuda.is_available())
    logger.info("DEVICE COUNT: %s", torch.cuda.device_count())
    logger.info(f"Hardware detection: using {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, 'models')

    def _load_whisper_model(name: str):
        local_pt = os.path.join(models_path, name + ".pt")
        if os.path.exists(local_pt):
            return whisper.load_model(name, device=device, in_memory=True, download_root=models_path)
        return whisper.load_model(name, device=device, in_memory=True)

    logger.info(f"Whisper backend: local models fast={model_fast_name}, smart={model_smart_name}")
    model_fast = _load_whisper_model(model_fast_name)
    model_smart = _load_whisper_model(model_smart_name)

# transcripts:  tenant_id -> { chunk_id -> {'transcript': str} }
transcriptd = {}
transcripts_lock = threading.Lock()

# FIFO queue of pending audio chunks awaiting transcription.
audio_stack = queue.Queue()
VALID_SOURCES = {"mic", "file", "url", "stdin", "youtube"}
latest_session_by_source = {s: None for s in VALID_SOURCES}  # source -> (tenant_id, created_ts) or None
session_lock = threading.Lock()
SESSION_TTL_SECONDS = int(os.getenv('SESSION_TTL_SECONDS', '7200'))


# Small helpers

def _parse_int_arg(args, name: str, default: int = None, required: bool = False) -> int:
    """
    Parse a query-string argument as an int. On invalid input, abort with HTTP
    400 instead of letting `int()` raise and be turned into a 500.

    Returns ``default`` if the argument is missing and not required.
    """
    raw = args.get(name)
    if raw is None or raw == "":
        if required:
            abort(400, f"Missing required query parameter: {name}")
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        abort(400, f"Query parameter {name!r} must be an integer, got {raw!r}")


def _chunk_id_int(k):
    """
    Best-effort int() of a chunk_id. Returns ``None`` for keys that cannot
    be interpreted as integers, so callers can defensively skip them
    rather than crashing the endpoint with a 500.
    """
    try:
        return int(k)
    except (TypeError, ValueError):
        return None


def _numeric_sorted_keys(transcripts, reverse: bool = False) -> list:
    """
    Return the chunk_ids of ``transcripts`` sorted numerically, skipping
    any that can't be parsed as ints. Used by every endpoint that does
    "first" / "latest" / range-filtered lookups.
    """
    pairs = []
    for k in transcripts.keys():
        n = _chunk_id_int(k)
        if n is not None:
            pairs.append((n, k))
    pairs.sort(reverse=reverse)
    return [k for _, k in pairs]


def _in_chunk_range(k, fromid: int, untilid: int) -> bool:
    """``True`` iff ``k`` parses to an int and lies within [fromid, untilid]."""
    n = _chunk_id_int(k)
    return n is not None and fromid <= n <= untilid


def _resolve_tenant(args, default='0000'):
    """
    Resolve which tenant_id a read request is targeting.

    Priority:
      1. Explicit ?tenant_id=<id> wins (covers manual override / debugging).
      2. ?source=<mic|file|url|stdin|youtube> resolves to the most recently
         registered, non-expired session for that source. An unknown
         source value aborts with HTTP 400 so client typos surface
         loudly instead of masquerading as "no transcripts yet". A known
         source with no active session returns None so the caller can
         short-circuit with an empty response.
      3. Fall back to ``default`` (legacy behaviour).
    """
    explicit = args.get('tenant_id')
    if explicit:
        return explicit
    source = args.get('source')
    if source:
        if source not in VALID_SOURCES:
            abort(
                400,
                f"Invalid source '{source}'. "
                f"Must be one of: {sorted(VALID_SOURCES)}.",
            )
        now = time.time()
        with session_lock:
            entry = latest_session_by_source.get(source)
            if entry is None:
                return None
            tenant_id, created_ts = entry
            if now - created_ts > SESSION_TTL_SECONDS:
                # Expire stale session pointer.
                latest_session_by_source[source] = None
                return None
            return tenant_id
    return default


def _pcm_int16_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Wrap a mono 16-bit PCM numpy array in a minimal RIFF/WAV container so it
    can be POSTed to whisper.cpp's /inference endpoint, which insists on a
    real audio file (raw PCM bytes will be rejected).
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16, copy=False).tobytes())
    return buf.getvalue()


def _whisper_server_transcribe(audio_int16: np.ndarray) -> dict:
    """
    POST a single chunk to whisper.cpp's /inference endpoint and return its
    JSON-decoded body. Raises requests.RequestException on transport errors.
    """
    wav_bytes = _pcm_int16_to_wav_bytes(audio_int16)
    files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}
    data = {'response_format': 'json'}
    inference_url = whisper_server + '/inference'
    response = requests.post(inference_url, files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()

def _next_payload():
    """
    Pull the next audio payload from ``audio_stack``, dropping any superseded
    duplicates so we only transcribe the latest version of each
    (tenant_id, chunk_id).
    """
    tenant_id, chunk_id, audiob64 = audio_stack.get()
    while True:
        with audio_stack.mutex:
            has_newer = any(
                t == tenant_id and c == chunk_id
                for (t, c, _) in audio_stack.queue
            )
        if not has_newer:
            return tenant_id, chunk_id, audiob64
        # Current entry is stale; discard it (correctly accounted) and grab
        # the next one from the head.
        audio_stack.task_done()
        tenant_id, chunk_id, audiob64 = audio_stack.get()


def process_audio():
    while True:
        tenant_id, chunk_id, audiob64 = _next_payload()
        logger.debug(f"Queue length: {audio_stack.qsize()}")
        try:
            audio_data = base64.b64decode(audiob64)
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            if audio_int16.size == 0:
                logger.warning(f"Invalid audio data for chunk_id {chunk_id}")
                continue

            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            if np.isnan(audio_float32).any():
                logger.warning(f"NaN values in audio array for chunk_id {chunk_id}")
                continue

            qsize = audio_stack.qsize()
            if use_whisper_server:
                # Whisper.cpp server doesn't expose a fast/smart distinction;
                # send everything to /inference. The server itself decides
                # how to schedule it.
                try:
                    result = _whisper_server_transcribe(audio_int16)
                except requests.RequestException as exc:
                    logger.error(f"Whisper server error for chunk_id {chunk_id}: {exc}")
                    continue
            else:
                # Local-model branch: torch was already imported at module
                # load time when use_whisper_server=False, so this is cheap.
                import torch 
                model = model_fast if qsize > 20 else model_smart
                audio_tensor = torch.from_numpy(audio_float32)
                result = model.transcribe(audio_tensor, temperature=0)

            transcript = (result.get('text') or '').strip()

            if is_valid(transcript):
                logger.info(f"VALID transcript for chunk_id {chunk_id}: {transcript}")
                with transcripts_lock:
                    transcripts = transcriptd.get(tenant_id)
                    if not transcripts:
                        transcripts = {}
                        transcriptd[tenant_id] = transcripts

                    current_transcript = transcripts.get(chunk_id)
                    if current_transcript:
                        # buffer for the same chunk, so overwrite rather than concatenate.
                        current_transcript['transcript'] = transcript
                    else:
                        transcripts[chunk_id] = {'transcript': transcript}
            else:
                logger.warning(f"INVALID transcript for chunk_id {chunk_id}: {transcript}")

            # Periodic GC of stale tenants/chunks.
            clean_old_transcripts()

        except Exception:
            logger.error(f"Error processing audio chunk {chunk_id}", exc_info=True)
        finally:
            audio_stack.task_done()


# Check if the transcript is valid: Contains at least one ASCII character and no forbidden words
def is_valid(transcript):
    transcript_lower = transcript.lower()
    # Check for at least one ASCII character with a code < 128 and code > 32 (we omit space in this case)
    has_ascii_char = any(32 < ord(char) < 128 for char in transcript)

    # Check for forbidden words (case insensitive)
    forbidden_phrases = {"thank you", "bye!", "thanks for watching", "click, click", "click click", "cough cough", "뉴", "스", "김", "수", "근", "입", "니", "다"}
    contains_forbidden_phrases = any(word in transcript_lower for word in forbidden_phrases)
    forbidden_strings = {"eh.", "you", "bye.", "it's fine"}
    is_forbidden_string = any(word == transcript_lower for word in forbidden_strings)

    # check if the transcript has words which are longer than 40 characters
    contains_long_words = any(len(word) > 40 for word in transcript.split())

    # Return true only if both conditions are met
    return has_ascii_char and not contains_forbidden_phrases and not is_forbidden_string and not contains_long_words


# Clean old transcripts: remove all chunks older than two hours and any tenants
def clean_old_transcripts():
    current_time_ms = int(time.time() * 1000)
    two_hours_ago_ms = current_time_ms - (2 * 60 * 60 * 1000)

    with transcripts_lock:
        empty_tenants = []
        # Snapshot the tenant ids before iterating; we mutate inside the loop.
        for tenant_id in list(transcriptd.keys()):
            transcripts = transcriptd.get(tenant_id)
            if not transcripts:
                empty_tenants.append(tenant_id)
                continue

            # Snapshot chunk ids; some chunk_ids may be non-numeric in principle, so we defensively skip those rather than crashing the worker thread.
            stale_chunks = []
            for chunk_id in list(transcripts.keys()):
                try:
                    if int(chunk_id) < two_hours_ago_ms:
                        stale_chunks.append(chunk_id)
                except (TypeError, ValueError):
                    # Unknown id format -> leave it alone.
                    continue

            for chunk_id in stale_chunks:
                transcripts.pop(chunk_id, None)

            if not transcripts:
                empty_tenants.append(tenant_id)

        for tenant_id in empty_tenants:
            transcriptd.pop(tenant_id, None)

def merge_and_split_transcripts(transcripts):
    """
    Take a ``{chunk_id: {'transcript': str}}`` mapping and produce a new
    mapping of the same shape where text has been re-flowed onto sentence
    boundaries (``.``, ``!``, ``?``).

    The output preserves chunk_ids from the input (a subset of them: only
    the chunk_ids at which a sentence boundary actually falls, plus the
    last chunk for any trailing fragment). Values are dicts with a
    ``'transcript'`` key so callers can use the same access pattern as
    the underlying ``transcriptd`` store.
    """
    sec = ".!?"
    merged = ""
    result = {}
    keys = list(transcripts.keys())
    for key in keys:
        raw = transcripts[key]
        text = (raw.get('transcript') if isinstance(raw, dict) else str(raw or '')).strip()

        if not merged:
            merged += text
        else:
            if len(text) > 1:
                merged += " " + text[0].lower() + text[1:]
            elif text:
                merged += " " + text

        # Drain every complete sentence currently in `merged` onto this key.
        while any(char in sec for char in merged):
            index = next(i for i, c in enumerate(merged) if c in sec)
            head = merged[:index + 1].strip()
            head = head[0].capitalize() + head[1:] if len(head) > 1 else head
            existing = result.get(key, {}).get('transcript')
            if existing:
                result[key] = {'transcript': existing + " " + head}
            else:
                result[key] = {'transcript': head}
            merged = merged[index + 1:].strip()

    # Any leftover (no terminal punctuation) attaches to the final input key.
    if merged and keys:
        last_key = keys[-1]
        existing = result.get(last_key, {}).get('transcript')
        if existing:
            result[last_key] = {'transcript': existing + " " + merged}
        else:
            result[last_key] = {'transcript': merged}

    return result

transcribe_input_model = api.model('Transcribe', {
    'audio_b64': fields.String(required=True, description='Base64 encoded audio data'),
    'chunk_id': fields.String(required=True, description='ID of the audio chunk'),
    'tenant_id': fields.String(required=False, description='Tenant ID', default='0000')
})

transcribe_response_model = api.model('TranscribeAck', {
    'chunk_id': fields.String(description='ID of the audio chunk'),
    'tenant_id': fields.String(description='Tenant ID'),
    'status': fields.String(description='processing flag')
})

transcript_response_model = api.model('Transcript', {
    'chunk_id': fields.String(description='ID of the audio chunk'),
    'transcript': fields.String(description='The transcribed text')
})

list_transcripts_response_model = api.model('ListTranscriptsResponse', {
    'transcripts': fields.List(fields.Nested(transcript_response_model), description='List of transcripts')
})

size_response_model = api.model('SizeResponse', {
    'size': fields.Integer(description='The number of transcripts')
})

session_input_model = api.model('SessionRequest', {
    'source': fields.String(
        required=True,
        description='Input source name; one of: mic, file, url, stdin, youtube',
        enum=sorted(VALID_SOURCES),
    ),
})

session_response_model = api.model('SessionResponse', {
    'tenant_id': fields.String(description='Server-minted tenant ID for this run'),
    'source': fields.String(description='Source name this session is registered under'),
})



# Shared Swagger parameter blocks (referenced by the REST resources below).
_TENANT_PARAM = {'description': 'Tenant ID', 'default': '0000'}
_SOURCE_PARAM = {
    'description': 'Resolve to the latest session for a source (mic|file|url|stdin|youtube). '
                   'Ignored if tenant_id is given. Unknown values return HTTP 400.',
    'type': 'string',
    'enum': ['mic', 'file', 'url', 'stdin', 'youtube'],
}
_SENTENCES_PARAM = {'description': 'Merge and split transcripts into sentences', 'type': 'boolean', 'default': False}
_FROM_PARAM = {'description': 'Starting chunk ID', 'type': 'string', 'default': '0'}
_UNTIL_PARAM = {'description': 'End chunk ID (defaults to "now" in ms)', 'type': 'string'}

_EMPTY_TRANSCRIPT = {'chunk_id': '-1', 'transcript': ''}


def _wants_sentences() -> bool:
    return request.args.get('sentences', default='false').strip().lower() == 'true'


def _session_logic(success_status: int = 201):
    data = request.get_json(force=True, silent=True) or {}
    source = data.get('source') or request.args.get('source')
    if source not in VALID_SOURCES:
        return {"error": f"source must be one of {sorted(VALID_SOURCES)}"}, 400

    new_tenant_id = uuid.uuid4().hex
    with session_lock:
        latest_session_by_source[source] = (new_tenant_id, time.time())

    logger.info(f"New session for source={source}: tenant_id={new_tenant_id}")
    return {"tenant_id": new_tenant_id, "source": source}, success_status


def _transcribe_logic(success_status: int = 202):
    data = request.get_json(force=True, silent=True)
    if not data:
        return {"error": "No JSON payload received"}, 400

    audio_b64 = data.get('audio_b64')
    chunk_id = data.get('chunk_id')
    tenant_id = data.get('tenant_id', '0000')

    if not audio_b64 or not chunk_id:
        return {"error": "Missing required fields"}, 400

    # push to processing queue
    audio_stack.put((tenant_id, chunk_id, audio_b64))
    return {"chunk_id": chunk_id, "tenant_id": tenant_id, "status": "processing"}, success_status


def _get_transcript_logic(chunk_id):
    tenant_id = _resolve_tenant(request.args)
    with transcripts_lock:
        t = dict(transcriptd.get(tenant_id, {}))
    if len(t) == 0:
        return dict(_EMPTY_TRANSCRIPT)
    if _wants_sentences():
        t = merge_and_split_transcripts(t)
    chunk_id = None if chunk_id is None else str(chunk_id)
    if chunk_id in t:
        return {'chunk_id': chunk_id, 'transcript': t[chunk_id]['transcript']}
    return {'chunk_id': chunk_id, 'transcript': ''}


def _first_transcript_logic():
    tenant_id = _resolve_tenant(request.args)
    with transcripts_lock:
        t = dict(transcriptd.get(tenant_id, {}))
    if len(t) == 0:
        return dict(_EMPTY_TRANSCRIPT)
    if _wants_sentences():
        t = merge_and_split_transcripts(t)
    fromid = _parse_int_arg(request.args, 'from', default=0)
    first_chunk_id = next(
        (k for k in _numeric_sorted_keys(t) if _chunk_id_int(k) >= fromid),
        None,
    )
    if first_chunk_id is None:
        return dict(_EMPTY_TRANSCRIPT)
    return {'chunk_id': first_chunk_id, 'transcript': t[first_chunk_id]['transcript']}


def _pop_first_logic():
    tenant_id = _resolve_tenant(request.args)
    sentences = _wants_sentences()
    fromid = _parse_int_arg(request.args, 'from', default=0)

    with transcripts_lock:
        stored = transcriptd.get(tenant_id)
        if not stored:
            return dict(_EMPTY_TRANSCRIPT)

        view = merge_and_split_transcripts(stored) if sentences else stored
        first_chunk_id = next(
            (k for k in _numeric_sorted_keys(view) if _chunk_id_int(k) >= fromid),
            None,
        )
        if first_chunk_id is None:
            return dict(_EMPTY_TRANSCRIPT)

        entry = stored.pop(first_chunk_id, None)
        if sentences:
            first_transcript = view[first_chunk_id]['transcript']
        else:
            first_transcript = entry['transcript'] if entry else ''
    return {'chunk_id': first_chunk_id, 'transcript': first_transcript}


def _latest_transcript_logic():
    tenant_id = _resolve_tenant(request.args)
    with transcripts_lock:
        t = dict(transcriptd.get(tenant_id, {}))
    if len(t) == 0:
        return dict(_EMPTY_TRANSCRIPT)
    if _wants_sentences():
        t = merge_and_split_transcripts(t)
    untilid = _parse_int_arg(request.args, 'until', default=int(time.time() * 1000))
    latest_chunk_id = next(
        (k for k in _numeric_sorted_keys(t, reverse=True) if _chunk_id_int(k) < untilid),
        None,
    )
    if latest_chunk_id is None:
        return dict(_EMPTY_TRANSCRIPT)
    return {'chunk_id': latest_chunk_id, 'transcript': t[latest_chunk_id]['transcript']}


def _pop_latest_logic():
    tenant_id = _resolve_tenant(request.args)
    sentences = _wants_sentences()
    untilid = _parse_int_arg(request.args, 'until', default=int(time.time() * 1000))

    with transcripts_lock:
        stored = transcriptd.get(tenant_id)
        if not stored:
            return dict(_EMPTY_TRANSCRIPT)

        view = merge_and_split_transcripts(stored) if sentences else stored
        latest_chunk_id = next(
            (k for k in _numeric_sorted_keys(view, reverse=True) if _chunk_id_int(k) < untilid),
            None,
        )
        if latest_chunk_id is None:
            return dict(_EMPTY_TRANSCRIPT)

        entry = stored.pop(latest_chunk_id, None)
        if sentences:
            latest_transcript = view[latest_chunk_id]['transcript']
        else:
            latest_transcript = entry['transcript'] if entry else ''
    return {'chunk_id': latest_chunk_id, 'transcript': latest_transcript}


def _delete_transcript_logic(chunk_id):
    tenant_id = _resolve_tenant(request.args)
    chunk_id = None if chunk_id is None else str(chunk_id)
    with transcripts_lock:
        stored = transcriptd.get(tenant_id, {})
        if chunk_id in stored:
            entry = stored.pop(chunk_id, None)
            return {'chunk_id': chunk_id, 'transcript': entry['transcript']}
    return {'chunk_id': chunk_id, 'transcript': ''}


def _list_transcripts_logic():
    tenant_id = _resolve_tenant(request.args)
    sentences = _wants_sentences()
    fromid = _parse_int_arg(request.args, 'from', default=0)
    untilid = _parse_int_arg(request.args, 'until', default=int(time.time() * 1000))
    with transcripts_lock:
        t = dict(transcriptd.get(tenant_id, {}))
    if sentences:
        t = merge_and_split_transcripts(t)
    return {k: v for k, v in t.items() if _in_chunk_range(k, fromid, untilid)}


def _transcripts_size_logic():
    tenant_id = _resolve_tenant(request.args)
    sentences = _wants_sentences()
    fromid = _parse_int_arg(request.args, 'from', default=0)
    untilid = _parse_int_arg(request.args, 'until', default=int(time.time() * 1000))
    with transcripts_lock:
        t = dict(transcriptd.get(tenant_id, {}))
    if sentences:
        t = merge_and_split_transcripts(t)
    t = {k: v for k, v in t.items() if _in_chunk_range(k, fromid, untilid)}
    return {'size': len(t)}


@api.route('/session')
class Session(Resource):
    @api.expect(session_input_model)
    @api.response(201, 'Created', session_response_model)
    @api.response(400, 'Invalid source')
    def post(self):
        try:
            return _session_logic(success_status=201)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in POST /session", exc_info=True)
            return {"error": str(e)}, 500


@api.route('/transcripts')
class Transcripts(Resource):
    @api.expect(transcribe_input_model)
    @api.response(202, 'Accepted', transcribe_response_model)
    @api.response(400, 'Bad Request')
    def post(self):
        '''
        Submit an audio chunk for transcription.
        Transcription is asynchronous: Poll GET /transcripts (or the first/latest/<id>
        sub-resources) to retrieve results.
        '''
        try:
            return _transcribe_logic(success_status=202)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in POST /transcripts", exc_info=True)
            return {"error": str(e)}, 500

    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'from': _FROM_PARAM,
        'until': _UNTIL_PARAM,
    })
    @api.response(200, 'Success', list_transcripts_response_model)
    def get(self):
        '''List all transcripts for a tenant, filtered by the from/until chunk range.'''
        return jsonify(_list_transcripts_logic())


@api.route('/transcripts/count')
class TranscriptsCount(Resource):
    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'from': _FROM_PARAM,
        'until': _UNTIL_PARAM,
    })
    @api.response(200, 'Success', size_response_model)
    def get(self):
        '''Get the number of transcripts for a tenant (within the from/until range).'''
        return jsonify(_transcripts_size_logic())


@api.route('/transcripts/first')
class TranscriptsFirst(Resource):
    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'from': _FROM_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def get(self):
        '''Retrieve the first transcript for a tenant (non-destructive).'''
        return jsonify(_first_transcript_logic())

    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'from': _FROM_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def delete(self):
        '''Retrieve and remove (pop) the first transcript for a tenant.'''
        return jsonify(_pop_first_logic())


@api.route('/transcripts/latest')
class TranscriptsLatest(Resource):
    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'until': _UNTIL_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def get(self):
        '''Retrieve the latest transcript for a tenant (non-destructive).'''
        return jsonify(_latest_transcript_logic())

    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
        'until': _UNTIL_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def delete(self):
        '''Retrieve and remove (pop) the latest transcript for a tenant.'''
        return jsonify(_pop_latest_logic())


@api.route('/transcripts/<int:chunk_id>')
class TranscriptByID(Resource):
    # chunk_ids are millisecond timestamps, so the <int:...> converter both validates the path segment and guarantees it can never shadow the static /transcripts/first|latest|count routes above.
    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
        'sentences': _SENTENCES_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def get(self, chunk_id):
        '''Retrieve the transcript for a specific chunk_id.'''
        return jsonify(_get_transcript_logic(chunk_id))

    @api.doc(params={
        'tenant_id': _TENANT_PARAM,
        'source': _SOURCE_PARAM,
    })
    @api.response(200, 'Success', transcript_response_model)
    def delete(self, chunk_id):
        '''Delete the transcript for a specific chunk_id.'''
        return jsonify(_delete_transcript_logic(chunk_id))


# Deprecated RPC-style aliases.
# Kept for one release so existing clients (older grabbers, bookmarked curls,
# the HTML pages) don't break the moment this merges. Hidden from Swagger via
# doc=False to keep the published API surface clean. 
@api.route('/transcribe', doc=False)
class TranscribeLegacy(Resource):
    def post(self):
        '''DEPRECATED: use POST /transcripts.'''
        logger.warning("Deprecated POST /transcribe called; use POST /transcripts.")
        try:
            return _transcribe_logic(success_status=200)
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in /transcribe (deprecated)", exc_info=True)
            return {"error": str(e)}, 500


@api.route('/list_transcripts', doc=False)
class ListTranscriptsLegacy(Resource):
    def get(self):
        '''DEPRECATED: use GET /transcripts.'''
        return jsonify(_list_transcripts_logic())


@api.route('/transcripts_size', doc=False)
class TranscriptsSizeLegacy(Resource):
    def get(self):
        '''DEPRECATED: use GET /transcripts/count.'''
        return jsonify(_transcripts_size_logic())


@api.route('/get_transcript', doc=False)
class GetTranscriptLegacy(Resource):
    def get(self):
        '''DEPRECATED: use GET /transcripts/<chunk_id>.'''
        return jsonify(_get_transcript_logic(request.args.get('chunk_id')))


@api.route('/get_first_transcript', doc=False)
class GetFirstTranscriptLegacy(Resource):
    def get(self):
        '''DEPRECATED: use GET /transcripts/first.'''
        return jsonify(_first_transcript_logic())


@api.route('/pop_first_transcript', doc=False)
class PopFirstTranscriptLegacy(Resource):
    def delete(self):
        '''DEPRECATED: use DELETE /transcripts/first.'''
        return jsonify(_pop_first_logic())

    def get(self):
        '''DEPRECATED (and destructive): use DELETE /transcripts/first.'''
        logger.warning("Deprecated GET /pop_first_transcript called; use DELETE /transcripts/first.")
        return jsonify(_pop_first_logic())


@api.route('/get_latest_transcript', doc=False)
class GetLatestTranscriptLegacy(Resource):
    def get(self):
        '''DEPRECATED: use GET /transcripts/latest.'''
        return jsonify(_latest_transcript_logic())


@api.route('/pop_latest_transcript', doc=False)
class PopLatestTranscriptLegacy(Resource):
    def delete(self):
        '''DEPRECATED: use DELETE /transcripts/latest.'''
        return jsonify(_pop_latest_logic())

    def get(self):
        '''DEPRECATED (and destructive): use DELETE /transcripts/latest.'''
        logger.warning("Deprecated GET /pop_latest_transcript called; use DELETE /transcripts/latest.")
        return jsonify(_pop_latest_logic())


@api.route('/delete_transcript', doc=False)
class DeleteTranscriptLegacy(Resource):
    def delete(self):
        '''DEPRECATED: use DELETE /transcripts/<chunk_id>.'''
        return jsonify(_delete_transcript_logic(request.args.get('chunk_id')))

    def get(self):
        '''DEPRECATED (and destructive): use DELETE /transcripts/<chunk_id>.'''
        logger.warning("Deprecated GET /delete_transcript called; use DELETE /transcripts/<chunk_id>.")
        return jsonify(_delete_transcript_logic(request.args.get('chunk_id')))


_worker_thread = None
_worker_lock = threading.Lock()


def _start_worker_once():
    """Start the audio-worker thread exactly once per process. Idempotent."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _worker_thread = threading.Thread(
            target=process_audio,
            name="audio-worker",
            daemon=True,
        )
        _worker_thread.start()
        logger.info("Audio worker thread started")
        return _worker_thread


if _env_bool('TRANSCRIBE_AUTOSTART_WORKER', True):
    _start_worker_once()


if __name__ == '__main__':
    # Server bind config is env-driven so the defaults are SAFE:
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5040'))
    debug = _env_bool('FLASK_DEBUG', False)

    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning(
            "FLASK_DEBUG=true with host=%s exposes the Werkzeug debugger to "
            "the network. This is remote-code-execution. Set FLASK_HOST=127.0.0.1 "
            "or disable debug.",
            host,
        )

    # use_reloader=False because the audio-worker thread above must not be spawned twice (the reloader runs the module twice, which would otherwise create a duplicate consumer on the queue).
    app.run(host=host, port=port, debug=debug, use_reloader=False)
