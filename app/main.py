from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.api import api_router
from app.db import engine, Base
from app.core.config import settings

# Create database tables (only if database is available)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Could not create database tables: {e}")
    print("Database will be initialized when connection is available")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="Inference Server",
    description="A full-featured ML inference engine",
    version="1.0.0"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Inference Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Basic health check - API is running"""
    return {"status": "healthy"}


@app.get("/playground", response_class=HTMLResponse)
async def playground():
    return """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Inference Server Playground</title>
    <style>
      :root { color-scheme: light; }
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f6f7f9; color: #111; }
      header { padding: 16px 20px; background: #fff; border-bottom: 1px solid #e5e7eb; }
      header h1 { font-size: 18px; margin: 0; }
      main { max-width: 1100px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; grid-template-columns: 420px 1fr; }
      .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
      .card h2 { font-size: 14px; margin: 0 0 10px; }
      label { display: block; font-size: 12px; color: #374151; margin: 10px 0 6px; }
      select, input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; font-size: 14px; }
      textarea { min-height: 120px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; }
      .row { display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }
      .actions { display: flex; gap: 10px; margin-top: 12px; }
      button { border: 1px solid #d1d5db; background: #111827; color: #fff; border-radius: 8px; padding: 10px 12px; font-size: 14px; cursor: pointer; }
      button.secondary { background: #fff; color: #111827; }
      button:disabled { opacity: .5; cursor: not-allowed; }
      .muted { color: #6b7280; font-size: 12px; }
      .pill { display: inline-block; font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid #e5e7eb; background: #f9fafb; }
      pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12.5px; line-height: 1.35; }
      .kv { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      .error { color: #b91c1c; }
      .ok { color: #047857; }
      footer { padding: 16px 20px; color: #6b7280; font-size: 12px; }
      @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
    </style>
  </head>
  <body>
    <header>
      <h1>Inference Server Playground</h1>
      <div class=\"muted\">Select a model from the curated catalog, mount it, run a prediction, and view results.</div>
    </header>
    <main>
      <section class=\"card\">
        <h2>Model</h2>
        <div class=\"row\">
          <div>
            <label for=\"category\">Category</label>
            <select id=\"category\"></select>
          </div>
          <div>
            <label for=\"hardware\">Hardware</label>
            <select id=\"hardware\">
              <option value=\"\">Any</option>
              <option value=\"cpu\">CPU</option>
              <option value=\"gpu\">GPU</option>
            </select>
          </div>
        </div>

        <label for=\"model\">Curated model</label>
        <select id=\"model\"></select>

        <div id=\"modelMeta\" class=\"muted\" style=\"margin-top:10px\"></div>

        <label for=\"mountName\">Mount name (optional)</label>
        <input id=\"mountName\" placeholder=\"e.g. qwen3-playground\" />
        <div class=\"actions\">
          <button id=\"btnMount\">Mount model</button>
          <button id=\"btnRefresh\" class=\"secondary\">Refresh catalog</button>
        </div>
        <div id=\"mountStatus\" class=\"muted\" style=\"margin-top:10px\"></div>
      </section>


      <section class=\"card\">
        <h2>Catalog Admin</h2>
        <div class=\"muted\">Create, edit, or delete curated models (requires <code>CATALOG_ADMIN_TOKEN</code> on the server).</div>

        <label for=\"adminToken\">Admin token</label>
        <input id=\"adminToken\" placeholder=\"X-Catalog-Admin-Token\" />

        <label for=\"adminJson\">Selected model JSON (CatalogModel)</label>
        <textarea id=\"adminJson\">{\n  \"id\": \"\",\n  \"name\": \"\",\n  \"description\": \"\",\n  \"model_type\": \"text-generation\",\n  \"model_path\": \"\",\n  \"size\": \"small\",\n  \"vram_gb\": 0,\n  \"recommended_hardware\": \"cpu\",\n  \"tags\": [],\n  \"downloads\": null,\n  \"license\": null\n}</textarea>

        <div class=\"actions\">
          <button id=\"btnAdminSave\">Save (create/update)</button>
          <button id=\"btnAdminDelete\" class=\"secondary\">Delete</button>
        </div>
        <div id=\"adminStatus\" class=\"muted\" style=\"margin-top:10px\"></div>
      </section>

      <section class=\"card\">
        <h2>Run</h2>
        <div class=\"kv\">
          <span class=\"pill\">POST /v1/predictions</span>
          <span id=\"pillModelId\" class=\"pill\">model_id: (not mounted)</span>
          <span id=\"pillStatus\" class=\"pill\">status: idle</span>
        </div>

        <label for=\"inputJson\">Input JSON</label>
        <textarea id=\"inputJson\">{\n  \"prompt\": \"Hello world\"\n}</textarea>

        <div id=\"ttsPanel\" style=\"display:none; margin-top:10px\">
          <label for=\"refUpload\">Reference audio (optional)</label>
          <input id=\"refUpload\" type=\"file\" accept=\"audio/*\" />
          <div class=\"actions\" style=\"margin-top:10px\">
            <button id=\"btnUploadRef\" class=\"secondary\">Upload reference audio</button>
          </div>
          <div id=\"refStatus\" class=\"muted\" style=\"margin-top:10px\"></div>
        </div>

        <div class=\"actions\">
          <button id=\"btnRun\" disabled>Run prediction</button>
          <button id=\"btnCancel\" class=\"secondary\" disabled>Cancel</button>
        </div>

        <label>Response</label>
        <div id=\"audioOut\" class=\"card\" style=\"display:none; margin-bottom:10px\"></div>
        <div class=\"card\" style=\"background:#0b1020; color:#e5e7eb; border-color:#0b1020;\">
          <pre id=\"out\">(no output yet)</pre>
        </div>
        <div id=\"hint\" class=\"muted\" style=\"margin-top:10px\"></div>
      </section>
    </main>
    <footer>
      Tip: If predictions stay in \"starting\" or \"processing\", ensure Redis + Celery worker are running.
    </footer>

    <script>
      const SUPPORTED_TYPES = new Set([\"text-generation\", \"image-generation\", \"text-to-image\", \"text-to-speech\"]);

      const el = (id) => document.getElementById(id);

      let catalog = null;
      let mountedModelId = null;
      let lastPredictionId = null;
      let pollTimer = null;
      let selectedCatalogModel = null;

      function setStatus(text, kind) {
        el('pillStatus').textContent = `status: ${text}`;
        el('pillStatus').className = `pill ${kind || ''}`.trim();
      }

      function setOut(obj) {
        if (typeof obj === 'string') {
          el('out').textContent = obj;
          return;
        }
        el('out').textContent = JSON.stringify(obj, null, 2);
      }

      function setAudioOut(prediction) {
        const box = el('audioOut');
        box.style.display = 'none';
        box.innerHTML = '';

        const url = prediction && prediction.output && prediction.output.audio_url;
        if (!url) return;

        box.style.display = 'block';
        box.innerHTML = `<div class="muted" style="margin-bottom:8px">Audio output</div>` +
          `<audio controls src="${url}" style="width:100%"></audio>` +
          `<div class="muted" style="margin-top:8px">${url}</div>`;
      }

      async function api(path, options) {
        const res = await fetch(path, {
          headers: { 'Content-Type': 'application/json' },
          ...options,
        });
        const text = await res.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
        if (!res.ok) {
          const msg = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : `HTTP ${res.status}`);
          throw new Error(msg);
        }
        return data;
      }

      function deriveCategoriesFromModels(models) {
        const set = new Set();
        for (const m of (models || [])) {
          if (m && m.model_type) set.add(m.model_type);
        }
        return Array.from(set).sort();
      }

      function getFilteredModels() {
        if (!catalog) return [];
        const category = el('category').value;
        const hw = el('hardware').value;
        let models = catalog.models || [];
        if (category && category !== '__all__') models = models.filter(m => m.model_type === category);
        if (hw) models = models.filter(m => m.recommended_hardware === hw);
        models = models.filter(m => SUPPORTED_TYPES.has(m.model_type));
        return models;
      }

      function renderModelSelect() {
        const models = getFilteredModels();
        el('model').innerHTML = '';

        if (models.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'No supported models match filters';
          el('model').appendChild(opt);
          el('modelMeta').innerHTML = '<span class="error">No supported models available for current filters.</span>';
          return;
        }

        for (const m of models) {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = `${m.name} (${m.id})`;
          el('model').appendChild(opt);
        }
        renderModelMeta();
      }

      function renderModelMeta() {
        const models = getFilteredModels();
        const id = el('model').value;
        const m = models.find(x => x.id === id);
        selectedCatalogModel = m || null;
        if (!m) {
          el('modelMeta').textContent = '';
          return;
        }
        const parts = [];
        parts.push(`<div><strong>${m.name}</strong></div>`);
        parts.push(`<div class="muted">${m.description || ''}</div>`);
        parts.push(`<div class="kv" style="margin-top:8px">` +
          `<span class="pill">type: ${m.model_type}</span>` +
          `<span class="pill">hw: ${m.recommended_hardware}</span>` +
          (m.vram_gb ? `<span class="pill">vram: ${m.vram_gb}GB</span>` : '') +
          `<span class="pill">size: ${m.size}</span>` +
        `</div>`);
        parts.push(`<div class="muted" style="margin-top:6px">path: <code>${m.model_path}</code></div>`);
        el('modelMeta').innerHTML = parts.join('');

        // Toggle TTS helper panel
        el('ttsPanel').style.display = (m.model_type === 'text-to-speech') ? 'block' : 'none';
        el('refStatus').textContent = '';

        // Set a helpful default input template
        if (m.model_type === 'text-generation') {
          el('inputJson').value = `{
  "prompt": "Write a short paragraph about koalas."
}`;
        } else if (m.model_type === 'text-to-speech') {
          el('inputJson').value = `{
  "text": "Hello! This is a test of Chatterbox text-to-speech.",
  "reference_audio": null
}`;
        } else {
          el('inputJson').value = `{
  "prompt": "A cinematic photo of a lighthouse on a stormy sea"
}`;
        }

        // Populate admin JSON editor for selected model
        try {
          el('adminJson').value = JSON.stringify(m, null, 2);
        } catch (_) {
          // ignore
        }

        // If this is an Airaci Character TTS model, require reference_audio (master wav) before allowing Run
        const tags = Array.isArray(m.tags) ? m.tags : [];
        const isAiraciCharacter = tags.some(t => String(t || '').toLowerCase() === 'airaci-character');
        if (m.model_type === 'text-to-speech' && isAiraciCharacter) {
          el('refStatus').innerHTML = '<span class="error">Airaci Character: upload a master WAV reference clip before running.</span>';
          el('btnRun').disabled = true;
        }
      }

      async function loadCatalog() {
        setStatus('loading catalog...', '');
        el('mountStatus').textContent = '';
        try {
          catalog = await api('/v1/catalog/models');

          const rawCategories = (catalog && Array.isArray(catalog.categories) && catalog.categories.length > 0)
            ? catalog.categories
            : deriveCategoriesFromModels(catalog && catalog.models);
          const categories = ['__all__', ...rawCategories];
          el('category').innerHTML = '';
          for (const c of categories) {
            const opt = document.createElement('option');
            opt.value = c;
            opt.textContent = (c === '__all__') ? 'All categories' : c;
            el('category').appendChild(opt);
          }
          el('category').value = '__all__';
          renderModelSelect();

          setStatus('idle', '');
          if (!catalog || !Array.isArray(catalog.models)) {
            el('hint').innerHTML = '<span class="error">Catalog response did not include a models list. Check server logs.</span>';
          } else {
            el('hint').innerHTML = 'Supported in this build: <code>text-generation</code>, <code>text-to-image</code>, <code>text-to-speech</code>.';
          }
        } catch (e) {
          setStatus('error', 'error');
          el('hint').innerHTML = `<span class="error">Failed to load catalog: ${e.message}</span>`;
        }
      }

      async function adminSave() {
        el('adminStatus').textContent = '';
        const token = el('adminToken').value;
        if (!token) {
          el('adminStatus').innerHTML = '<span class="error">Admin token is required.</span>';
          return;
        }

        let body = null;
        try { body = JSON.parse(el('adminJson').value); } catch (_) {
          el('adminStatus').innerHTML = '<span class="error">Admin JSON is invalid.</span>';
          return;
        }

        if (!body || !body.id) {
          el('adminStatus').innerHTML = '<span class="error">CatalogModel.id is required.</span>';
          return;
        }

        try {
          const res = await fetch('/v1/catalog/admin/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'X-Catalog-Admin-Token': token },
            body: JSON.stringify(body),
          });
          const text = await res.text();
          let data = null;
          try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
          if (!res.ok) {
            const msg = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : `HTTP ${res.status}`);
            throw new Error(msg);
          }
          el('adminStatus').innerHTML = '<span class="ok">Saved.</span>';
          await loadCatalog();
          setOut(data);
        } catch (e) {
          el('adminStatus').innerHTML = `<span class="error">Save failed: ${e.message}</span>`;
        }
      }

      async function adminDelete() {
        el('adminStatus').textContent = '';
        const token = el('adminToken').value;
        if (!token) {
          el('adminStatus').innerHTML = '<span class="error">Admin token is required.</span>';
          return;
        }

        let body = null;
        try { body = JSON.parse(el('adminJson').value); } catch (_) {
          el('adminStatus').innerHTML = '<span class="error">Admin JSON is invalid.</span>';
          return;
        }

        const id = body && body.id;
        if (!id) {
          el('adminStatus').innerHTML = '<span class="error">CatalogModel.id is required.</span>';
          return;
        }

        try {
          const res = await fetch(`/v1/catalog/admin/models/${encodeURIComponent(id)}`, {
            method: 'DELETE',
            headers: { 'X-Catalog-Admin-Token': token },
          });
          const text = await res.text();
          let data = null;
          try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
          if (!res.ok) {
            const msg = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : `HTTP ${res.status}`);
            throw new Error(msg);
          }
          el('adminStatus').innerHTML = '<span class="ok">Deleted.</span>';
          await loadCatalog();
          setOut(data || { success: true });
        } catch (e) {
          el('adminStatus').innerHTML = `<span class="error">Delete failed: ${e.message}</span>`;
        }
      }

      async function uploadReferenceAudio() {
        el('refStatus').textContent = '';
        const fileInput = el('refUpload');
        if (!fileInput.files || fileInput.files.length === 0) {
          el('refStatus').innerHTML = '<span class="error">Pick an audio file first.</span>';
          return;
        }

        const f = fileInput.files[0];
        const fd = new FormData();
        fd.append('file', f);

        el('btnUploadRef').disabled = true;
        el('refStatus').textContent = 'Uploading...';
        try {
          const res = await fetch('/v1/files/upload', { method: 'POST', body: fd });
          const text = await res.text();
          let data = null;
          try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
          if (!res.ok) {
            const msg = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : `HTTP ${res.status}`);
            throw new Error(msg);
          }

          // Patch input JSON: set reference_audio to returned file_path
          let input = null;
          try { input = JSON.parse(el('inputJson').value); } catch (_) { input = {}; }
          input.reference_audio = data.file_path;
          el('inputJson').value = JSON.stringify(input, null, 2);

          el('refStatus').innerHTML = `<span class="ok">Uploaded:</span> <code>${data.file_path}</code>`;
          // If this was blocking an Airaci Character run, re-enable Run now that reference is set.
          if (selectedCatalogModel && selectedCatalogModel.model_type === 'text-to-speech') {
            const tags = Array.isArray(selectedCatalogModel.tags) ? selectedCatalogModel.tags : [];
            const isAiraciCharacter = tags.some(t => String(t || '').toLowerCase() === 'airaci-character');
            if (isAiraciCharacter) {
              el('btnRun').disabled = false;
            }
          }
          setOut(data);
        } catch (e) {
          el('refStatus').innerHTML = `<span class="error">Upload failed: ${e.message}</span>`;
        } finally {
          el('btnUploadRef').disabled = false;
        }
      }

      async function mountSelected() {
        const id = el('model').value;
        if (!id) return;
        el('btnMount').disabled = true;
        el('mountStatus').textContent = 'Mounting...';
        try {
          const body = {
            catalog_id: id,
            name: el('mountName').value ? el('mountName').value : null,
          };
          const res = await api('/v1/catalog/mount', { method: 'POST', body: JSON.stringify(body) });
          mountedModelId = res.model_id;
          el('pillModelId').textContent = `model_id: ${mountedModelId}`;
          el('mountStatus').innerHTML = `<span class="ok">${res.message}</span>`;
          el('btnRun').disabled = false;
          setOut(res);
        } catch (e) {
          el('mountStatus').innerHTML = `<span class="error">Mount failed: ${e.message}</span>`;
          setOut({ error: e.message });
        } finally {
          el('btnMount').disabled = false;
        }
      }

      async function runPrediction() {
        if (!mountedModelId) return;
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        lastPredictionId = null;
        el('btnRun').disabled = true;
        el('btnCancel').disabled = true;
        setStatus('submitting...', '');
        try {
          let input = null;
          try { input = JSON.parse(el('inputJson').value); } catch (_) {
            throw new Error('Input JSON is invalid');
          }

          const m = selectedCatalogModel;
          const tags = m && Array.isArray(m.tags) ? m.tags : [];
          const isAiraciCharacter = tags.some(t => String(t || '').toLowerCase() === 'airaci-character');
          if (m && m.model_type === 'text-to-speech' && isAiraciCharacter) {
            const ref = input && input.reference_audio;
            if (!ref || (typeof ref === 'string' && ref.trim() === '')) {
              el('refStatus').innerHTML = '<span class="error">Airaci Character requires a master WAV reference clip. Upload one to set <code>reference_audio</code>.</span>';
              throw new Error('reference_audio is required for Airaci Character TTS');
            }
          }

          const payload = { model_id: mountedModelId, input };
          const res = await api('/v1/predictions/', { method: 'POST', body: JSON.stringify(payload) });
          lastPredictionId = res.id;
          setOut(res);
          el('btnCancel').disabled = false;
          await pollOnce();
          pollTimer = setInterval(pollOnce, 1500);
        } catch (e) {
          setStatus('error', 'error');
          setOut({ error: e.message });
          el('btnRun').disabled = false;
        }
      }

      async function pollOnce() {
        if (!lastPredictionId) return;
        try {
          const res = await api(`/v1/predictions/${lastPredictionId}`);
          setStatus(res.status, res.status === 'failed' ? 'error' : (res.status === 'succeeded' ? 'ok' : ''));
          setOut(res);
          setAudioOut(res);
          if (['succeeded', 'failed', 'canceled'].includes(res.status)) {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
            el('btnRun').disabled = false;
            el('btnCancel').disabled = true;
          }
        } catch (e) {
          setStatus('error', 'error');
          setOut({ error: e.message });
          if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
          el('btnRun').disabled = false;
          el('btnCancel').disabled = true;
        }
      }

      async function cancelPrediction() {
        if (!lastPredictionId) return;
        el('btnCancel').disabled = true;
        try {
          const res = await api(`/v1/predictions/${lastPredictionId}/cancel`, { method: 'POST' });
          setOut(res);
          await pollOnce();
        } catch (e) {
          setOut({ error: e.message });
        }
      }

      el('btnRefresh').addEventListener('click', loadCatalog);
      el('btnMount').addEventListener('click', mountSelected);
      el('btnRun').addEventListener('click', runPrediction);
      el('btnCancel').addEventListener('click', cancelPrediction);
      el('btnAdminSave').addEventListener('click', adminSave);
      el('btnAdminDelete').addEventListener('click', adminDelete);
      el('btnUploadRef').addEventListener('click', uploadReferenceAudio);
      el('category').addEventListener('change', renderModelSelect);
      el('hardware').addEventListener('change', renderModelSelect);
      el('model').addEventListener('change', renderModelMeta);

      loadCatalog();
    </script>
  </body>
</html>"""


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check with service status"""
    from app.db import engine
    import redis

    status = {
        "api": "healthy",
        "database": "unknown",
        "redis": "unknown",
        "gpu": "unknown"
    }

    # Check database
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "healthy"
    except Exception as e:
        status["database"] = f"unavailable: {str(e)[:100]}"

    # Check Redis
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB
        )
        r.ping()
        status["redis"] = "healthy"
    except Exception as e:
        status["redis"] = f"unavailable: {str(e)[:100]}"

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            status["gpu"] = f"available: {gpu_count} GPU(s) - {gpu_name}"
        else:
            status["gpu"] = "unavailable: No CUDA devices found"
    except Exception as e:
        status["gpu"] = f"unavailable: {str(e)[:100]}"

    overall_status = "healthy" if status["database"] == "healthy" else "degraded"

    return {
        "status": overall_status,
        "services": status,
        "version": "1.0.0"
    }


# Include API routes
app.include_router(api_router, prefix="/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.WORKERS,
        reload=True
    )
