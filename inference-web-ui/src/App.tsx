import { useMemo, useState } from 'react'
import './App.css'

type NavKey =
  | 'dashboard'
  | 'api-keys'
  | 'models'
  | 'catalog'
  | 'predictions'
  | 'files'
  | 'system'

type ApiKeyItem = {
  id: string
  name: string
  prefix: string
  is_active: boolean
  is_admin: boolean
}

type ModelItem = {
  id: string
  name: string
  description?: string | null
  model_type: string
  version: string
  model_path: string
  input_schema: Record<string, unknown>
  hardware: string
  created_at: string
  updated_at: string
  owner_id?: string | null
}

type PredictionItem = {
  id: string
  status: string
  model_id: string
  input: Record<string, unknown>
  output?: Record<string, unknown> | null
  error?: string | null
  logs?: string | null
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  webhook?: string | null
}

type CatalogModel = {
  id: string
  name: string
  description: string
  model_type: string
  model_path: string
  size: string
  vram_gb?: number | null
  recommended_hardware: string
  tags?: string[]
  downloads?: string | null
  license?: string | null
}

type CatalogResponse = {
  categories: string[]
  total_models: number
  models: CatalogModel[]
}

const DEFAULT_BASE_URL = 'http://localhost:8000'

const readStorage = (key: string, fallback: string) => {
  const value = localStorage.getItem(key)
  return value === null ? fallback : value
}

const parseJson = <T,>(raw: string, fallback: T): T => {
  try {
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

const pretty = (value: unknown) => JSON.stringify(value, null, 2)

const navItems: { key: NavKey; label: string; hint: string }[] = [
  { key: 'dashboard', label: 'Dashboard', hint: 'Status + health' },
  { key: 'api-keys', label: 'API Keys', hint: 'Create and revoke' },
  { key: 'models', label: 'Models', hint: 'Mount and manage' },
  { key: 'catalog', label: 'Catalog', hint: 'Browse and admin' },
  { key: 'predictions', label: 'Predictions', hint: 'Run and observe' },
  { key: 'files', label: 'Files', hint: 'Upload artifacts' },
  { key: 'system', label: 'System', hint: 'Raw endpoints' },
]

function App() {
  const [active, setActive] = useState<NavKey>('dashboard')
  const [baseUrl, setBaseUrl] = useState(() =>
    readStorage('inference.baseUrl', DEFAULT_BASE_URL),
  )
  const [apiKey, setApiKey] = useState(() =>
    readStorage('inference.apiKey', ''),
  )
  const [catalogToken, setCatalogToken] = useState(() =>
    readStorage('inference.catalogToken', ''),
  )
  const [log, setLog] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [busy, setBusy] = useState(false)

  const headers = useMemo(() => {
    const out: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (apiKey) {
      out['x-api-key'] = apiKey
    }
    return out
  }, [apiKey])

  const api = async function apiRequest<T>(
    path: string,
    init?: RequestInit,
    options?: { allowRaw?: boolean; catalogAdmin?: boolean },
  ): Promise<T> {
    const url = `${baseUrl.replace(/\/$/, '')}${path}`
    const extraHeaders: Record<string, string> = {}
    if (options?.catalogAdmin && catalogToken) {
      extraHeaders['x-catalog-admin-token'] = catalogToken
    }
    const response = await fetch(url, {
      ...init,
      headers: {
        ...headers,
        ...(init?.headers ?? {}),
        ...extraHeaders,
      },
    })
    const text = await response.text()
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}: ${text}`)
    }
    if (options?.allowRaw) {
      return text as T
    }
    return (text ? JSON.parse(text) : {}) as T
  }

  const persist = () => {
    localStorage.setItem('inference.baseUrl', baseUrl)
    localStorage.setItem('inference.apiKey', apiKey)
    localStorage.setItem('inference.catalogToken', catalogToken)
  }

  const withBusy = async (fn: () => Promise<void>) => {
    setBusy(true)
    setError('')
    try {
      await fn()
    } catch (err) {
      const message =
        err instanceof Error ? err.message : `Unknown error: ${String(err)}`
      setError(message)
      setLog(message)
    } finally {
      setBusy(false)
    }
  }

  const [dashboard, setDashboard] = useState({
    health: null as null | Record<string, unknown>,
    detailed: null as null | Record<string, unknown>,
    system: null as null | Record<string, unknown>,
    db: null as null | Record<string, unknown>,
  })

  const [apiKeys, setApiKeys] = useState<ApiKeyItem[]>([])
  const [newKeyName, setNewKeyName] = useState('admin')
  const [newKeyAdmin, setNewKeyAdmin] = useState(true)
  const [newKeyResult, setNewKeyResult] = useState<string>('')

  const [models, setModels] = useState<ModelItem[]>([])
  const [modelDraft, setModelDraft] = useState(
    pretty({
      name: 'example-model',
      description: 'Example model',
      model_type: 'text-generation',
      version: '1.0.0',
      model_path: 'gpt2',
      input_schema: {
        prompt: { type: 'string', description: 'Prompt' },
        max_length: { type: 'integer', default: 64, minimum: 1, maximum: 512 },
      },
      hardware: 'cpu',
    }),
  )

  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)
  const [catalogCategory, setCatalogCategory] = useState('')
  const [catalogHardware, setCatalogHardware] = useState('')
  const [catalogSize, setCatalogSize] = useState('')
  const [catalogMountId, setCatalogMountId] = useState('')
  const [catalogAdminDraft, setCatalogAdminDraft] = useState(
    pretty({
      id: 'catalog-id',
      name: 'Model name',
      description: 'Description',
      model_type: 'text-generation',
      model_path: 'gpt2',
      size: 'small',
      vram_gb: 8,
      recommended_hardware: 'gpu',
      tags: ['demo'],
      downloads: null,
      license: null,
    }),
  )

  const [predictions, setPredictions] = useState<PredictionItem[]>([])
  const [predictionDraft, setPredictionDraft] = useState(
    pretty({
      model_id: '',
      input: { prompt: 'Hello world', max_length: 32 },
      webhook: null,
    }),
  )
  const [predictionDetail, setPredictionDetail] = useState<string>('')

  const [fileUploadResult, setFileUploadResult] = useState<string>('')
  const [filePath, setFilePath] = useState('')

  const loadDashboard = () =>
    withBusy(async () => {
      const health = await api<Record<string, unknown>>('/health')
      const detailed = await api<Record<string, unknown>>('/health/detailed')
      const system = await api<Record<string, unknown>>('/v1/system/status')
      const db = await api<Record<string, unknown>>('/v1/system/db-info')
      setDashboard({ health, detailed, system, db })
      setLog('Dashboard loaded')
    })

  const loadApiKeys = () =>
    withBusy(async () => {
      const list = await api<ApiKeyItem[]>('/v1/admin/api-keys')
      setApiKeys(list)
      setLog('API keys loaded')
    })

  const createApiKey = () =>
    withBusy(async () => {
      const body = { name: newKeyName, is_admin: newKeyAdmin }
      const result = await api<{
        id: string
        api_key: string
        prefix: string
        name: string
        is_admin: boolean
      }>('/v1/admin/api-keys', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setNewKeyResult(pretty(result))
      await loadApiKeys()
    })

  const revokeApiKey = (id: string) =>
    withBusy(async () => {
      await api(`/v1/admin/api-keys/${id}/revoke`, { method: 'POST' })
      await loadApiKeys()
    })

  const loadModels = () =>
    withBusy(async () => {
      const result = await api<{ models: ModelItem[] }>('/v1/models/')
      setModels(result.models)
      setLog('Models loaded')
    })

  const createModel = () =>
    withBusy(async () => {
      const body = parseJson(modelDraft, null)
      if (!body) {
        setLog('Model JSON is invalid')
        return
      }
      const result = await api<ModelItem>('/v1/models/', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setLog(`Model created: ${result.id}`)
      await loadModels()
    })

  const deleteModel = (id: string) =>
    withBusy(async () => {
      await api(`/v1/models/${id}`, { method: 'DELETE' })
      await loadModels()
    })

  const loadCatalog = () =>
    withBusy(async () => {
      const params = new URLSearchParams()
      if (catalogCategory) params.set('category', catalogCategory)
      if (catalogSize) params.set('size', catalogSize)
      if (catalogHardware) params.set('hardware', catalogHardware)
      const query = params.toString()
      const result = await api<CatalogResponse>(
        `/v1/catalog/models${query ? `?${query}` : ''}`,
      )
      setCatalog(result)
      setLog('Catalog loaded')
    })

  const mountCatalog = () =>
    withBusy(async () => {
      if (!catalogMountId.trim()) {
        setLog('Catalog ID is required')
        return
      }
      const body = { catalog_id: catalogMountId }
      const result = await api('/v1/catalog/mount', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setLog(pretty(result))
      await loadModels()
    })

  const upsertCatalogModel = () =>
    withBusy(async () => {
      const body = parseJson(catalogAdminDraft, null)
      if (!body) {
        setLog('Catalog JSON is invalid')
        return
      }
      const result = await api<CatalogModel>('/v1/catalog/admin/models', {
        method: 'POST',
        body: JSON.stringify(body),
      }, { catalogAdmin: true })
      setLog(`Catalog saved: ${result.id}`)
      await loadCatalog()
    })

  const deleteCatalogModel = (id: string) =>
    withBusy(async () => {
      await api(`/v1/catalog/admin/models/${id}`, { method: 'DELETE' }, { catalogAdmin: true })
      await loadCatalog()
    })

  const loadPredictions = () =>
    withBusy(async () => {
      const result = await api<{ predictions: PredictionItem[] }>(
        '/v1/predictions/',
      )
      setPredictions(result.predictions)
      setLog('Predictions loaded')
    })

  const createPrediction = () =>
    withBusy(async () => {
      const body = parseJson(predictionDraft, null)
      if (!body) {
        setLog('Prediction JSON is invalid')
        return
      }
      const result = await api<PredictionItem>('/v1/predictions/', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      setLog(`Prediction created: ${result.id}`)
      await loadPredictions()
    })

  const cancelPrediction = (id: string) =>
    withBusy(async () => {
      await api(`/v1/predictions/${id}/cancel`, { method: 'POST' })
      await loadPredictions()
    })

  const loadPrediction = () =>
    withBusy(async () => {
      if (!predictionDetail.trim()) {
        setLog('Prediction ID required')
        return
      }
      const result = await api<PredictionItem>(
        `/v1/predictions/${predictionDetail}`,
      )
      setLog(pretty(result))
    })

  const uploadFile = async (file: File) => {
    const url = `${baseUrl.replace(/\/$/, '')}/v1/files/upload`
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(url, {
      method: 'POST',
      headers: apiKey ? { 'x-api-key': apiKey } : undefined,
      body: form,
    })
    const text = await response.text()
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}: ${text}`)
    }
    return text
  }

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) =>
    withBusy(async () => {
      const file = event.target.files?.[0]
      if (!file) return
      const result = await uploadFile(file)
      setFileUploadResult(result)
      setLog('File uploaded')
    })

  const openFile = () => {
    if (!filePath.trim()) {
      setLog('File path is required')
      return
    }
    const url = `${baseUrl.replace(/\/$/, '')}/v1/files/${filePath}`
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const renderDashboard = () => (
    <section className="panel">
      <header>
        <h2>Dashboard</h2>
        <p>Status overview and health checks.</p>
      </header>
      <div className="actions">
        <button onClick={loadDashboard} disabled={busy}>
          Refresh
        </button>
      </div>
      <div className="grid two">
        <article>
          <h3>Health</h3>
          <pre>{dashboard.health ? pretty(dashboard.health) : 'No data'}</pre>
        </article>
        <article>
          <h3>Detailed</h3>
          <pre>
            {dashboard.detailed ? pretty(dashboard.detailed) : 'No data'}
          </pre>
        </article>
      </div>
      <article>
        <h3>System</h3>
        <pre>{dashboard.system ? pretty(dashboard.system) : 'No data'}</pre>
      </article>
      <article>
        <h3>Database</h3>
        <pre>{dashboard.db ? pretty(dashboard.db) : 'No data'}</pre>
      </article>
      {error && (
        <article>
          <h3>Last error</h3>
          <pre>{error}</pre>
        </article>
      )}
    </section>
  )

  const renderApiKeys = () => (
    <section className="panel">
      <header>
        <h2>API Keys</h2>
        <p>Create or revoke API keys. Store new keys securely.</p>
      </header>
      <div className="actions">
        <button onClick={loadApiKeys} disabled={busy}>
          Refresh
        </button>
      </div>
      <div className="grid two">
        <article>
          <h3>Create key</h3>
          <label>
            Name
            <input
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
            />
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={newKeyAdmin}
              onChange={(e) => setNewKeyAdmin(e.target.checked)}
            />
            Admin privileges
          </label>
          <button onClick={createApiKey} disabled={busy}>
            Create API key
          </button>
          <pre>{newKeyResult || 'New key response appears here.'}</pre>
        </article>
        <article>
          <h3>Active keys</h3>
          <div className="table">
            {apiKeys.length === 0 && <p>No keys yet.</p>}
            {apiKeys.map((key) => (
              <div key={key.id} className="row">
                <div>
                  <strong>{key.name}</strong>
                  <span className="muted">({key.prefix})</span>
                </div>
                <div className="row-meta">
                  <span>{key.is_admin ? 'admin' : 'standard'}</span>
                  <span>{key.is_active ? 'active' : 'revoked'}</span>
                </div>
                <button
                  onClick={() => revokeApiKey(key.id)}
                  disabled={busy || !key.is_active}
                >
                  Revoke
                </button>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )

  const renderModels = () => (
    <section className="panel">
      <header>
        <h2>Models</h2>
        <p>Manage mounted models.</p>
      </header>
      <div className="actions">
        <button onClick={loadModels} disabled={busy}>
          Refresh
        </button>
      </div>
      <div className="grid two">
        <article>
          <h3>Create model</h3>
          <textarea
            value={modelDraft}
            onChange={(e) => setModelDraft(e.target.value)}
          />
          <button onClick={createModel} disabled={busy}>
            Create model
          </button>
        </article>
        <article>
          <h3>Mounted models</h3>
          <div className="table">
            {models.length === 0 && <p>No models yet.</p>}
            {models.map((model) => (
              <div key={model.id} className="row">
                <div>
                  <strong>{model.name}</strong>
                  <span className="muted">{model.model_type}</span>
                </div>
                <div className="row-meta">
                  <span>{model.hardware}</span>
                  <span>{model.version}</span>
                </div>
                <button onClick={() => deleteModel(model.id)} disabled={busy}>
                  Delete
                </button>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )

  const renderCatalog = () => (
    <section className="panel">
      <header>
        <h2>Catalog</h2>
        <p>Browse available models and mount them.</p>
      </header>
      <div className="actions">
        <button onClick={loadCatalog} disabled={busy}>
          Refresh
        </button>
      </div>
      <div className="filters">
        <input
          placeholder="Category"
          value={catalogCategory}
          onChange={(e) => setCatalogCategory(e.target.value)}
        />
        <input
          placeholder="Size"
          value={catalogSize}
          onChange={(e) => setCatalogSize(e.target.value)}
        />
        <input
          placeholder="Hardware"
          value={catalogHardware}
          onChange={(e) => setCatalogHardware(e.target.value)}
        />
      </div>
      <div className="grid two">
        <article>
          <h3>Mount from catalog</h3>
          <input
            placeholder="Catalog ID"
            value={catalogMountId}
            onChange={(e) => setCatalogMountId(e.target.value)}
          />
          <button onClick={mountCatalog} disabled={busy}>
            Mount model
          </button>
          <pre>{catalog ? `Total: ${catalog.total_models}` : 'No data'}</pre>
        </article>
        <article>
          <h3>Admin: add/update catalog</h3>
          <textarea
            value={catalogAdminDraft}
            onChange={(e) => setCatalogAdminDraft(e.target.value)}
          />
          <button onClick={upsertCatalogModel} disabled={busy}>
            Save catalog model
          </button>
        </article>
      </div>
      <article>
        <h3>Catalog models</h3>
        <div className="table">
          {!catalog?.models?.length && <p>No catalog results.</p>}
          {catalog?.models?.map((item) => (
            <div key={item.id} className="row">
              <div>
                <strong>{item.name}</strong>
                <span className="muted">{item.model_type}</span>
              </div>
              <div className="row-meta">
                <span>{item.size}</span>
                <span>{item.recommended_hardware}</span>
              </div>
              <button onClick={() => deleteCatalogModel(item.id)} disabled={busy}>
                Delete
              </button>
            </div>
          ))}
        </div>
      </article>
    </section>
  )

  const renderPredictions = () => (
    <section className="panel">
      <header>
        <h2>Predictions</h2>
        <p>Run and monitor predictions.</p>
      </header>
      <div className="actions">
        <button onClick={loadPredictions} disabled={busy}>
          Refresh
        </button>
      </div>
      <div className="grid two">
        <article>
          <h3>Create prediction</h3>
          <textarea
            value={predictionDraft}
            onChange={(e) => setPredictionDraft(e.target.value)}
          />
          <button onClick={createPrediction} disabled={busy}>
            Create prediction
          </button>
          <div className="row">
            <input
              placeholder="Prediction ID"
              value={predictionDetail}
              onChange={(e) => setPredictionDetail(e.target.value)}
            />
            <button onClick={loadPrediction} disabled={busy}>
              Load
            </button>
          </div>
        </article>
        <article>
          <h3>Recent predictions</h3>
          <div className="table">
            {predictions.length === 0 && <p>No predictions yet.</p>}
            {predictions.map((prediction) => (
              <div key={prediction.id} className="row">
                <div>
                  <strong>{prediction.id}</strong>
                  <span className="muted">{prediction.model_id}</span>
                </div>
                <div className="row-meta">
                  <span>{prediction.status}</span>
                  <span>{prediction.created_at}</span>
                </div>
                <button
                  onClick={() => cancelPrediction(prediction.id)}
                  disabled={busy}
                >
                  Cancel
                </button>
              </div>
            ))}
          </div>
        </article>
      </div>
    </section>
  )

  const renderFiles = () => (
    <section className="panel">
      <header>
        <h2>Files</h2>
        <p>Upload files and retrieve by path.</p>
      </header>
      <div className="grid two">
        <article>
          <h3>Upload</h3>
          <input type="file" onChange={handleUpload} />
          <pre>{fileUploadResult || 'Upload response appears here.'}</pre>
        </article>
        <article>
          <h3>Open file by path</h3>
          <input
            placeholder="path/to/file"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
          />
          <button onClick={openFile} disabled={busy}>
            Open file
          </button>
        </article>
      </div>
    </section>
  )

  const renderSystem = () => (
    <section className="panel">
      <header>
        <h2>System</h2>
        <p>Raw endpoint calls and quick checks.</p>
      </header>
      <div className="actions">
        <button onClick={loadDashboard} disabled={busy}>
          Refresh system + health
        </button>
      </div>
      <article>
        <h3>System status</h3>
        <pre>{dashboard.system ? pretty(dashboard.system) : 'No data'}</pre>
      </article>
      <article>
        <h3>Logs</h3>
        <pre>{log || 'No logs yet.'}</pre>
      </article>
      {error && (
        <article>
          <h3>Last error</h3>
          <pre>{error}</pre>
        </article>
      )}
    </section>
  )

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">∞</span>
          <div>
            <h1>Inference Admin</h1>
            <p>Single control plane for your server.</p>
          </div>
        </div>
        <div className="nav">
          {navItems.map((item) => (
            <button
              key={item.key}
              className={active === item.key ? 'active' : ''}
              onClick={() => setActive(item.key)}
            >
              <span>{item.label}</span>
              <small>{item.hint}</small>
            </button>
          ))}
        </div>
      </aside>
      <main>
        <section className="panel config">
          <header>
            <h2>Runtime Configuration</h2>
            <p>Set your server URL and API keys. Stored locally in this browser.</p>
          </header>
          <div className="config-grid">
            <label>
              API base URL
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={DEFAULT_BASE_URL}
              />
            </label>
            <label>
              API key (x-api-key)
              <input
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Leave empty for public endpoints"
              />
            </label>
            <label>
              Catalog admin token
              <input
                value={catalogToken}
                onChange={(e) => setCatalogToken(e.target.value)}
                placeholder="Required for catalog admin APIs"
              />
            </label>
            <button onClick={persist} disabled={busy}>
              Save settings
            </button>
          </div>
        </section>
        {active === 'dashboard' && renderDashboard()}
        {active === 'api-keys' && renderApiKeys()}
        {active === 'models' && renderModels()}
        {active === 'catalog' && renderCatalog()}
        {active === 'predictions' && renderPredictions()}
        {active === 'files' && renderFiles()}
        {active === 'system' && renderSystem()}
      </main>
    </div>
  )
}

export default App
