
import { useMemo, useRef, useState } from 'react'
import './App.css'

type NavKey =
  | 'dashboard'
  | 'catalog'
  | 'models'
  | 'predictions'
  | 'files'
  | 'api-keys'
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
  input_schema?: Record<string, unknown>
  source?: string | null
  source_url?: string | null
  schema_source?: string | null
  latest_update?: string | null
}

type CatalogResponse = {
  categories: string[]
  total_models: number
  models: CatalogModel[]
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

type ModelSchemaField = {
  type: string
  description?: string | null
  default?: unknown
  minimum?: number | null
  maximum?: number | null
  enum?: unknown[] | null
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

const CACHE_TTL = {
  apiKeys: 30_000,
  models: 15_000,
  catalog: 60_000,
  predictions: 15_000,
}

const pretty = (value: unknown) => JSON.stringify(value, null, 2)

const toSchemaField = (raw: unknown): ModelSchemaField | null => {
  if (!raw || typeof raw !== 'object') return null
  const field = raw as ModelSchemaField
  if (!field.type) return null
  return field
}

const coerceValue = (type: string, value: string) => {
  if (type === 'integer') {
    const parsed = Number.parseInt(value, 10)
    return Number.isNaN(parsed) ? value : parsed
  }
  if (type === 'number') {
    const parsed = Number.parseFloat(value)
    return Number.isNaN(parsed) ? value : parsed
  }
  if (type === 'boolean') {
    return value === 'true'
  }
  return value
}

const shorten = (value: string, length = 6) =>
  value.length <= length ? value : `${value.slice(0, length)}…`

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
  const cacheRef = useRef(new Map<string, { ts: number; value: unknown }>())

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
    options?: { catalogAdmin?: boolean },
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
    return (text ? JSON.parse(text) : {}) as T
  }

  const cacheKey = (key: string) => `inference.cache.${key}.${baseUrl}`

  const readCache = <T,>(key: string, maxAgeMs: number): T | null => {
    const now = Date.now()
    const entry = cacheRef.current.get(key)
    if (entry && now - entry.ts <= maxAgeMs) {
      return entry.value as T
    }
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const parsed = parseJson<{ ts: number; value: T } | null>(raw, null)
    if (!parsed) return null
    if (now - parsed.ts > maxAgeMs) return null
    cacheRef.current.set(key, { ts: parsed.ts, value: parsed.value })
    return parsed.value
  }

  const writeCache = <T,>(key: string, value: T) => {
    const entry = { ts: Date.now(), value }
    cacheRef.current.set(key, entry)
    localStorage.setItem(key, JSON.stringify(entry))
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
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)
  const [catalogCategory, setCatalogCategory] = useState('')
  const [catalogHardware, setCatalogHardware] = useState('')
  const [catalogSize, setCatalogSize] = useState('')
  const [catalogSort, setCatalogSort] = useState<'latest' | 'name' | 'size'>(
    'latest',
  )
  const [catalogMountId, setCatalogMountId] = useState('')
  const [selectedCatalogId, setSelectedCatalogId] = useState('')
  const [reconStatus, setReconStatus] = useState<Record<string, unknown> | null>(
    null,
  )
  const [reconSources, setReconSources] = useState('huggingface,replicate')

  const sortedCatalogModels = useMemo(() => {
    const list = catalog?.models ? [...catalog.models] : []
    if (!list.length) return list
    if (catalogSort === 'name') {
      return list.sort((a, b) => a.name.localeCompare(b.name))
    }
    if (catalogSort === 'size') {
      const order: Record<string, number> = {
        tiny: 0,
        small: 1,
        medium: 2,
        large: 3,
        xl: 4,
      }
      return list.sort(
        (a, b) =>
          (order[a.size] ?? 99) - (order[b.size] ?? 99) ||
          a.name.localeCompare(b.name),
      )
    }
    return list.sort((a, b) => {
      const aTime = a.latest_update ? Date.parse(a.latest_update) : 0
      const bTime = b.latest_update ? Date.parse(b.latest_update) : 0
      return bTime - aTime
    })
  }, [catalog, catalogSort])

  const selectedCatalogModel = useMemo(
    () => catalog?.models?.find((item) => item.id === selectedCatalogId) || null,
    [catalog, selectedCatalogId],
  )

  const [selectedModelId, setSelectedModelId] = useState('')
  const [predictionInput, setPredictionInput] = useState<Record<string, unknown>>(
    { prompt: 'Hello world', max_length: 32 },
  )
  const [predictionWebhook, setPredictionWebhook] = useState('')
  const [predictionDraft, setPredictionDraft] = useState(
    pretty({
      model_id: '',
      input: { prompt: 'Hello world', max_length: 32 },
      webhook: null,
    }),
  )
  const [predictions, setPredictions] = useState<PredictionItem[]>([])
  const [predictionResult, setPredictionResult] = useState<string>('')
  const [predictionDetail, setPredictionDetail] = useState<string>('')

  const [fileUploadResult, setFileUploadResult] = useState<string>('')
  const [filePath, setFilePath] = useState('')

  const selectedModel = models.find((m) => m.id === selectedModelId) || null

  const loadDashboard = () =>
    withBusy(async () => {
      const health = await api<Record<string, unknown>>('/health')
      const detailed = await api<Record<string, unknown>>('/health/detailed')
      const system = await api<Record<string, unknown>>('/v1/system/status')
      let db: Record<string, unknown> | null = null
      try {
        db = await api<Record<string, unknown>>('/v1/system/db-info')
      } catch {
        db = null
      }
      setDashboard({ health, detailed, system, db })
      setLog('Dashboard loaded')
    })

  const loadApiKeys = (force = false) =>
    withBusy(async () => {
      const key = cacheKey('api-keys')
      if (!force) {
        const cached = readCache<ApiKeyItem[]>(key, CACHE_TTL.apiKeys)
        if (cached) {
          setApiKeys(cached)
          setLog('API keys loaded (cached)')
          return
        }
      }
      const list = await api<ApiKeyItem[]>('/v1/admin/api-keys')
      setApiKeys(list)
      writeCache(key, list)
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
      await loadApiKeys(true)
    })

  const revokeApiKey = (id: string) =>
    withBusy(async () => {
      await api(`/v1/admin/api-keys/${id}/revoke`, { method: 'POST' })
      await loadApiKeys(true)
    })

  const loadModels = (force = false) =>
    withBusy(async () => {
      const key = cacheKey('models')
      if (!force) {
        const cached = readCache<{ models: ModelItem[] }>(key, CACHE_TTL.models)
        if (cached) {
          setModels(cached.models)
          setLog('Models refreshed (cached)')
          return
        }
      }
      const result = await api<{ models: ModelItem[] }>('/v1/models/')
      setModels(result.models)
      writeCache(key, result)
      if (!selectedModelId && result.models.length > 0) {
        setSelectedModelId(result.models[0].id)
        const schema = result.models[0].input_schema || {}
        const nextInput: Record<string, unknown> = {}
        Object.entries(schema).forEach(([key, raw]) => {
          const field = toSchemaField(raw)
          if (!field) return
          if (field.default !== undefined && field.default !== null) {
            nextInput[key] = field.default
          } else if (field.type === 'boolean') {
            nextInput[key] = false
          } else if (field.type === 'integer' || field.type === 'number') {
            nextInput[key] = field.minimum ?? 0
          } else if (field.type === 'array') {
            nextInput[key] = []
          } else {
            nextInput[key] = ''
          }
        })
        if (Object.keys(nextInput).length > 0) {
          setPredictionInput(nextInput)
          setPredictionDraft(
            pretty({
              model_id: result.models[0].id,
              input: nextInput,
              webhook: null,
            }),
          )
        }
      }
      setLog('Models refreshed')
    })

  const deleteModel = (id: string) =>
    withBusy(async () => {
      await api(`/v1/models/${id}/unmount`, { method: 'POST' })
      await loadModels(true)
    })

  const loadCatalog = (force = false) =>
    withBusy(async () => {
      const params = new URLSearchParams()
      if (catalogCategory) params.set('category', catalogCategory)
      if (catalogSize) params.set('size', catalogSize)
      if (catalogHardware) params.set('hardware', catalogHardware)
      const query = params.toString()
      const key = cacheKey(
        `catalog.${catalogCategory}.${catalogSize}.${catalogHardware}`,
      )
      if (!force) {
        const cached = readCache<CatalogResponse>(key, CACHE_TTL.catalog)
        if (cached) {
          setCatalog(cached)
          setLog('Catalog refreshed (cached)')
          return
        }
      }
      const result = await api<CatalogResponse>(
        `/v1/catalog/models${query ? `?${query}` : ''}`,
      )
      setCatalog(result)
      if (!selectedCatalogId && result.models.length > 0) {
        setSelectedCatalogId(result.models[0].id)
      }
      writeCache(key, result)
      setLog('Catalog refreshed')
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
      await loadModels(true)
    })

  const refreshCatalogSchema = (catalogId: string) =>
    withBusy(async () => {
      if (!catalogToken) {
        setLog('Catalog admin token required for schema refresh')
        return
      }
      const result = await api<CatalogModel>(
        `/v1/catalog/models/${catalogId}/schema/refresh`,
        { method: 'POST' },
        { catalogAdmin: true },
      )
      setCatalog((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          models: prev.models.map((item) =>
            item.id === catalogId ? result : item,
          ),
        }
      })
      setLog(`Schema refreshed for ${catalogId}`)
    })

  const handleSelectCatalog = (catalogId: string) => {
    setCatalogMountId(catalogId)
    setSelectedCatalogId(catalogId)
    const model = catalog?.models?.find((item) => item.id === catalogId)
    if (
      model &&
      (!model.input_schema || Object.keys(model.input_schema).length === 0) &&
      catalogToken
    ) {
      void refreshCatalogSchema(catalogId)
    }
  }

  const loadReconStatus = () =>
    withBusy(async () => {
      const status = await api<Record<string, unknown>>(
        '/v1/catalog/recon/status',
        undefined,
        { catalogAdmin: true },
      )
      setReconStatus(status)
      setLog('Recon status loaded')
    })

  const runRecon = () =>
    withBusy(async () => {
      const query = reconSources ? `?sources=${encodeURIComponent(reconSources)}` : ''
      const status = await api<Record<string, unknown>>(
        `/v1/catalog/recon${query}`,
        { method: 'POST' },
        { catalogAdmin: true },
      )
      setReconStatus(status)
      setLog('Recon triggered')
      await loadCatalog(true)
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
      setPredictionResult(pretty(result))
      setLog(`Prediction created: ${result.id}`)
    })

  const loadPredictions = (force = false) =>
    withBusy(async () => {
      const key = cacheKey('predictions')
      if (!force) {
        const cached = readCache<{ predictions: PredictionItem[] }>(
          key,
          CACHE_TTL.predictions,
        )
        if (cached) {
          setPredictions(cached.predictions)
          setLog('Predictions loaded (cached)')
          return
        }
      }
      const result = await api<{ predictions: PredictionItem[] }>(
        '/v1/predictions/',
      )
      setPredictions(result.predictions)
      writeCache(key, result)
      setLog('Predictions loaded')
    })

  const cancelPrediction = (id: string) =>
    withBusy(async () => {
      await api(`/v1/predictions/${id}/cancel`, { method: 'POST' })
      await loadPredictions(true)
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
      setPredictionResult(pretty(result))
      setLog(`Prediction loaded: ${result.id}`)
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

  return (
    <div className="admin-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">∞</span>
          <div>
            <h1>Inference Admin</h1>
            <p>Modern control plane</p>
          </div>
        </div>
        <nav className="nav">
          {[
            ['dashboard', 'Dashboard'],
            ['catalog', 'Catalog'],
            ['models', 'Models'],
            ['predictions', 'Predictions'],
            ['files', 'Files'],
            ['api-keys', 'API Keys'],
            ['system', 'System'],
          ].map(([key, label]) => (
            <button
              key={key}
              className={active === key ? 'active' : ''}
              onClick={() => setActive(key as NavKey)}
            >
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Admin Console</p>
            <h1>{active.replace('-', ' ')}</h1>
          </div>
          <div className="topbar-actions">
            <button onClick={() => loadCatalog(true)} disabled={busy}>
              Refresh catalog
            </button>
            <button onClick={() => loadModels(true)} disabled={busy}>
              Refresh models
            </button>
          </div>
        </header>

        <section className="panel config">
          <div>
            <h2>Connection</h2>
            <p>Set server URL + admin credentials.</p>
          </div>
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
                placeholder="Paste master key"
              />
            </label>
            <label>
              Catalog admin token
              <input
                value={catalogToken}
                onChange={(e) => setCatalogToken(e.target.value)}
                placeholder="Required for recon/admin"
              />
            </label>
            <button onClick={persist} disabled={busy}>
              Save
            </button>
          </div>
        </section>
        {active === 'dashboard' && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Health</h2>
                  <p>API status.</p>
                </div>
                <button onClick={loadDashboard} disabled={busy}>
                  Refresh
                </button>
              </div>
              <pre>{dashboard.health ? pretty(dashboard.health) : 'No data'}</pre>
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Detailed</h2>
                  <p>Database, Redis, GPU.</p>
                </div>
              </div>
              <pre>{dashboard.detailed ? pretty(dashboard.detailed) : 'No data'}</pre>
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>System</h2>
                </div>
              </div>
              <pre>{dashboard.system ? pretty(dashboard.system) : 'No data'}</pre>
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Database</h2>
                </div>
              </div>
              <pre>{dashboard.db ? pretty(dashboard.db) : 'No data'}</pre>
            </div>
          </section>
        )}
        {active === 'catalog' && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Catalog</h2>
                  <p>Browse and mount models.</p>
                </div>
                <button onClick={() => loadCatalog(true)} disabled={busy}>
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
                <select
                  value={catalogSort}
                  onChange={(e) =>
                    setCatalogSort(e.target.value as 'latest' | 'name' | 'size')
                  }
                >
                  <option value="latest">Sort: latest update</option>
                  <option value="name">Sort: name</option>
                  <option value="size">Sort: size</option>
                </select>
              </div>
              <div className="recon-bar">
                <input
                  placeholder="Recon sources (huggingface,replicate)"
                  value={reconSources}
                  onChange={(e) => setReconSources(e.target.value)}
                />
                <button onClick={runRecon} disabled={busy}>
                  Run recon
                </button>
                <button onClick={loadReconStatus} disabled={busy}>
                  Status
                </button>
              </div>
              {reconStatus && <pre>{pretty(reconStatus)}</pre>}
              <div className="mount">
                <input
                  placeholder="Catalog ID"
                  value={catalogMountId}
                  onChange={(e) => setCatalogMountId(e.target.value)}
                />
                <button onClick={mountCatalog} disabled={busy}>
                  Mount
                </button>
              </div>
              <div className="list">
                {!catalog?.models?.length && (
                  <p className="muted">No catalog data.</p>
                )}
                {sortedCatalogModels.map((item) => (
                  <button
                    key={item.id}
                    className="list-item"
                    onClick={() => handleSelectCatalog(item.id)}
                  >
                    <div>
                      <strong>{item.name}</strong>
                      <span>{item.model_type}</span>
                    </div>
                    <span className="pill">{item.size}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Selected schema</h2>
                  <p>Server-curated input schema.</p>
                </div>
                <button
                  onClick={() =>
                    selectedCatalogId && refreshCatalogSchema(selectedCatalogId)
                  }
                  disabled={busy || !selectedCatalogId}
                >
                  Refresh schema
                </button>
              </div>
              <pre>
                {selectedCatalogModel
                  ? pretty(selectedCatalogModel.input_schema || {})
                  : 'Select a model'}
              </pre>
            </div>
          </section>
        )}
        {active === 'models' && (
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Mounted models</h2>
                <p>Unload with one click.</p>
              </div>
              <button onClick={() => loadModels(true)} disabled={busy}>
                Refresh
              </button>
            </div>
            <div className="list">
              {!models.length && <p className="muted">No models mounted.</p>}
              {models.map((model) => (
                <div key={model.id} className="list-row">
                  <div>
                    <strong>{model.name}</strong>
                    <span>{model.model_type}</span>
                  </div>
                  <div className="row-actions">
                    <span className="pill">{model.hardware}</span>
                    <button onClick={() => deleteModel(model.id)} disabled={busy}>
                      Unmount
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
        {active === 'predictions' && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Run prediction</h2>
                  <p>Adapted to model schema.</p>
                </div>
              </div>
              <div className="prediction-select">
                <label>
                  Model
                  <select
                    value={selectedModelId}
                    onChange={(e) => {
                      const nextId = e.target.value
                      setSelectedModelId(nextId)
                      const nextModel = models.find((m) => m.id === nextId)
                      if (!nextModel) return
                      const schema = nextModel.input_schema || {}
                      const nextInput: Record<string, unknown> = {}
                      Object.entries(schema).forEach(([key, raw]) => {
                        const field = toSchemaField(raw)
                        if (!field) return
                        if (field.default !== undefined && field.default !== null) {
                          nextInput[key] = field.default
                        } else if (field.type === 'boolean') {
                          nextInput[key] = false
                        } else if (
                          field.type === 'integer' ||
                          field.type === 'number'
                        ) {
                          nextInput[key] = field.minimum ?? 0
                        } else if (field.type === 'array') {
                          nextInput[key] = []
                        } else {
                          nextInput[key] = ''
                        }
                      })
                      setPredictionInput(nextInput)
                      setPredictionDraft(
                        pretty({
                          model_id: nextId,
                          input: nextInput,
                          webhook: predictionWebhook ? predictionWebhook : null,
                        }),
                      )
                    }}
                  >
                    <option value="">Select model</option>
                    {models.map((model) => (
                      <option key={model.id} value={model.id}>
                        {model.name} ({shorten(model.id)})
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="form-grid">
                {(() => {
                  if (!selectedModel) {
                    return <p className="muted">Select a model to load inputs.</p>
                  }
                  const schema = selectedModel.input_schema || {}
                  const entries = Object.entries(schema)
                  if (!entries.length) {
                    return <p className="muted">No input schema defined.</p>
                  }
                  return entries.map(([key, raw]) => {
                    const field = toSchemaField(raw)
                    if (!field) return null
                    const value = predictionInput[key]
                    const description = field.description || ''
                    if (field.enum && field.enum.length > 0) {
                      return (
                        <label key={key}>
                          {key}
                          <select
                            value={String(value ?? '')}
                            onChange={(e) => {
                              const next = { ...predictionInput, [key]: e.target.value }
                              setPredictionInput(next)
                              setPredictionDraft(
                                pretty({
                                  model_id: selectedModelId,
                                  input: next,
                                  webhook: predictionWebhook ? predictionWebhook : null,
                                }),
                              )
                            }}
                          >
                            <option value="">Select</option>
                            {field.enum.map((option) => (
                              <option key={String(option)} value={String(option)}>
                                {String(option)}
                              </option>
                            ))}
                          </select>
                          {description && <span className="hint">{description}</span>}
                        </label>
                      )
                    }
                    if (field.type === 'boolean') {
                      return (
                        <label key={key} className="toggle">
                          <input
                            type="checkbox"
                            checked={Boolean(value)}
                            onChange={(e) => {
                              const next = {
                                ...predictionInput,
                                [key]: e.target.checked,
                              }
                              setPredictionInput(next)
                              setPredictionDraft(
                                pretty({
                                  model_id: selectedModelId,
                                  input: next,
                                  webhook: predictionWebhook ? predictionWebhook : null,
                                }),
                              )
                            }}
                          />
                          {key}
                          {description && <span className="hint">{description}</span>}
                        </label>
                      )
                    }
                    return (
                      <label key={key}>
                        {key}
                        <input
                          value={
                            value === undefined || value === null ? '' : String(value)
                          }
                          onChange={(e) => {
                            const nextValue = coerceValue(field.type, e.target.value)
                            const next = { ...predictionInput, [key]: nextValue }
                            setPredictionInput(next)
                            setPredictionDraft(
                              pretty({
                                model_id: selectedModelId,
                                input: next,
                                webhook: predictionWebhook ? predictionWebhook : null,
                              }),
                            )
                          }}
                        />
                        {description && <span className="hint">{description}</span>}
                      </label>
                    )
                  })
                })()}
              </div>
              <label>
                Webhook (optional)
                <input
                  value={predictionWebhook}
                  onChange={(e) => {
                    const nextWebhook = e.target.value
                    setPredictionWebhook(nextWebhook)
                    setPredictionDraft(
                      pretty({
                        model_id: selectedModelId,
                        input: predictionInput,
                        webhook: nextWebhook ? nextWebhook : null,
                      }),
                    )
                  }}
                />
              </label>
              <label>
                Raw JSON
                <textarea
                  value={predictionDraft}
                  onChange={(e) => setPredictionDraft(e.target.value)}
                />
              </label>
              <div className="row-actions">
                <button onClick={createPrediction} disabled={busy}>
                  Run
                </button>
                <input
                  placeholder="Prediction ID"
                  value={predictionDetail}
                  onChange={(e) => setPredictionDetail(e.target.value)}
                />
                <button onClick={loadPrediction} disabled={busy}>
                  Load
                </button>
              </div>
              <pre>{predictionResult || 'Results appear here.'}</pre>
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Recent predictions</h2>
                  <p>Status + cancel.</p>
                </div>
                <button onClick={() => loadPredictions(true)} disabled={busy}>
                  Refresh
                </button>
              </div>
              <div className="list">
                {!predictions.length && <p className="muted">No predictions yet.</p>}
                {predictions.map((prediction) => (
                  <div key={prediction.id} className="list-row">
                    <div>
                      <strong>{prediction.id}</strong>
                      <span>{prediction.model_id}</span>
                    </div>
                    <div className="row-actions">
                      <span className="pill">{prediction.status}</span>
                      <button
                        onClick={() => cancelPrediction(prediction.id)}
                        disabled={busy}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
        {active === 'files' && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Upload</h2>
                  <p>Upload artifacts.</p>
                </div>
              </div>
              <input type="file" onChange={handleUpload} />
              <pre>{fileUploadResult || 'Upload response appears here.'}</pre>
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Open file</h2>
                  <p>Fetch by path.</p>
                </div>
              </div>
              <input
                placeholder="path/to/file"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
              />
              <button onClick={openFile} disabled={busy}>
                Open
              </button>
            </div>
          </section>
        )}
        {active === 'api-keys' && (
          <section className="grid two">
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Create key</h2>
                  <p>Generate API keys.</p>
                </div>
              </div>
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
            </div>
            <div className="panel">
              <div className="panel-header">
                <div>
                  <h2>Active keys</h2>
                </div>
                <button onClick={() => loadApiKeys(true)} disabled={busy}>
                  Refresh
                </button>
              </div>
              <div className="list">
                {!apiKeys.length && <p className="muted">No keys yet.</p>}
                {apiKeys.map((key) => (
                  <div key={key.id} className="list-row">
                    <div>
                      <strong>{key.name}</strong>
                      <span>({key.prefix})</span>
                    </div>
                    <div className="row-actions">
                      <span className="pill">
                        {key.is_admin ? 'admin' : 'standard'}
                      </span>
                      <button
                        onClick={() => revokeApiKey(key.id)}
                        disabled={busy || !key.is_active}
                      >
                        Revoke
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
        {active === 'system' && (
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>System</h2>
                <p>Quick checks.</p>
              </div>
              <button onClick={loadDashboard} disabled={busy}>
                Refresh
              </button>
            </div>
            <pre>{dashboard.system ? pretty(dashboard.system) : 'No data'}</pre>
            {error && (
              <div className="error">
                <strong>Last error</strong>
                <span>{error}</span>
              </div>
            )}
          </section>
        )}

        <section className="panel footnote">
          <div>
            <h3>Log</h3>
            <p>{log || 'No actions yet.'}</p>
          </div>
          {error && (
            <div className="error">
              <strong>Last error</strong>
              <span>{error}</span>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
