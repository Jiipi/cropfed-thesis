import React, { useEffect, useMemo, useState } from "react";
import { api, clearApiToken, setApiToken } from "./api";

const initialForm = {
  name: "FedAvg — Dirichlet α=0.5",
  execution_mode: "synthetic-smoke",
  algorithm: "fedavg",
  partition_kind: "dirichlet",
  num_clients: 4,
  num_rounds: 5,
  local_epochs: 2,
  learning_rate: 0.05,
  batch_size: 32,
  dirichlet_alpha: 0.5,
  proximal_mu: 0.01,
  seed: 2026,
};

const statusText = {
  draft: "Bản nháp",
  queued: "Đang chờ",
  running: "Đang chạy",
  completed: "Hoàn tất",
  failed: "Thất bại",
};

export default function App() {
  const [project, setProject] = useState(null);
  const [classes, setClasses] = useState([]);
  const [dataProfiles, setDataProfiles] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [form, setForm] = useState(initialForm);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [authChecked, setAuthChecked] = useState(false);
  const [principal, setPrincipal] = useState(null);
  const [tokenInput, setTokenInput] = useState("");
  const [registeredClients, setRegisteredClients] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [compareIds, setCompareIds] = useState(new Set());
  const [compareData, setCompareData] = useState(null);
  const [predictResult, setPredictResult] = useState(null);
  const [predictFile, setPredictFile] = useState(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [activeTab, setActiveTab] = useState("experiments");

  const hasActiveJob = useMemo(
    () => experiments.some((item) => ["queued", "running"].includes(item.status)),
    [experiments],
  );

  async function refresh() {
    try {
      const [projectData, classData, experimentData, profileData, clientData, checkpointData] = await Promise.all([
        api.project(),
        api.classes(),
        api.experiments(),
        api.dataProfiles().catch(() => ({ items: [] })),
        api.listClients().catch(() => []),
        api.listCheckpoints().catch(() => []),
      ]);
      setProject(projectData);
      setClasses(classData.items || []);
      setExperiments(experimentData);
      setDataProfiles(profileData.items || []);
      setRegisteredClients(clientData);
      setCheckpoints(checkpointData);
      setError("");
    } catch (requestError) {
      if (requestError.status === 401) {
        clearApiToken();
        setPrincipal(null);
      }
      setError(requestError.message);
    }
  }

  useEffect(() => {
    async function bootstrap() {
      try {
        const identity = await api.me();
        setPrincipal(identity);
        await refresh();
      } catch (requestError) {
        if (requestError.status === 401) clearApiToken();
        else setError(requestError.message);
      } finally {
        setAuthChecked(true);
      }
    }
    bootstrap();
  }, []);

  useEffect(() => {
    if (!hasActiveJob) return undefined;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [hasActiveJob]);

  function updateField(event) {
    const { name, value, type } = event.target;
    setForm((current) => {
      const nextValue = type === "number" ? Number(value) : value;
      if (name === "execution_mode") {
        return {
          ...current,
          execution_mode: value,
          num_clients: 4,
          learning_rate: value === "flower" ? 0.001 : 0.05,
        };
      }
      return { ...current, [name]: nextValue };
    });
  }

  async function createExperiment(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await api.createExperiment(form);
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function startExperiment(id) {
    setBusy(true);
    try {
      await api.startExperiment(id);
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function login(event) {
    event.preventDefault();
    setBusy(true);
    setApiToken(tokenInput);
    try {
      const identity = await api.me();
      setPrincipal(identity);
      setTokenInput("");
      setError("");
      await refresh();
    } catch (requestError) {
      clearApiToken();
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function logout() {
    clearApiToken();
    setPrincipal(null);
    setProject(null);
    setExperiments([]);
    setRegisteredClients([]);
    setCheckpoints([]);
    setCompareData(null);
    setPredictResult(null);
    setError("");
  }

  async function handleCreateClient(event) {
    event.preventDefault();
    setBusy(true);
    try {
      const fd = new FormData(event.target);
      await api.createClient({
        name: fd.get("client_name"),
        description: fd.get("client_desc") || "",
        partition_id: fd.get("partition_id") ? Number(fd.get("partition_id")) : null,
      });
      event.target.reset();
      await refresh();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function toggleCompare(id) {
    setCompareIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function runComparison() {
    setBusy(true);
    try {
      const result = await api.compareExperiments([...compareIds]);
      setCompareData(result);
      setActiveTab("compare");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handlePredict(event) {
    event.preventDefault();
    if (!predictFile) return;
    setBusy(true);
    try {
      const result = await api.predict(predictFile, selectedCheckpoint || null);
      setPredictResult(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleExport() {
    const completed = experiments.filter((e) => e.status === "completed").map((e) => e.id);
    if (!completed.length) {
      setError("Không có thí nghiệm hoàn tất để xuất.");
      return;
    }
    setBusy(true);
    try {
      await api.exportCsv(completed);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  if (!authChecked) {
    return <main className="auth-shell"><p>Đang kiểm tra phiên đăng nhập…</p></main>;
  }

  if (!principal) {
    return (
      <main className="auth-shell">
        <form className="card auth-card" onSubmit={login}>
          <p className="eyebrow">CROPFED · SECURE DASHBOARD</p>
          <h1>Đăng nhập</h1>
          <p>Nhập bearer token do quản trị viên hệ thống cấp.</p>
          {error && <div className="alert">{error}</div>}
          <label>
            API token
            <input
              autoComplete="current-password"
              autoFocus
              onChange={(event) => setTokenInput(event.target.value)}
              required
              type="password"
              value={tokenInput}
            />
          </label>
          <button disabled={busy} type="submit">
            {busy ? "Đang xác thực…" : "Mở dashboard"}
          </button>
        </form>
      </main>
    );
  }

  const canWrite = principal.role === "admin";

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">ĐỒ ÁN KỸ THUẬT PHẦN MỀM · FEDERATED LEARNING</p>
        <h1>CropFed</h1>
        <p className="official-title">
          {project?.official_title || "Đang tải tên đề tài chính thức…"}
        </p>
        <div className="scope-note">
          <strong>Phạm vi hiện tại:</strong> phân loại đa lớp ở mức ảnh, 10 lớp cà chua,
          4 client mô phỏng. Ảnh thô không được gửi lên server.
        </div>
        <div className="session-bar">
          <span>Quyền hiện tại: <strong>{principal.role}</strong></span>
          {principal.authentication_enabled && (
            <button className="secondary" type="button" onClick={logout}>
              Đăng xuất
            </button>
          )}
        </div>
      </header>

      {error && <div className="alert">{error}</div>}

      <section className="layout">
        {canWrite ? (
        <form className="card form-card" onSubmit={createExperiment}>
          <div className="section-heading">
            <div>
              <p className="eyebrow">CẤU HÌNH</p>
              <h2>Tạo thí nghiệm</h2>
            </div>
            <span className="tag">
              {form.execution_mode === "flower" ? "Flower worker" : "Synthetic only"}
            </span>
          </div>

          <label>
            Tên thí nghiệm
            <input name="name" value={form.name} onChange={updateField} />
          </label>
          <div className="field-grid">
            <label>
              Chế độ chạy
              <select
                name="execution_mode"
                value={form.execution_mode}
                onChange={updateField}
              >
                <option value="synthetic-smoke">Synthetic smoke</option>
                <option value="flower">Flower image training</option>
              </select>
            </label>
            <label>
              Thuật toán
              <select name="algorithm" value={form.algorithm} onChange={updateField}>
                <option value="fedavg">FedAvg</option>
                <option value="fedprox">FedProx</option>
                <option value="fedbn">FedBN</option>
                <option value="scaffold">SCAFFOLD</option>
                <option value="moon">MOON</option>
              </select>
            </label>
            <label>
              Phân hoạch
              <select
                name="partition_kind"
                value={form.partition_kind}
                onChange={updateField}
              >
                <option value="dirichlet">Non-IID Dirichlet</option>
                <option value="iid">IID</option>
                <option value="quantity_skew">Non-IID lệch số lượng</option>
                <option value="feature_skew">Non-IID lệch đặc trưng</option>
              </select>
            </label>
            <NumberField label="Số client" name="num_clients" value={form.num_clients} onChange={updateField} min="2" disabled={form.execution_mode === "flower"} />
            <NumberField label="Số round" name="num_rounds" value={form.num_rounds} onChange={updateField} min="1" />
            <NumberField label="Local epoch" name="local_epochs" value={form.local_epochs} onChange={updateField} min="1" />
            <NumberField label="Dirichlet α" name="dirichlet_alpha" value={form.dirichlet_alpha} onChange={updateField} min="0.01" step="0.01" />
            <NumberField label="Learning rate" name="learning_rate" value={form.learning_rate} onChange={updateField} min="0.000001" step="0.0001" />
            <NumberField label="Batch size" name="batch_size" value={form.batch_size} onChange={updateField} min="1" />
            {form.algorithm === "fedprox" && (
              <NumberField label="FedProx μ" name="proximal_mu" value={form.proximal_mu} onChange={updateField} min="0" step="0.001" />
            )}
            <NumberField label="Seed" name="seed" value={form.seed} onChange={updateField} min="0" />
          </div>
          <button disabled={busy} type="submit">
            {busy ? "Đang xử lý…" : "Lưu cấu hình"}
          </button>
          <p className="fine-print">
            {form.execution_mode === "flower"
              ? "Flower job chỉ được xếp hàng khi worker đã bật và profile dữ liệu đã audit tồn tại trên server. Request không nhận command hoặc đường dẫn dữ liệu."
              : "Smoke experiment dùng vector tổng hợp để kiểm tra luồng. Không dùng số liệu này trong chương kết quả của báo cáo."}
          </p>
        </form>
        ) : (
          <section className="card form-card">
            <p className="eyebrow">CHẾ ĐỘ CHỈ ĐỌC</p>
            <h2>Quyền viewer</h2>
            <p>
              Bạn có thể xem cấu hình và kết quả; chỉ admin mới được tạo hoặc chạy
              thí nghiệm.
            </p>
          </section>
        )}

        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">LỊCH SỬ</p>
              <h2>Thí nghiệm gần đây</h2>
            </div>
            <button className="secondary" type="button" onClick={refresh}>
              Làm mới
            </button>
          </div>

          <div className="experiment-list">
            {experiments.length === 0 && (
              <p className="empty">Chưa có thí nghiệm. Hãy lưu cấu hình đầu tiên.</p>
            )}
            {experiments.map((experiment) => {
              const metrics = experiment.result?.final_metrics;
              const flowerResult = experiment.result?.flower;
              const flowerHistory = experiment.result?.history || [];
              const latestFlowerMetrics = flowerHistory.at(-1)?.central_evaluate;
              const experimentHistory = experiment.result?.history || [];
              const communication = experiment.result?.communication;
              const latestRound = Math.max(
                0,
                ...((experiment.result?.client_history || []).map((item) =>
                  Number(item.round),
                )),
              );
              const clientF1 = (experiment.result?.client_history || [])
                .filter(
                  (item) =>
                    item.phase === "evaluate" && Number(item.round) === latestRound,
                )
                .map((item) => Number(item.metrics?.eval_macro_f1))
                .filter(Number.isFinite);
              const worstClientF1 = clientF1.length ? Math.min(...clientF1) : null;
              const isSelected = compareIds.has(experiment.id);
              return (
                <article className={`experiment ${isSelected ? "experiment-selected" : ""}`} key={experiment.id}>
                  <div className="experiment-top">
                    <div>
                      <label className="compare-check">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleCompare(experiment.id)}
                        />
                        <h3>{experiment.name}</h3>
                      </label>
                      <p>
                        {experiment.execution_mode} · {experiment.algorithm.toUpperCase()} · {experiment.num_clients} client
                        · {experiment.num_rounds} round · α={experiment.dirichlet_alpha}
                      </p>
                    </div>
                    <span className={`status status-${experiment.status}`}>
                      {statusText[experiment.status] || experiment.status}
                    </span>
                  </div>
                  {metrics && (
                    <div className="metrics">
                      <Metric label="Accuracy" value={metrics.accuracy} />
                      <Metric label="Macro F1" value={metrics.macro_f1} />
                      <Metric
                        label="Dữ liệu"
                        value={`${experiment.result.data.num_train} mẫu`}
                        numeric={false}
                      />
                    </div>
                  )}
                  {flowerResult && (
                    <div className="metrics">
                      {latestFlowerMetrics ? (
                        <>
                          <Metric label="Accuracy" value={latestFlowerMetrics.central_accuracy} />
                          <Metric label="Macro F1" value={latestFlowerMetrics.central_macro_f1} />
                          <Metric
                            label="Bỏ sót có hại"
                            value={latestFlowerMetrics.central_harmful_missed_as_healthy_rate}
                          />
                          <Metric
                            label="Spider-mite F1"
                            value={latestFlowerMetrics.central_spider_mite_f1}
                          />
                        </>
                      ) : (
                        <Metric label="Client" value={`${flowerResult.num_clients}/4`} numeric={false} />
                      )}
                      <Metric label="Checkpoint" value={`${(flowerResult.checkpoint_bytes / 1_000_000).toFixed(2)} MB`} numeric={false} />
                      {Number.isFinite(worstClientF1) && (
                        <Metric label="Worst-client F1" value={worstClientF1} />
                      )}
                      {communication?.payload_total_bytes != null && (
                        <Metric
                          label="Payload"
                          value={`${(communication.payload_total_bytes / 1_000_000).toFixed(2)} MB`}
                          numeric={false}
                        />
                      )}
                    </div>
                  )}
                  {experiment.status === "completed" && (
                    <RoundChart history={experimentHistory} />
                  )}
                  {latestFlowerMetrics?.central_confusion_matrix_flat && (
                    <ConfusionMatrix
                      flat={latestFlowerMetrics.central_confusion_matrix_flat}
                      size={latestFlowerMetrics.central_confusion_matrix_size}
                      classes={classes}
                    />
                  )}
                  {metrics?.confusion_matrix && (
                    <ConfusionMatrix matrix={metrics.confusion_matrix} classes={classes} />
                  )}
                  {experiment.error_message && (
                    <p className="inline-error">{experiment.error_message}</p>
                  )}
                  {canWrite && ["draft", "failed"].includes(experiment.status) && (
                    <button
                      className="secondary"
                      disabled={busy}
                      type="button"
                      onClick={() => startExperiment(experiment.id)}
                    >
                      {experiment.execution_mode === "flower" ? "Xếp hàng Flower" : "Chạy smoke test"}
                    </button>
                  )}
                  {clientF1.length > 0 && (
                    <ClientF1Bar clientF1={clientF1} />
                  )}
                </article>
              );
            })}
          </div>

          {experiments.length > 0 && (
            <div className="export-bar">
              {compareIds.size >= 2 && (
                <button className="secondary" type="button" onClick={runComparison} disabled={busy}>
                  So sánh {compareIds.size} thí nghiệm
                </button>
              )}
              <button className="secondary" type="button" onClick={handleExport} disabled={busy}>
                Xuất CSV
              </button>
            </div>
          )}
        </section>
      </section>

      <DataProfileSection profiles={dataProfiles} classes={classes} />

      <nav className="tab-bar">
        <button type="button" className={activeTab === "experiments" ? "tab-active" : ""} onClick={() => setActiveTab("experiments")}>Thí nghiệm</button>
        <button type="button" className={activeTab === "clients" ? "tab-active" : ""} onClick={() => setActiveTab("clients")}>Cơ sở</button>
        <button type="button" className={activeTab === "predict" ? "tab-active" : ""} onClick={() => setActiveTab("predict")}>Dự đoán</button>
        <button type="button" className={activeTab === "compare" ? "tab-active" : ""} onClick={() => setActiveTab("compare")}>So sánh</button>
        <button type="button" className={activeTab === "checkpoints" ? "tab-active" : ""} onClick={() => setActiveTab("checkpoints")}>Checkpoint</button>
      </nav>

      {activeTab === "clients" && (
        <ClientSection
          clients={registeredClients}
          canWrite={canWrite}
          onCreateClient={handleCreateClient}
          busy={busy}
        />
      )}

      {activeTab === "predict" && (
        <PredictSection
          file={predictFile}
          onFileChange={setPredictFile}
          checkpoints={checkpoints}
          selectedCheckpoint={selectedCheckpoint}
          onCheckpointChange={setSelectedCheckpoint}
          onSubmit={handlePredict}
          result={predictResult}
          busy={busy}
        />
      )}

      {activeTab === "compare" && (
        <CompareSection data={compareData} />
      )}

      {activeTab === "checkpoints" && (
        <CheckpointSection checkpoints={checkpoints} />
      )}
    </main>
  );
}

function DataProfileSection({ profiles, classes }) {
  const ready = profiles.filter((profile) => profile.available);
  if (!ready.length) return null;

  return (
    <section className="card profile-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">DỮ LIỆU PHÂN TÁN</p>
          <h2>Phân bố lớp theo client</h2>
        </div>
        <span className="tag">Chỉ số liệu thống kê</span>
      </div>
      <p className="fine-print profile-note">
        Server chỉ đọc số lượng từ partition summary; không nhận byte ảnh hoặc đường dẫn cục bộ.
      </p>
      <div className="profile-grid">
        {ready.map((profile) => (
          <PartitionHeatmap key={profile.name} profile={profile} classes={classes} />
        ))}
      </div>
    </section>
  );
}

function profileTitle(profile) {
  // Drive the label from skew_type, not from dirichlet_alpha alone: quantity and
  // feature skew carry no alpha and used to render as "Non-IID α=null".
  switch (profile.skew_type) {
    case "label":
      return `Non-IID lệch nhãn α=${profile.dirichlet_alpha}`;
    case "quantity":
      return "Non-IID lệch số lượng";
    case "feature":
      return `Non-IID lệch đặc trưng (strength=${profile.feature_skew_strength})`;
    default:
      return "IID";
  }
}

function PartitionHeatmap({ profile, classes }) {
  const maximum = Math.max(
    1,
    ...profile.clients.flatMap((client) => client.class_counts.map(Number)),
  );
  const title = profileTitle(profile);

  return (
    <figure className="partition-card">
      <figcaption>
        <strong>{title}</strong>
        <span>{profile.num_samples} ảnh train + validation</span>
      </figcaption>
      <div className="confusion-scroll">
        <table className="partition-heatmap">
          <thead>
            <tr>
              <th>Client</th>
              {classes.map((item) => <th key={item.id} title={item.name}>{item.id}</th>)}
              <th>Tổng</th>
            </tr>
          </thead>
          <tbody>
            {profile.clients.map((client) => (
              <tr key={client.client_id}>
                <th>C{client.client_id}</th>
                {client.class_counts.map((count, index) => (
                  <td
                    key={index}
                    title={`Client ${client.client_id} · ${classes[index]?.name || index}: ${count}`}
                    style={{ backgroundColor: `rgba(49, 91, 56, ${0.06 + (count / maximum) * 0.72})` }}
                  >
                    {count}
                  </td>
                ))}
                <th>{client.num_samples}</th>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}

function NumberField({ label, ...props }) {
  return (
    <label>
      {label}
      <input type="number" {...props} />
    </label>
  );
}

function Metric({ label, value, numeric = true }) {
  const display = numeric
    ? Number.isFinite(Number(value))
      ? Number(value).toFixed(4)
      : "—"
    : value;
  return (
    <div>
      <span>{label}</span>
      <strong>{display}</strong>
    </div>
  );
}

function RoundChart({ history }) {
  const values = history
    .map((item) => ({
      round: Number(item.round),
      macroF1: Number(
        item.central_evaluate?.central_macro_f1 ??
          item.federated_evaluate?.eval_macro_f1 ??
          item.macro_f1,
      ),
    }))
    .filter((item) => Number.isFinite(item.round) && Number.isFinite(item.macroF1));

  if (values.length === 0) return null;

  const width = 320;
  const height = 82;
  const insetX = 18;
  const insetY = 12;
  const minRound = Math.min(...values.map((item) => item.round));
  const maxRound = Math.max(...values.map((item) => item.round));
  const roundSpan = Math.max(1, maxRound - minRound);
  const coordinates = values.map((item) => ({
    ...item,
    x: insetX + ((item.round - minRound) / roundSpan) * (width - insetX * 2),
    y: height - insetY - Math.max(0, Math.min(1, item.macroF1)) * (height - insetY * 2),
  }));
  const points = coordinates.map((item) => `${item.x},${item.y}`).join(" ");

  return (
    <figure className="round-chart">
      <figcaption>
        <span>Macro-F1 theo round</span>
        <strong>{values.at(-1).macroF1.toFixed(4)}</strong>
      </figcaption>
      <svg
        role="img"
        aria-label={`Đường cong Macro-F1 từ round ${minRound} đến ${maxRound}`}
        viewBox={`0 0 ${width} ${height}`}
      >
        <line x1={insetX} y1={insetY} x2={insetX} y2={height - insetY} />
        <line
          x1={insetX}
          y1={height - insetY}
          x2={width - insetX}
          y2={height - insetY}
        />
        <polyline points={points} />
        {coordinates.map((item) => (
          <circle key={item.round} cx={item.x} cy={item.y} r="2.8">
            <title>{`Round ${item.round}: ${item.macroF1.toFixed(4)}`}</title>
          </circle>
        ))}
      </svg>
      <div className="chart-axis-labels">
        <span>Round {minRound}</span>
        <span>Round {maxRound}</span>
      </div>
    </figure>
  );
}

function ConfusionMatrix({ flat, matrix, size, classes }) {
  const dimension = Number(size) || matrix?.length || 0;
  const values = matrix || Array.from({ length: dimension }, (_, row) =>
    flat.slice(row * dimension, (row + 1) * dimension),
  );
  if (!dimension || values.length !== dimension) return null;
  const maximum = Math.max(1, ...values.flat().map(Number));
  const className = (index) => classes[index]?.name || `Lớp ${index}`;

  return (
    <figure className="confusion-card">
      <figcaption>
        <strong>Confusion matrix</strong>
        <span>Hàng: nhãn thật · Cột: dự đoán</span>
      </figcaption>
      <div className="confusion-scroll">
        <table className="confusion-matrix">
          <thead>
            <tr>
              <th aria-label="Nhãn thật và dự đoán" />
              {Array.from({ length: dimension }, (_, index) => (
                <th key={index} title={className(index)}>{index}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {values.map((row, actual) => (
              <tr key={actual}>
                <th title={className(actual)}>{actual}</th>
                {row.map((value, predicted) => {
                  const numeric = Number(value);
                  const intensity = 0.08 + (numeric / maximum) * 0.72;
                  return (
                    <td
                      key={predicted}
                      title={`${className(actual)} → ${className(predicted)}: ${numeric}`}
                      style={{ backgroundColor: `rgba(49, 91, 56, ${intensity})` }}
                    >
                      {numeric}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="class-legend">
        {Array.from({ length: dimension }, (_, index) => (
          <span key={index}><strong>{index}</strong> {className(index)}</span>
        ))}
      </div>
    </figure>
  );
}

function ClientF1Bar({ clientF1 }) {
  return (
    <div className="client-f1-bar">
      <span>F1 từng client:</span>
      <div className="client-f1-chips">
        {clientF1.map((val, idx) => (
          <span key={idx} className="chip">
            C{idx}: <strong>{val.toFixed(4)}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function ClientSection({ clients, canWrite, onCreateClient, busy }) {
  return (
    <section className="card tab-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">QUẢN LÝ</p>
          <h2>Cơ sở nông nghiệp</h2>
        </div>
        <span className="tag">{clients.length} cơ sở</span>
      </div>

      {canWrite && (
        <form className="client-form" onSubmit={onCreateClient}>
          <h3>Đăng ký cơ sở mới</h3>
          <div className="field-grid">
            <label>
              Tên cơ sở
              <input name="client_name" placeholder="Nông trại Củ Chi..." required />
            </label>
            <label>
              Partition ID
              <input name="partition_id" type="number" min="0" max="19" placeholder="0" />
            </label>
          </div>
          <label>
            Mô tả
            <input name="client_desc" placeholder="Trang trại cà chua hữu cơ..." />
          </label>
          <button disabled={busy} type="submit">
            {busy ? "Đang xử lý…" : "Đăng ký cơ sở"}
          </button>
        </form>
      )}

      <div className="client-list">
        <h3>Danh sách cơ sở đã đăng ký</h3>
        {clients.length === 0 ? (
          <p className="empty">Chưa có cơ sở nào được đăng ký.</p>
        ) : (
          <div className="confusion-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tên</th>
                  <th>Mô tả</th>
                  <th>Partition</th>
                  <th>Trạng thái</th>
                  <th>Ngày tạo</th>
                </tr>
              </thead>
              <tbody>
                {clients.map((c) => (
                  <tr key={c.id}>
                    <td><strong>{c.name}</strong></td>
                    <td>{c.description || "—"}</td>
                    <td>{c.partition_id != null ? `Client ${c.partition_id}` : "Tự do"}</td>
                    <td>
                      <span className={`status status-${c.status === "connected" ? "completed" : "queued"}`}>
                        {c.status}
                      </span>
                    </td>
                    <td>{new Date(c.created_at).toLocaleDateString("vi-VN")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function PredictSection({
  file,
  onFileChange,
  checkpoints,
  selectedCheckpoint,
  onCheckpointChange,
  onSubmit,
  result,
  busy,
}) {
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  return (
    <section className="card tab-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">CHẨN ĐOÁN</p>
          <h2>Dự đoán ảnh lá cà chua</h2>
        </div>
        <span className="tag">Inference local / server demo</span>
      </div>

      <div className="privacy-banner">
        <strong>⚠️ Lưu ý Quyền riêng tư & Quy trình:</strong>
        <p>
          Luồng chẩn đoán ảnh từ web này là <strong>tính năng tự nguyện</strong> nhằm mục đích thử nghiệm và demo.
          Ảnh đăng tải ở đây <strong>KHÔNG</strong> được đưa vào tập huấn luyện của Federated Learning.
          Dữ liệu huấn luyện riêng tư của các cơ sở nông nghiệp luôn giữ nguyên tại thiết bị client và không bao giờ tải lên server.
        </p>
      </div>

      <form className="predict-form" onSubmit={onSubmit}>
        <div className="field-grid">
          <label>
            Chọn mô hình (Checkpoint)
            <select
              value={selectedCheckpoint}
              onChange={(e) => onCheckpointChange(e.target.value)}
            >
              <option value="">Mặc định (Deployed Checkpoint)</option>
              {checkpoints.map((cp) => (
                <option key={cp.filename} value={cp.filename}>
                  {cp.filename} ({cp.model_name || "MobileNetV2"} — {(cp.size_bytes / 1_000_000).toFixed(1)} MB)
                </option>
              ))}
            </select>
          </label>
          <label>
            Chọn ảnh lá cà chua
            <input
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => onFileChange(e.target.files[0] || null)}
              required
            />
          </label>
        </div>
        {file && (
          <div className="image-preview">
            <img src={previewUrl} alt="Preview" />
            <span>{file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
          </div>
        )}
        <button disabled={busy || !file} type="submit">
          {busy ? "Đang xử lý…" : "Chạy chẩn đoán"}
        </button>
      </form>

      {result && (
        <div className="predict-result card">
          <div className="section-heading">
            <h3>Kết quả chẩn đoán</h3>
            <span className="tag tag-success">
              {result.inference_ms} ms
            </span>
          </div>

          <div className="metrics">
            <Metric label="Cây trồng" value={result.crop} numeric={false} />
            <Metric label="Nhóm" value={result.predicted_group} numeric={false} />
            <Metric label="Chẩn đoán chính" value={result.predicted_label} numeric={false} />
            <Metric label="Độ tin cậy" value={`${(result.confidence * 100).toFixed(2)}%`} numeric={false} />
          </div>

          <h4>Top-3 dự đoán có xác suất cao nhất:</h4>
          <table className="data-table">
            <thead>
              <tr>
                <th>Hạng</th>
                <th>Tên bệnh / tình trạng</th>
                <th>Nhóm</th>
                <th>Xác suất</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p, idx) => (
                <tr key={idx}>
                  <td>#{idx + 1}</td>
                  <td><strong>{p.label}</strong></td>
                  <td>{p.group}</td>
                  <td>{(p.confidence * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="disclaimer-note">
            <p><strong>Cảnh báo:</strong> {result.warning}</p>
            <p className="fine-print">{result.privacy_notice}</p>
          </div>
        </div>
      )}
    </section>
  );
}

function CompareSection({ data }) {
  if (!data || !data.items || data.items.length === 0) {
    return (
      <section className="card tab-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">SO SÁNH TRỰC QUAN</p>
            <h2>Bảng so sánh thí nghiệm</h2>
          </div>
        </div>
        <p className="empty">
          Hãy chọn ít nhất 2 thí nghiệm ở tab "Thí nghiệm" (bằng checkbox) rồi nhấn "So sánh".
        </p>
      </section>
    );
  }

  return (
    <section className="card tab-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">SO SÁNH TRỰC QUAN</p>
          <h2>Bảng so sánh {data.items.length} thí nghiệm</h2>
        </div>
      </div>

      <div className="confusion-scroll">
        <table className="data-table compare-table">
          <thead>
            <tr>
              <th>Tiêu chí</th>
              {data.items.map((item) => (
                <th key={item.id}>
                  <strong>{item.name}</strong>
                  <div className="fine-print">{item.algorithm.toUpperCase()} · {item.partition_kind}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Chế độ</th>
              {data.items.map((item) => <td key={item.id}>{item.execution_mode}</td>)}
            </tr>
            <tr>
              <th>Số client / Round</th>
              {data.items.map((item) => <td key={item.id}>{item.num_clients} / {item.num_rounds}</td>)}
            </tr>
            <tr>
              <th>Accuracy (cuối)</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  <strong>{item.final_accuracy != null ? item.final_accuracy.toFixed(4) : "—"}</strong>
                </td>
              ))}
            </tr>
            <tr>
              <th>Macro F1 (cuối)</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  <strong>{item.final_macro_f1 != null ? item.final_macro_f1.toFixed(4) : "—"}</strong>
                </td>
              ))}
            </tr>
            <tr>
              <th>Worst-Client F1</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  {item.worst_client_f1 != null ? item.worst_client_f1.toFixed(4) : "—"}
                </td>
              ))}
            </tr>
            <tr>
              <th>Bỏ sót có hại</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  {item.final_harmful_rate != null ? item.final_harmful_rate.toFixed(4) : "—"}
                </td>
              ))}
            </tr>
            <tr>
              <th>Tổng dung lượng (MB)</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  {((item.total_bytes_up + item.total_bytes_down) / 1_000_000).toFixed(2)} MB
                </td>
              ))}
            </tr>
            <tr>
              <th>Thời gian (giây)</th>
              {data.items.map((item) => (
                <td key={item.id}>
                  {item.total_elapsed ? item.total_elapsed.toFixed(1) : "—"}s
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CheckpointSection({ checkpoints }) {
  return (
    <section className="card tab-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">QUẢN LÝ CHECKPOINT</p>
          <h2>Mô hình đã lưu (.pt)</h2>
        </div>
        <span className="tag">{checkpoints.length} file</span>
      </div>

      {checkpoints.length === 0 ? (
        <p className="empty">Chưa có checkpoint nào được tạo.</p>
      ) : (
        <div className="confusion-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Tên file</th>
                <th>Model</th>
                <th>Kích thước</th>
                <th>Loại thí nghiệm</th>
                <th>Ngày tạo</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((cp) => (
                <tr key={cp.path}>
                  <td><strong>{cp.filename}</strong></td>
                  <td>{cp.model_name || "MobileNetV2"}</td>
                  <td>{(cp.size_bytes / 1_000_000).toFixed(2)} MB</td>
                  <td>{cp.experiment_type || "Federated"}</td>
                  <td>{cp.created_at ? new Date(cp.created_at).toLocaleString("vi-VN") : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
