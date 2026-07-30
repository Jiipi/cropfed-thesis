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

  const hasActiveJob = useMemo(
    () => experiments.some((item) => ["queued", "running"].includes(item.status)),
    [experiments],
  );

  async function refresh() {
    try {
      const [projectData, classData, experimentData, profileData] = await Promise.all([
        api.project(),
        api.classes(),
        api.experiments(),
        api.dataProfiles().catch(() => ({ items: [] })),
      ]);
      setProject(projectData);
      setClasses(classData.items || []);
      setExperiments(experimentData);
      setDataProfiles(profileData.items || []);
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
    setError("");
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
              return (
                <article className="experiment" key={experiment.id}>
                  <div className="experiment-top">
                    <div>
                      <h3>{experiment.name}</h3>
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
                </article>
              );
            })}
          </div>
        </section>
      </section>

      <DataProfileSection profiles={dataProfiles} classes={classes} />
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
          <p className="eyebrow">Dá»® LIá»†U PHĂ‚N TĂN</p>
          <h2>PhĂ¢n bá»‘ lá»›p theo client</h2>
        </div>
        <span className="tag">Chá»‰ sá»‘ liá»‡u thá»‘ng kĂª</span>
      </div>
      <p className="fine-print profile-note">
        Server chá»‰ Ä‘á»c sá»‘ lÆ°á»£ng tá»« partition summary; khĂ´ng nháº­n byte áº£nh hoáº·c Ä‘Æ°á»ng dáº«n cá»¥c bá»™.
      </p>
      <div className="profile-grid">
        {ready.map((profile) => (
          <PartitionHeatmap key={profile.name} profile={profile} classes={classes} />
        ))}
      </div>
    </section>
  );
}

function PartitionHeatmap({ profile, classes }) {
  const maximum = Math.max(
    1,
    ...profile.clients.flatMap((client) => client.class_counts.map(Number)),
  );
  const title = profile.partition_kind === "iid"
    ? "IID"
    : `Non-IID Î±=${profile.dirichlet_alpha}`;

  return (
    <figure className="partition-card">
      <figcaption>
        <strong>{title}</strong>
        <span>{profile.num_samples} áº£nh train + validation</span>
      </figcaption>
      <div className="confusion-scroll">
        <table className="partition-heatmap">
          <thead>
            <tr>
              <th>Client</th>
              {classes.map((item) => <th key={item.id} title={item.name}>{item.id}</th>)}
              <th>Tá»•ng</th>
            </tr>
          </thead>
          <tbody>
            {profile.clients.map((client) => (
              <tr key={client.client_id}>
                <th>C{client.client_id}</th>
                {client.class_counts.map((count, index) => (
                  <td
                    key={index}
                    title={`Client ${client.client_id} Â· ${classes[index]?.name || index}: ${count}`}
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
