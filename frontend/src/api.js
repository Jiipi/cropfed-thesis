const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const TOKEN_KEY = "cropfed.apiToken";

export function setApiToken(token) {
  window.sessionStorage.setItem(TOKEN_KEY, token.trim());
}

export function clearApiToken() {
  window.sessionStorage.removeItem(TOKEN_KEY);
}

function getApiToken() {
  return window.sessionStorage.getItem(TOKEN_KEY) || "";
}

async function request(path, options = {}) {
  const token = getApiToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = Array.isArray(body.detail)
      ? body.detail
          .map((item) => {
            const location = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
            return [location, item.msg].filter(Boolean).join(": ");
          })
          .join("; ")
      : body.detail;
    const error = new Error(detail || `HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export const api = {
  me: () => request("/auth/me"),
  project: () => request("/project"),
  classes: () => request("/classes"),
  dataProfiles: () => request("/data-profiles"),
  experiments: () => request("/experiments"),
  rounds: (id) => request(`/experiments/${id}/rounds`),
  clients: (id) => request(`/experiments/${id}/clients`),
  createExperiment: (payload) =>
    request("/experiments", { method: "POST", body: JSON.stringify(payload) }),
  startExperiment: (id) => request(`/experiments/${id}/start`, { method: "POST" }),

  // Client management
  listClients: () => request("/clients"),
  createClient: (payload) =>
    request("/clients", { method: "POST", body: JSON.stringify(payload) }),
  updateClientStatus: (id, payload) =>
    request(`/clients/${id}/status`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  // Prediction (voluntary upload)
  predict: async (file, checkpointName) => {
    const formData = new FormData();
    formData.append("image", file);
    const token = getApiToken();
    const url = `${API_BASE}/predict${checkpointName ? `?checkpoint_name=${encodeURIComponent(checkpointName)}` : ""}`;
    const response = await fetch(url, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const error = new Error(body.detail || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  },

  // Checkpoints
  listCheckpoints: () => request("/checkpoints"),

  // Comparison
  compareExperiments: (ids) => {
    const params = ids.map((id) => `ids=${id}`).join("&");
    return request(`/experiments/compare?${params}`);
  },

  // CSV export
  exportCsv: async (ids) => {
    const params = ids.map((id) => `ids=${id}`).join("&");
    const token = getApiToken();
    const response = await fetch(`${API_BASE}/experiments/export-csv?${params}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cropfed_experiments.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  },
};
