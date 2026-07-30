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
  createExperiment: (payload) =>
    request("/experiments", { method: "POST", body: JSON.stringify(payload) }),
  startExperiment: (id) => request(`/experiments/${id}/start`, { method: "POST" }),
};
