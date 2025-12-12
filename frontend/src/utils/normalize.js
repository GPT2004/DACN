// Helper to normalize various API response shapes into arrays
export function ensureArray(resp, fallback = []) {
  if (!resp) return fallback;
  if (Array.isArray(resp)) return resp;

  // common wrappers at top level
  if (Array.isArray(resp.data)) return resp.data;
  if (Array.isArray(resp.items)) return resp.items;
  if (Array.isArray(resp.users)) return resp.users;
  if (Array.isArray(resp.medicines)) return resp.medicines;

  // If resp.data is an object that contains arrays (e.g. { data: { users: [...], total: ... }})
  if (resp.data && typeof resp.data === 'object') {
    // Priority: users, medicines, items, data
    if (Array.isArray(resp.data.users)) return resp.data.users;
    if (Array.isArray(resp.data.medicines)) return resp.data.medicines;
    if (Array.isArray(resp.data.items)) return resp.data.items;
    if (Array.isArray(resp.data.data)) return resp.data.data;
    
    // fallback: find any array property in data
    for (const k of Object.keys(resp.data)) {
      if (Array.isArray(resp.data[k])) return resp.data[k];
    }
  }

  // fallback: try to find first array property at any level
  for (const key of Object.keys(resp)) {
    if (Array.isArray(resp[key])) return resp[key];
  }

  return fallback;
}
