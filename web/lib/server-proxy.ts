const HOP_BY_HOP = [
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
];

const FORWARDED_REQUEST_HEADERS = [
  "accept",
  "content-type",
  "if-modified-since",
  "if-none-match",
  "range",
  "x-workspace-id",
];

const DEFAULT_PROXY_MAX_BODY_MB = 52;

type ProxyEnv = Readonly<Record<string, string | undefined>>;

function backendUrl(request: Request, path: string[], env: ProxyEnv): URL {
  const base =
    env.LATTICE_API_BASE || env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const normalized = base.endsWith("/") ? base : `${base}/`;
  const target = new URL(path.map(encodeURIComponent).join("/"), normalized);
  target.search = new URL(request.url).search;
  return target;
}

function upstreamHeaders(request: Request, env: ProxyEnv): Headers {
  const headers = new Headers();
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  const token = env.LATTICE_AUTH_TOKEN?.trim();
  if (token) headers.set("authorization", `Bearer ${token}`);
  return headers;
}

function maxBodyBytes(env: ProxyEnv): number {
  const configured = Number(
    env.LATTICE_PROXY_MAX_BODY_MB ?? DEFAULT_PROXY_MAX_BODY_MB,
  );
  const megabytes =
    Number.isFinite(configured) && configured > 0
      ? configured
      : DEFAULT_PROXY_MAX_BODY_MB;
  return Math.floor(megabytes * 1024 * 1024);
}

function bodyTooLarge(): Response {
  return Response.json(
    { detail: "request body exceeds proxy limit" },
    { status: 413 },
  );
}

export async function proxyRequest(
  request: Request,
  path: string[],
  env: ProxyEnv = process.env,
): Promise<Response> {
  try {
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    let body: ArrayBuffer | undefined;
    if (hasBody) {
      const cap = maxBodyBytes(env);
      const advertisedSize = Number(request.headers.get("content-length"));
      if (Number.isFinite(advertisedSize) && advertisedSize > cap) {
        return bodyTooLarge();
      }
      body = await request.arrayBuffer();
      if (body.byteLength > cap) return bodyTooLarge();
    }
    const upstream = await fetch(backendUrl(request, path, env), {
      method: request.method,
      headers: upstreamHeaders(request, env),
      body,
      cache: "no-store",
      redirect: "manual",
      signal: request.signal,
    });
    const headers = new Headers(upstream.headers);
    for (const name of HOP_BY_HOP) headers.delete(name);
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  } catch {
    return Response.json({ detail: "backend unavailable" }, { status: 502 });
  }
}
