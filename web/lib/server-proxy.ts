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
  const headers = new Headers(request.headers);
  for (const name of HOP_BY_HOP) headers.delete(name);
  headers.delete("host");
  headers.delete("content-length");
  headers.delete("authorization");
  const token = env.LATTICE_AUTH_TOKEN?.trim();
  if (token) headers.set("authorization", `Bearer ${token}`);
  return headers;
}

export async function proxyRequest(
  request: Request,
  path: string[],
  env: ProxyEnv = process.env,
): Promise<Response> {
  try {
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    const body = hasBody ? await request.arrayBuffer() : undefined;
    const upstream = await fetch(backendUrl(request, path, env), {
      method: request.method,
      headers: upstreamHeaders(request, env),
      body,
      cache: "no-store",
      redirect: "manual",
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
