import { proxyRequest } from "@/lib/server-proxy";

type RouteContext = { params: Promise<{ path: string[] }> };

async function handle(request: Request, context: RouteContext): Promise<Response> {
  const { path } = await context.params;
  return proxyRequest(request, path);
}

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const GET = handle;
export const POST = handle;
export const PUT = handle;
export const PATCH = handle;
export const DELETE = handle;
