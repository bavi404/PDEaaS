/**
 * Same-origin proxy to the FastAPI backend.
 * Why a route handler instead of next.config rewrites: the backend URL is read at RUNTIME (PDEFLOW_API_URL), so one
 * built image works in Docker/HF Spaces and locally, and an unreachable backend becomes a clear 502 JSON error
 * instead of an opaque proxy failure. Only the documented endpoints are forwarded.
 */
import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const ALLOWED = /^(health|model|samples(\/[A-Za-z0-9._-]+)?|activity|predict)$/;
const TIMEOUT_MS = 30_000;

function backend(): string {
  return (process.env.PDEFLOW_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
}

async function forward(req: NextRequest, path: string[]): Promise<NextResponse> {
  const joined = path.join("/");
  if (!ALLOWED.test(joined)) return NextResponse.json({ detail: "Not found" }, { status: 404 });
  const url = `${backend()}/${joined}${req.nextUrl.search}`;
  const headers: Record<string, string> = { Accept: "application/json" };
  const ct = req.headers.get("content-type");
  if (ct) headers["Content-Type"] = ct;
  const rid = req.headers.get("x-request-id");
  if (rid) headers["X-Request-ID"] = rid;
  try {
    const res = await fetch(url, {
      method: req.method,
      headers,
      body: req.method === "POST" ? await req.text() : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    const text = await res.text();
    const out = new NextResponse(text, {
      status: res.status,
      headers: { "Content-Type": res.headers.get("content-type") ?? "application/json" },
    });
    const backendRid = res.headers.get("x-request-id");
    if (backendRid) out.headers.set("X-Request-ID", backendRid);
    return out;
  } catch {
    return NextResponse.json({ detail: "The PDEFlow API is unavailable. Check that the backend is running." }, { status: 502 });
  }
}

type Ctx = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, ctx: Ctx) {
  return forward(req, (await ctx.params).path);
}

export async function POST(req: NextRequest, ctx: Ctx) {
  return forward(req, (await ctx.params).path);
}
