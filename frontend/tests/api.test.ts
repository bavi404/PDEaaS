import { describe, expect, it, vi } from "vitest";
import { ApiError, createApi, formatDetail } from "@/lib/api";

const json = (status: number, body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

describe("API client", () => {
  it("calls the same-origin proxy and returns typed data", async () => {
    const f = vi.fn(() => json(200, { status: "ok", model_loaded: true, checkpoint_sha256: "abc", n_samples: 24 }));
    const h = await createApi(f as unknown as typeof fetch).health();
    expect(h.n_samples).toBe(24);
    expect(f).toHaveBeenCalledWith("/api/health", expect.objectContaining({ cache: "no-store" }));
  });

  it("POSTs predictions as JSON", async () => {
    const f = vi.fn((_url: string, _init?: RequestInit) => json(200, { prediction: [[1]] }));
    await createApi(f as unknown as typeof fetch).predict({ sample_id: "test-1", resolution: 64, return_targets: true });
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/api/predict");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ sample_id: "test-1", resolution: 64, return_targets: true });
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
  });

  it("maps a 422 with pydantic errors to a readable validation error with request id", async () => {
    const f = vi.fn(() => json(422, { detail: [{ loc: ["body", "resolution"], msg: "Input should be 64 or 128" }], request_id: "rid-1" }));
    const err = await createApi(f as unknown as typeof fetch).predict({ sample_id: "x", resolution: 64 }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.kind).toBe("validation");
    expect(err.message).toBe("resolution: Input should be 64 or 128");
    expect(err.requestId).toBe("rid-1");
  });

  it("maps a string detail (server-side field validation) verbatim", async () => {
    const f = vi.fn(() => json(422, { detail: "permeability must be strictly positive", request_id: "r" }));
    const err = await createApi(f as unknown as typeof fetch).predict({ permeability: [[1]] }).catch((e) => e);
    expect(err.message).toBe("permeability must be strictly positive");
  });

  it("treats network failure as backend unavailable", async () => {
    const f = vi.fn(() => Promise.reject(new TypeError("fetch failed")));
    const err = await createApi(f as unknown as typeof fetch).samples().catch((e) => e);
    expect(err.kind).toBe("unavailable");
    expect(err.message).toMatch(/could not be reached/);
  });

  it("treats the proxy's 502 as backend unavailable", async () => {
    const f = vi.fn(() => json(502, { detail: "The PDEFlow API is unavailable." }));
    const err = await createApi(f as unknown as typeof fetch).model().catch((e) => e);
    expect(err.kind).toBe("unavailable");
    expect(err.status).toBe(502);
  });

  it("reports non-JSON responses as malformed", async () => {
    const f = vi.fn(() => Promise.resolve(new Response("<html>oops</html>", { status: 200 })));
    const err = await createApi(f as unknown as typeof fetch).health().catch((e) => e);
    expect(err.kind).toBe("malformed");
  });

  it("maps 404 and 413", async () => {
    const api404 = createApi((() => json(404, { detail: "unknown sample_id 'x'" })) as unknown as typeof fetch);
    expect((await api404.sample("x", 64).catch((e) => e)).kind).toBe("not_found");
    const api413 = createApi((() => json(413, { detail: "too big" })) as unknown as typeof fetch);
    expect((await api413.predict({ permeability: [[1]] }).catch((e) => e)).kind).toBe("too_large");
  });

  it("encodes sample ids in the URL", async () => {
    const f = vi.fn((_u: string) => json(200, {}));
    await createApi(f as unknown as typeof fetch).sample("a/b", 128);
    expect(f.mock.calls[0][0]).toBe("/api/samples/a%2Fb?resolution=128");
  });

  it("formatDetail handles unexpected shapes", () => {
    expect(formatDetail(undefined)).toBe("Unexpected error response");
    expect(formatDetail([{ msg: "x" }])).toBe("x");
  });
});
