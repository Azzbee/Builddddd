import { describe, expect, it } from "vitest";

import nextConfig, { securityHeaders } from "../next.config";

describe("web security headers", () => {
  it("builds a traced standalone production server", () => {
    expect(nextConfig.output).toBe("standalone");
  });

  it("does not advertise the application framework", () => {
    expect(nextConfig.poweredByHeader).toBe(false);
  });

  it("applies the policy to every route", async () => {
    expect(nextConfig.headers).toBeTypeOf("function");
    const rules = await nextConfig.headers!();

    expect(rules).toHaveLength(1);
    expect(rules[0].source).toBe("/:path*");
    expect(rules[0].headers).toEqual([...securityHeaders]);
  });

  it("blocks embedding and unnecessary browser capabilities", () => {
    const headers = Object.fromEntries(
      securityHeaders.map(({ key, value }) => [key, value]),
    );

    expect(headers["Content-Security-Policy"]).toContain(
      "frame-ancestors 'none'",
    );
    expect(headers["Content-Security-Policy"]).toContain(
      "frame-src 'self' blob:",
    );
    expect(headers["Content-Security-Policy"]).toContain("object-src 'none'");
    expect(headers["X-Content-Type-Options"]).toBe("nosniff");
    expect(headers["Permissions-Policy"]).toContain("camera=()");
  });
});
