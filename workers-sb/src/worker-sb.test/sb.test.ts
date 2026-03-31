/**
 * Worker service tests.
 *
 * Run with:
 *   npx jest  (after adding jest + supertest to devDependencies)
 *
 * Or directly via ts-jest:
 *   npx ts-jest --testPathPattern sb.test.ts
 */

import request from "supertest";
import { app } from "../index";

// ── Health ────────────────────────────────────────────────────────────────────

describe("GET /health", () => {
    it("returns 200 with running: true", async () => {
        const res = await request(app).get("/health");
        expect(res.status).toBe(200);
        expect(res.body.running).toBe(true);
        expect(res.body.message).toBeDefined();
    });
});

// ── Cache stats ───────────────────────────────────────────────────────────────

describe("GET /cache/stats", () => {
    it("returns 200 with cached_predictions count when Redis is available", async () => {
        const res = await request(app).get("/cache/stats");
        // If Redis is up, we expect a valid count; if not, a 503
        if (res.status === 200) {
            expect(typeof res.body.cached_predictions).toBe("number");
            expect(res.body.cached_predictions).toBeGreaterThanOrEqual(0);
        } else {
            expect(res.status).toBe(503);
            expect(res.body.error).toBeDefined();
        }
    });
});

// ── Cache flush ───────────────────────────────────────────────────────────────

describe("DELETE /cache/flush", () => {
    it("returns 200 with flushed count when Redis is available", async () => {
        const res = await request(app).delete("/cache/flush");
        if (res.status === 200) {
            expect(typeof res.body.flushed).toBe("number");
            expect(res.body.flushed).toBeGreaterThanOrEqual(0);
        } else {
            expect(res.status).toBe(503);
        }
    });
});

// ── Unknown routes ────────────────────────────────────────────────────────────

describe("Unknown routes", () => {
    it("returns 404 for unregistered paths", async () => {
        const res = await request(app).get("/not-a-real-route");
        expect(res.status).toBe(404);
    });
});
