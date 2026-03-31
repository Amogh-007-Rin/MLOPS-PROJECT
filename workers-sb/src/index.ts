import express from "express";
import type { Request, Response } from "express";
import "dotenv/config";
import { createClient } from "redis";

const app = express();
const port = process.env.PORT ?? "3000";
const REDIS_URL = process.env.REDIS_URL ?? "redis://localhost:6379/0";

// ── Redis client ──────────────────────────────────────────────────────────────

const redisClient = createClient({ url: REDIS_URL });

redisClient.on("error", (err) => console.error("Redis error:", err));

redisClient.connect().then(() => {
    console.log("Redis connected.");
}).catch((err) => {
    console.error("WARNING: Redis unavailable —", err.message);
});

// ── Routes ────────────────────────────────────────────────────────────────────

app.get("/health", function (_req: Request, res: Response) {
    res.status(200).json({ message: "worker is up and running", running: true });
});

// GET /cache/stats — how many prediction keys are cached
app.get("/cache/stats", async function (_req: Request, res: Response) {
    try {
        const keys = await redisClient.keys("pred:*");
        res.status(200).json({ cached_predictions: keys.length });
    } catch (err) {
        res.status(503).json({ error: "Redis unavailable" });
    }
});

// DELETE /cache/flush — evict all prediction cache entries
app.delete("/cache/flush", async function (_req: Request, res: Response) {
    try {
        const keys = await redisClient.keys("pred:*");
        if (keys.length > 0) {
            await redisClient.del(keys);
        }
        res.status(200).json({ flushed: keys.length });
    } catch (err) {
        res.status(503).json({ error: "Redis unavailable" });
    }
});

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(port, function () {
    console.log(`Worker running at: http://localhost:${port}`);
    console.log(`Health check     : http://localhost:${port}/health`);
    console.log(`Cache stats      : http://localhost:${port}/cache/stats`);
    console.log(`Cache flush      : DELETE http://localhost:${port}/cache/flush`);
});

export { app };
