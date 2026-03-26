# Stage 1 — build
FROM rust:1-slim AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

# Stage 2 — minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/sockets /usr/local/bin/sockets
EXPOSE 9001
CMD ["sockets"]
