/// WebSocket server for the MLOps prediction stream.
///
/// Each connected client receives a JSON event whenever the FastAPI server
/// publishes a prediction result.  The server itself just echoes messages back
/// to all connected peers (broadcast pattern).  In production the FastAPI
/// server would POST to /broadcast, which this service forwards to all clients.
///
/// Default listen address: ws://0.0.0.0:9001
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::broadcast;
use tokio_tungstenite::{accept_async, tungstenite::Message};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictionEvent {
    #[serde(rename = "type")]
    event_type: String,
    hazardous: bool,
    hazardous_probability: f64,
    miss_distance_km: f64,
}

// ── Broadcast channel ─────────────────────────────────────────────────────────

type Tx = broadcast::Sender<String>;

// ── Connection handler ────────────────────────────────────────────────────────

async fn handle_connection(stream: TcpStream, addr: SocketAddr, tx: Tx) {
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            eprintln!("WS handshake error from {addr}: {e}");
            return;
        }
    };

    println!("Client connected: {addr}");
    let mut rx = tx.subscribe();
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Send a welcome message
    let welcome = serde_json::json!({
        "type": "connected",
        "message": "MLOps prediction stream active"
    })
    .to_string();
    let _ = ws_sender.send(Message::Text(welcome.into())).await;

    loop {
        tokio::select! {
            // Forward broadcast messages to this client
            Ok(msg) = rx.recv() => {
                if ws_sender.send(Message::Text(msg.into())).await.is_err() {
                    break;
                }
            }
            // Echo messages from this client back as broadcast (demo behaviour)
            Some(Ok(msg)) = ws_receiver.next() => {
                match msg {
                    Message::Text(text) => {
                        let _ = tx.send(text.to_string());
                    }
                    Message::Ping(data) => {
                        let _ = ws_sender.send(Message::Pong(data)).await;
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            else => break,
        }
    }

    println!("Client disconnected: {addr}");
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let addr = "0.0.0.0:9001";
    let listener = TcpListener::bind(addr).await.expect("Failed to bind");
    println!("WebSocket server listening on ws://{addr}");

    // Broadcast channel with capacity for 256 in-flight messages
    let (tx, _rx) = broadcast::channel::<String>(256);

    loop {
        let (stream, peer_addr) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!("Accept error: {e}");
                continue;
            }
        };

        let tx_clone = tx.clone();
        tokio::spawn(handle_connection(stream, peer_addr, tx_clone));
    }
}
