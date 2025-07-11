use actix::{Actor, StreamHandler, Handler, Message, ActorContext};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use actix_web_actors::ws;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::database::embeddings::FaceEmbedding;

#[derive(Message, Serialize, Deserialize)]
#[rtype(result = "()")]
pub enum WsMessage {
    FaceDetected(FaceEmbedding),
    FaceUpdated(FaceEmbedding),
    FaceDeleted(String),
    Error(String),
}

pub struct WsConnection {
    id: String,
    tx: broadcast::Sender<WsMessage>,
}

impl Actor for WsConnection {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        let mut rx = self.tx.subscribe();
        let addr = ctx.address();

        actix_web::rt::spawn(async move {
            while let Ok(msg) = rx.recv().await {
                addr.do_send(msg);
            }
        });
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for WsConnection {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                println!("Received message: {}", text);
            }
            Ok(ws::Message::Close(reason)) => {
                ctx.close(reason);
                ctx.stop();
            }
            _ => (),
        }
    }
}

impl Handler<WsMessage> for WsConnection {
    type Result = ();

    fn handle(&mut self, msg: WsMessage, ctx: &mut Self::Context) {
        if let Ok(data) = serde_json::to_string(&msg) {
            ctx.text(data);
        }
    }
}

pub struct WsManager {
    connections: HashMap<String, broadcast::Sender<WsMessage>>,
}

impl WsManager {
    pub fn new() -> Self {
        Self {
            connections: HashMap::new(),
        }
    }

    pub fn create_connection(&mut self) -> (String, broadcast::Sender<WsMessage>) {
        let id = Uuid::new_v4().to_string();
        let (tx, _) = broadcast::channel(100);
        self.connections.insert(id.clone(), tx.clone());
        (id, tx)
    }

    pub fn remove_connection(&mut self, id: &str) {
        self.connections.remove(id);
    }

    pub fn broadcast(&self, msg: WsMessage) {
        for tx in self.connections.values() {
            let _ = tx.send(msg.clone());
        }
    }
}

pub async fn ws_handler(
    req: HttpRequest,
    stream: web::Payload,
    manager: web::Data<Arc<tokio::sync::Mutex<WsManager>>>,
) -> Result<HttpResponse, Error> {
    let mut ws_manager = manager.lock().await;
    let (id, tx) = ws_manager.create_connection();

    let ws = WsConnection { id, tx };
    let resp = ws::start(ws, &req, stream)?;
    Ok(resp)
}

pub async fn notify_face_detected(
    manager: &Arc<tokio::sync::Mutex<WsManager>>,
    face: FaceEmbedding,
) {
    let ws_manager = manager.lock().await;
    ws_manager.broadcast(WsMessage::FaceDetected(face));
}

pub async fn notify_face_updated(
    manager: &Arc<tokio::sync::Mutex<WsManager>>,
    face: FaceEmbedding,
) {
    let ws_manager = manager.lock().await;
    ws_manager.broadcast(WsMessage::FaceUpdated(face));
}

pub async fn notify_face_deleted(
    manager: &Arc<tokio::sync::Mutex<WsManager>>,
    face_id: String,
) {
    let ws_manager = manager.lock().await;
    ws_manager.broadcast(WsMessage::FaceDeleted(face_id));
}

pub async fn notify_error(
    manager: &Arc<tokio::sync::Mutex<WsManager>>,
    error: String,
) {
    let ws_manager = manager.lock().await;
    ws_manager.broadcast(WsMessage::Error(error));
} 