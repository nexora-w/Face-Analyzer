use yew::prelude::*;
use yew_router::prelude::*;
use gloo_net::http::Request;
use gloo_file::File;
use web_sys::{HtmlInputElement, FileList};
use wasm_bindgen::{JsCast, UnwrapThrowExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Route definition
#[derive(Clone, Routable, PartialEq)]
pub enum Route {
    #[at("/")]
    Home,
    #[at("/faces")]
    Faces,
    #[at("/faces/:id")]
    FaceDetails { id: String },
    #[at("/settings")]
    Settings,
}

// API types
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Face {
    face_id: String,
    name: Option<String>,
    tags: Vec<String>,
    confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    min_confidence: f32,
    include_embeddings: bool,
    auto_cleanup_days: i32,
}

// Main app component
pub struct App {
    faces: Vec<Face>,
    settings: Settings,
    loading: bool,
    error: Option<String>,
}

pub enum Msg {
    LoadFaces,
    FacesLoaded(Vec<Face>),
    UploadFace(File),
    FaceUploaded(Face),
    DeleteFace(String),
    FaceDeleted(String),
    UpdateSettings(Settings),
    Error(String),
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        ctx.link().send_message(Msg::LoadFaces);
        
        Self {
            faces: Vec::new(),
            settings: Settings {
                min_confidence: 0.8,
                include_embeddings: false,
                auto_cleanup_days: 30,
            },
            loading: true,
            error: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::LoadFaces => {
                self.loading = true;
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    match Request::get("/api/v1/faces")
                        .send()
                        .await
                        .and_then(|resp| resp.json::<Vec<Face>>().await)
                    {
                        Ok(faces) => link.send_message(Msg::FacesLoaded(faces)),
                        Err(err) => link.send_message(Msg::Error(err.to_string())),
                    }
                });
                false
            }
            Msg::FacesLoaded(faces) => {
                self.faces = faces;
                self.loading = false;
                true
            }
            Msg::UploadFace(file) => {
                self.loading = true;
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let form_data = web_sys::FormData::new().unwrap();
                    form_data.append_with_blob("file", &file.into()).unwrap();

                    match Request::post("/api/v1/analyze")
                        .body(form_data)
                        .send()
                        .await
                        .and_then(|resp| resp.json::<Face>().await)
                    {
                        Ok(face) => link.send_message(Msg::FaceUploaded(face)),
                        Err(err) => link.send_message(Msg::Error(err.to_string())),
                    }
                });
                false
            }
            Msg::FaceUploaded(face) => {
                self.faces.push(face);
                self.loading = false;
                true
            }
            Msg::DeleteFace(id) => {
                self.loading = true;
                let link = ctx.link().clone();
                wasm_bindgen_futures::spawn_local(async move {
                    match Request::delete(&format!("/api/v1/faces/{}", id))
                        .send()
                        .await
                    {
                        Ok(_) => link.send_message(Msg::FaceDeleted(id)),
                        Err(err) => link.send_message(Msg::Error(err.to_string())),
                    }
                });
                false
            }
            Msg::FaceDeleted(id) => {
                self.faces.retain(|face| face.face_id != id);
                self.loading = false;
                true
            }
            Msg::UpdateSettings(settings) => {
                self.settings = settings;
                true
            }
            Msg::Error(error) => {
                self.error = Some(error);
                self.loading = false;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <BrowserRouter>
                <div class="app">
                    <nav class="navbar">
                        <Link<Route> to={Route::Home}>{ "Home" }</Link<Route>>
                        <Link<Route> to={Route::Faces}>{ "Faces" }</Link<Route>>
                        <Link<Route> to={Route::Settings}>{ "Settings" }</Link<Route>>
                    </nav>

                    {if let Some(error) = &self.error {
                        html! {
                            <div class="error-banner">
                                { error }
                                <button onclick={ctx.link().callback(|_| Msg::Error(String::new()))}>
                                    { "✕" }
                                </button>
                            </div>
                        }
                    } else {
                        html! {}
                    }}

                    <main>
                        <Switch<Route> render={switch} />
                    </main>

                    {if self.loading {
                        html! {
                            <div class="loading-overlay">
                                <div class="spinner"></div>
                            </div>
                        }
                    } else {
                        html! {}
                    }}
                </div>
            </BrowserRouter>
        }
    }
}

// Route switch function
fn switch(routes: Route) -> Html {
    match routes {
        Route::Home => html! { <Home /> },
        Route::Faces => html! { <FacesList /> },
        Route::FaceDetails { id } => html! { <FaceDetails id={id} /> },
        Route::Settings => html! { <Settings /> },
    }
}

// Home component
#[function_component(Home)]
fn home() -> Html {
    let onupload = Callback::from(|files: FileList| {
        if let Some(file) = files.get(0) {
            // Handle file upload
        }
    });

    html! {
        <div class="home">
            <h1>{ "Face Analyzer" }</h1>
            <div class="upload-section">
                <label for="file-upload" class="upload-button">
                    { "Upload Image" }
                </label>
                <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onchange={move |e: Event| {
                        let input: HtmlInputElement = e.target_unchecked_into();
                        if let Some(files) = input.files() {
                            onupload.emit(files);
                        }
                    }}
                />
            </div>
        </div>
    }
}

// Faces list component
#[function_component(FacesList)]
fn faces_list() -> Html {
    html! {
        <div class="faces-list">
            <h2>{ "Detected Faces" }</h2>
            // Face grid will be populated here
        </div>
    }
}

// Face details component
#[derive(Properties, PartialEq)]
struct FaceDetailsProps {
    id: String,
}

#[function_component(FaceDetails)]
fn face_details(props: &FaceDetailsProps) -> Html {
    html! {
        <div class="face-details">
            <h2>{ format!("Face Details: {}", props.id) }</h2>
            // Face details will be displayed here
        </div>
    }
}

// Settings component
#[function_component(Settings)]
fn settings() -> Html {
    html! {
        <div class="settings">
            <h2>{ "Settings" }</h2>
            <form>
                <div class="form-group">
                    <label>{ "Minimum Confidence" }</label>
                    <input type="range" min="0" max="1" step="0.1" />
                </div>
                <div class="form-group">
                    <label>{ "Include Embeddings" }</label>
                    <input type="checkbox" />
                </div>
                <div class="form-group">
                    <label>{ "Auto Cleanup (days)" }</label>
                    <input type="number" min="1" />
                </div>
                <button type="submit">{ "Save Settings" }</button>
            </form>
        </div>
    }
} 