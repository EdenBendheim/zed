#![allow(unused, dead_code)]
use std::collections::VecDeque;
use std::future::Future;
use std::{path::PathBuf, rc::Rc, sync::Arc};

use anyhow::{Context as _, Result};
use client::proto::ViewId;
use collections::HashMap;
use editor::{Bias, CompletionContext, CompletionProvider, ExcerptId};
use feature_flags::{FeatureFlagAppExt as _, NotebookFeatureFlag};
use futures::FutureExt;
use futures::channel::oneshot;
use futures::future::Shared;
use gpui::{
    AnyElement, App, Entity, EventEmitter, FocusHandle, Focusable, ListScrollEvent, ListState,
    Point, Task, WeakEntity, actions, list, prelude::*,
};
use jupyter_protocol::JupyterKernelspec;
use language::{Anchor, Buffer, CharScopeContext, CodeLabel, Language, LanguageRegistry, ToOffset};
use project::{CompletionDisplayOptions, CompletionResponse, Project, ProjectEntryId, ProjectPath};
use settings::{Settings as _, SettingsStore};
use ui::{CommonAnimationExt, Tooltip, prelude::*};
use workspace::item::{ItemEvent, SaveOptions, TabContentParams};
use workspace::searchable::SearchableItemHandle;
use workspace::{Item, ItemHandle, Pane, ProjectItem, ToolbarItemLocation};

use super::{Cell, CellEvent, CellPosition, MarkdownCellEvent, RenderableCell};

use nbformat::v4::CellId;
use nbformat::v4::Metadata as NotebookMetadata;
use serde_json;
use uuid::Uuid;

use crate::JupyterSettings;
use crate::components::{KernelPickerDelegate, KernelSelector};
use crate::kernels::{
    Kernel, KernelSession, KernelSpecification, KernelStatus, LocalKernelSpecification,
    NativeRunningKernel, RemoteRunningKernel,
};
use crate::repl_store::ReplStore;

use picker::Picker;
use runtimelib::{
    CompleteReply, CompleteRequest, ExecuteRequest, JupyterMessage, JupyterMessageContent,
    ReplyStatus,
};
use ui::PopoverMenuHandle;
use zed_actions::editor::{MoveDown, MoveUp};

actions!(
    notebook,
    [
        /// Opens a Jupyter notebook file.
        OpenNotebook,
        /// Runs all cells in the notebook.
        RunAll,
        /// Runs the current cell.
        Run,
        /// Clears all cell outputs.
        ClearOutputs,
        /// Moves the current cell up.
        MoveCellUp,
        /// Moves the current cell down.
        MoveCellDown,
        /// Adds a new markdown cell.
        AddMarkdownBlock,
        /// Adds a new code cell.
        AddCodeBlock,
        /// Duplicates the currently selected cell.
        DuplicateCell,
        /// Deletes the currently selected cell.
        DeleteCell,
        /// Restarts the kernel.
        RestartKernel,
        /// Interrupts the current execution.
        InterruptKernel,
    ]
);

pub(crate) const MAX_TEXT_BLOCK_WIDTH: f32 = 9999.0;
pub(crate) const SMALL_SPACING_SIZE: f32 = 8.0;
pub(crate) const MEDIUM_SPACING_SIZE: f32 = 12.0;
pub(crate) const LARGE_SPACING_SIZE: f32 = 16.0;
pub(crate) const GUTTER_WIDTH: f32 = 19.0;
pub(crate) const CODE_BLOCK_INSET: f32 = MEDIUM_SPACING_SIZE;
pub(crate) const CONTROL_SIZE: f32 = 20.0;

fn notebook_editor_enabled(cx: &App) -> bool {
    JupyterSettings::notebooks_enabled(cx)
        || cx.has_flag::<NotebookFeatureFlag>()
        || std::env::var("LOCAL_NOTEBOOK_DEV").is_ok()
}

pub fn init(cx: &mut App) {
    if notebook_editor_enabled(cx) {
        workspace::register_project_item::<NotebookEditor>(cx);
    }

    cx.observe_flag::<NotebookFeatureFlag, _>({
        move |is_enabled, cx| {
            if is_enabled || JupyterSettings::notebooks_enabled(cx) {
                workspace::register_project_item::<NotebookEditor>(cx);
            } else {
                // todo: there is no way to unregister a project item, so if the feature flag
                // gets turned off they need to restart Zed.
            }
        }
    })
    .detach();

    cx.observe_global::<SettingsStore>(move |cx| {
        if notebook_editor_enabled(cx) {
            workspace::register_project_item::<NotebookEditor>(cx);
        }
    })
    .detach();
}

pub struct NotebookEditor {
    languages: Arc<LanguageRegistry>,
    project: Entity<Project>,
    worktree_id: project::WorktreeId,

    focus_handle: FocusHandle,
    notebook_item: Entity<NotebookItem>,
    notebook_language: Shared<Task<Option<Arc<Language>>>>,

    remote_id: Option<ViewId>,
    cell_list: ListState,

    selected_cell_index: usize,
    cell_order: Vec<CellId>,
    original_cell_order: Vec<CellId>,
    cell_map: HashMap<CellId, Cell>,
    kernel: Kernel,
    kernel_specification: Option<KernelSpecification>,
    execution_requests: HashMap<String, CellId>,
    completion_requests: HashMap<String, PendingNotebookCompletionRequest>,
    run_all_queue: VecDeque<CellId>,
    run_all_in_progress: bool,
    kernel_picker_handle: PopoverMenuHandle<Picker<KernelPickerDelegate>>,
    nav_history: Option<workspace::ItemNavHistory>,
    last_error: Option<SharedString>,
    notebook_metadata_dirty: bool,
}

struct PendingNotebookCompletionRequest {
    sender: oneshot::Sender<CompleteReply>,
}

struct NotebookCellCompletionProvider {
    notebook_editor: WeakEntity<NotebookEditor>,
    cell_id: CellId,
}

impl NotebookCellCompletionProvider {
    fn new(notebook_editor: WeakEntity<NotebookEditor>, cell_id: CellId) -> Self {
        Self {
            notebook_editor,
            cell_id,
        }
    }

    fn byte_offset_to_character_offset(text: &str, byte_offset: usize) -> usize {
        let mut clipped_offset = byte_offset.min(text.len());
        while clipped_offset > 0 && !text.is_char_boundary(clipped_offset) {
            clipped_offset = clipped_offset.saturating_sub(1);
        }
        text[..clipped_offset].chars().count()
    }

    fn character_offset_to_byte_offset(text: &str, character_offset: usize) -> usize {
        if character_offset == 0 {
            return 0;
        }

        text.char_indices()
            .nth(character_offset)
            .map(|(byte_offset, _)| byte_offset)
            .unwrap_or(text.len())
    }

    fn replacement_range_from_reply(
        snapshot: &language::TextBufferSnapshot,
        code: &str,
        completion_reply: &CompleteReply,
    ) -> std::ops::Range<Anchor> {
        let mut start_offset =
            Self::character_offset_to_byte_offset(code, completion_reply.cursor_start);
        let mut end_offset =
            Self::character_offset_to_byte_offset(code, completion_reply.cursor_end);
        if start_offset > end_offset {
            std::mem::swap(&mut start_offset, &mut end_offset);
        }

        let start_offset = snapshot.clip_offset(start_offset, Bias::Left);
        let end_offset = snapshot.clip_offset(end_offset, Bias::Right);

        snapshot.anchor_before(start_offset)..snapshot.anchor_after(end_offset.max(start_offset))
    }
}

impl CompletionProvider for NotebookCellCompletionProvider {
    fn completions(
        &self,
        _excerpt_id: ExcerptId,
        buffer: &Entity<Buffer>,
        buffer_position: Anchor,
        _trigger: CompletionContext,
        _window: &mut Window,
        cx: &mut Context<editor::Editor>,
    ) -> Task<Result<Vec<CompletionResponse>>> {
        let Some(notebook_editor) = self.notebook_editor.upgrade() else {
            return Task::ready(Ok(Vec::new()));
        };

        let snapshot = buffer.read(cx).text_snapshot();
        let code = snapshot.text();
        let buffer_offset = buffer_position.to_offset(&snapshot);
        let cursor_position = Self::byte_offset_to_character_offset(&code, buffer_offset);
        let request_code = code.clone();
        let completion_request = CompleteRequest {
            code: request_code,
            cursor_pos: cursor_position,
        };

        let completion_receiver = match notebook_editor.update(cx, |notebook_editor, cx| {
            notebook_editor.request_cell_completions(self.cell_id.clone(), completion_request, cx)
        }) {
            Ok(completion_receiver) => completion_receiver,
            Err(error) => return Task::ready(Err(error)),
        };

        cx.background_executor().spawn(async move {
            let completion_reply = completion_receiver
                .await
                .context("kernel completion request was cancelled")?;

            if completion_reply.status != ReplyStatus::Ok {
                return Err(anyhow::anyhow!("kernel completion request failed"));
            }

            let replacement_range =
                Self::replacement_range_from_reply(&snapshot, &code, &completion_reply);

            let completions = completion_reply
                .matches
                .into_iter()
                .map(|completion_match| project::Completion {
                    replace_range: replacement_range.clone(),
                    new_text: completion_match.clone(),
                    label: CodeLabel::plain(completion_match, None),
                    documentation: None,
                    source: project::CompletionSource::Custom,
                    icon_path: None,
                    match_start: None,
                    snippet_deduplication_key: None,
                    insert_text_mode: None,
                    confirm: None,
                })
                .collect();

            Ok(vec![CompletionResponse {
                completions,
                display_options: CompletionDisplayOptions::default(),
                is_incomplete: false,
            }])
        })
    }

    fn is_completion_trigger(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        text: &str,
        trigger_in_words: bool,
        cx: &mut Context<editor::Editor>,
    ) -> bool {
        let mut characters = text.chars();
        let typed_character = if let Some(character) = characters.next() {
            character
        } else {
            return false;
        };
        if characters.next().is_some() {
            return false;
        }

        let snapshot = buffer.read(cx).snapshot();
        let classifier = snapshot
            .char_classifier_at(position)
            .scope_context(Some(CharScopeContext::Completion));
        if trigger_in_words && classifier.is_word(typed_character) {
            return true;
        }

        buffer.read(cx).completion_triggers().contains(text) || typed_character == '.'
    }
}

impl NotebookEditor {
    pub fn new(
        project: Entity<Project>,
        notebook_item: Entity<NotebookItem>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let focus_handle = cx.focus_handle();

        let languages = project.read(cx).languages().clone();
        let worktree_id = notebook_item.read(cx).project_path.worktree_id;

        let notebook_language = notebook_item.read(cx).notebook_language();
        let notebook_language = cx
            .spawn_in(window, async move |_, _| notebook_language.await)
            .shared();

        let mut cell_order = vec![]; // Vec<CellId>
        let mut cell_map = HashMap::default(); // HashMap<CellId, Cell>
        let notebook_editor = cx.entity().downgrade();

        let cell_count = notebook_item.read(cx).notebook.cells.len();
        for index in 0..cell_count {
            let cell = notebook_item.read(cx).notebook.cells[index].clone();
            let cell_id = cell.id();
            cell_order.push(cell_id.clone());
            let cell_entity = Cell::load(&cell, &languages, notebook_language.clone(), window, cx);
            Self::subscribe_to_cell_events(&notebook_editor, &cell_id, &cell_entity, cx);
            cell_map.insert(cell_id.clone(), cell_entity);
        }

        let cell_count = cell_order.len();
        let cell_list = ListState::new(cell_count, gpui::ListAlignment::Top, px(1000.));

        let mut editor = Self {
            project,
            languages: languages.clone(),
            worktree_id,
            focus_handle,
            notebook_item: notebook_item.clone(),
            notebook_language,
            remote_id: None,
            cell_list,
            selected_cell_index: 0,
            cell_order: cell_order.clone(),
            original_cell_order: cell_order.clone(),
            cell_map: cell_map.clone(),
            kernel: Kernel::Shutdown, // TODO: use recommended kernel after the implementation is done in repl
            kernel_specification: None,
            execution_requests: HashMap::default(),
            completion_requests: HashMap::default(),
            run_all_queue: VecDeque::default(),
            run_all_in_progress: false,
            kernel_picker_handle: PopoverMenuHandle::default(),
            nav_history: None,
            last_error: None,
            notebook_metadata_dirty: false,
        };
        ReplStore::global(cx).update(cx, |store, cx| {
            store.refresh_kernelspecs(cx).detach();
            store
                .refresh_python_kernelspecs(worktree_id, &editor.project, cx)
                .detach();
        });
        editor.launch_kernel(window, cx);
        editor.refresh_language(cx);

        cx.subscribe(&notebook_item, |this, _item, _event, cx| {
            this.refresh_language(cx);
        })
        .detach();

        editor
    }

    fn subscribe_to_cell_events(
        notebook_editor: &WeakEntity<Self>,
        cell_id: &CellId,
        cell_entity: &Cell,
        cx: &mut Context<Self>,
    ) {
        match cell_entity {
            Cell::Code(code_cell) => {
                let cell_id_for_focus = cell_id.clone();
                cx.subscribe(code_cell, move |this, _cell, event, cx| match event {
                    CellEvent::Run(cell_id) => this.execute_cell(cell_id.clone(), cx),
                    CellEvent::FocusedIn(_) => {
                        if let Some(index) = this
                            .cell_order
                            .iter()
                            .position(|id| id == &cell_id_for_focus)
                        {
                            this.selected_cell_index = index;
                            cx.notify();
                        }
                    }
                })
                .detach();

                let cell_id_for_editor = cell_id.clone();
                let editor = code_cell.read(cx).editor().clone();
                let completion_provider: Rc<dyn CompletionProvider> = Rc::new(
                    NotebookCellCompletionProvider::new(notebook_editor.clone(), cell_id.clone()),
                );
                editor.update(cx, |editor, _cx| {
                    editor.set_completion_provider(Some(completion_provider));
                });
                cx.subscribe(&editor, move |this, _editor, event, cx| {
                    if let editor::EditorEvent::Focused = event
                        && let Some(index) = this
                            .cell_order
                            .iter()
                            .position(|id| id == &cell_id_for_editor)
                    {
                        this.selected_cell_index = index;
                        cx.notify();
                    }
                })
                .detach();
            }
            Cell::Markdown(markdown_cell) => {
                cx.subscribe(
                    markdown_cell,
                    move |_this, cell, event: &MarkdownCellEvent, cx| match event {
                        MarkdownCellEvent::FinishedEditing | MarkdownCellEvent::Run(_) => {
                            cell.update(cx, |cell, cx| {
                                cell.reparse_markdown(cx);
                            });
                        }
                    },
                )
                .detach();

                let cell_id_for_editor = cell_id.clone();
                let editor = markdown_cell.read(cx).editor().clone();
                cx.subscribe(&editor, move |this, _editor, event, cx| {
                    if let editor::EditorEvent::Focused = event
                        && let Some(index) = this
                            .cell_order
                            .iter()
                            .position(|id| id == &cell_id_for_editor)
                    {
                        this.selected_cell_index = index;
                        cx.notify();
                    }
                })
                .detach();
            }
            Cell::Raw(_) => {}
        }
    }

    fn refresh_language(&mut self, cx: &mut Context<Self>) {
        let notebook_language = self.notebook_item.read(cx).notebook_language();
        let task = cx.spawn(async move |this, cx| {
            let language = notebook_language.await;
            if let Some(this) = this.upgrade() {
                this.update(cx, |this, cx| {
                    for cell in this.cell_map.values() {
                        if let Cell::Code(code_cell) = cell {
                            code_cell.update(cx, |cell, cx| {
                                cell.set_language(language.clone(), cx);
                            });
                        }
                    }
                });
            }
            language
        });
        self.notebook_language = task.shared();
    }

    fn has_structural_changes(&self) -> bool {
        self.cell_order != self.original_cell_order
    }

    fn has_content_changes(&self, cx: &App) -> bool {
        self.cell_map.values().any(|cell| cell.is_dirty(cx))
    }

    fn has_metadata_changes(&self) -> bool {
        self.notebook_metadata_dirty
    }

    pub fn to_notebook(&self, cx: &App) -> nbformat::v4::Notebook {
        let cells: Vec<nbformat::v4::Cell> = self
            .cell_order
            .iter()
            .filter_map(|cell_id| {
                self.cell_map
                    .get(cell_id)
                    .map(|cell| cell.to_nbformat_cell(cx))
            })
            .collect();

        let metadata = self.notebook_item.read(cx).notebook.metadata.clone();

        nbformat::v4::Notebook {
            metadata,
            nbformat: 4,
            nbformat_minor: 5,
            cells,
        }
    }

    pub fn mark_as_saved(&mut self, cx: &mut Context<Self>) {
        self.original_cell_order = self.cell_order.clone();

        for cell in self.cell_map.values() {
            match cell {
                Cell::Code(code_cell) => {
                    code_cell.update(cx, |code_cell, cx| {
                        let editor = code_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                        code_cell.mark_saved();
                    });
                }
                Cell::Markdown(markdown_cell) => {
                    markdown_cell.update(cx, |markdown_cell, cx| {
                        let editor = markdown_cell.editor();
                        editor.update(cx, |editor, cx| {
                            editor.buffer().update(cx, |buffer, cx| {
                                if let Some(buf) = buffer.as_singleton() {
                                    buf.update(cx, |b, cx| {
                                        let version = b.version();
                                        b.did_save(version, None, cx);
                                    });
                                }
                            });
                        });
                    });
                }
                Cell::Raw(_) => {}
            }
        }
        self.notebook_metadata_dirty = false;
        cx.notify();
    }

    fn default_python_kernel_specification() -> KernelSpecification {
        KernelSpecification::Jupyter(LocalKernelSpecification {
            name: "python3".to_string(),
            path: PathBuf::from("python3"),
            kernelspec: JupyterKernelspec {
                argv: vec![
                    "python3".to_string(),
                    "-m".to_string(),
                    "ipykernel_launcher".to_string(),
                    "-f".to_string(),
                    "{connection_file}".to_string(),
                ],
                display_name: "Python 3".to_string(),
                language: "python".to_string(),
                interrupt_mode: None,
                metadata: None,
                env: None,
            },
        })
    }

    fn resolve_notebook_kernel_specification(&self, cx: &App) -> Option<KernelSpecification> {
        let kernelspec_metadata = self
            .notebook_item
            .read(cx)
            .notebook
            .metadata
            .kernelspec
            .clone();
        let kernelspec_value = kernelspec_metadata.and_then(|metadata| {
            serde_json::to_value(metadata)
                .ok()
                .and_then(|value| value.as_object().cloned())
        });

        let preferred_kernel_name = kernelspec_value
            .as_ref()
            .and_then(|value| value.get("name"))
            .and_then(|value| value.as_str())
            .map(|value| value.to_lowercase());
        let preferred_kernel_language = kernelspec_value
            .as_ref()
            .and_then(|value| value.get("language"))
            .and_then(|value| value.as_str())
            .map(|value| value.to_lowercase());

        let store = ReplStore::global(cx);
        let store = store.read(cx);

        if let Some(preferred_kernel_name) = preferred_kernel_name
            && let Some(specification) = store
                .kernel_specifications_for_worktree(self.worktree_id)
                .find(|specification| {
                    specification.name().to_string().to_lowercase() == preferred_kernel_name
                })
        {
            return Some(specification.clone());
        }

        if let Some(preferred_kernel_language) = preferred_kernel_language
            && let Some(specification) = store
                .kernel_specifications_for_worktree(self.worktree_id)
                .find(|specification| {
                    specification.language().to_string().to_lowercase() == preferred_kernel_language
                })
        {
            return Some(specification.clone());
        }

        None
    }

    fn launch_kernel(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let specification = self
            .kernel_specification
            .clone()
            .or_else(|| self.resolve_notebook_kernel_specification(cx))
            .unwrap_or_else(Self::default_python_kernel_specification);

        self.launch_kernel_with_spec(specification, window, cx);
    }

    fn launch_kernel_with_spec(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let entity_id = cx.entity_id();
        let working_directory = self
            .notebook_item
            .read(cx)
            .path
            .parent()
            .map(std::path::Path::to_path_buf)
            .or_else(|| {
                self.project
                    .read(cx)
                    .worktrees(cx)
                    .next()
                    .map(|worktree| worktree.read(cx).abs_path().to_path_buf())
            })
            .unwrap_or_else(std::env::temp_dir);
        let fs = self.project.read(cx).fs().clone();
        let view = cx.entity();

        self.kernel_specification = Some(spec.clone());

        self.notebook_item.update(cx, |item, cx| {
            let kernel_name = spec.name().to_string();
            let language = spec.language().to_string();

            let display_name = match &spec {
                KernelSpecification::Jupyter(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::PythonEnv(s) => s.kernelspec.display_name.clone(),
                KernelSpecification::Remote(s) => s.kernelspec.display_name.clone(),
            };

            let kernelspec_json = serde_json::json!({
                "display_name": display_name,
                "name": kernel_name,
                "language": language
            });

            if let Ok(k) = serde_json::from_value(kernelspec_json) {
                item.notebook.metadata.kernelspec = Some(k);
                cx.emit(());
            }
        });
        self.notebook_metadata_dirty = true;

        let kernel_task = match spec {
            KernelSpecification::Jupyter(local_spec)
            | KernelSpecification::PythonEnv(local_spec) => NativeRunningKernel::new(
                local_spec,
                entity_id,
                working_directory,
                fs,
                view,
                window,
                cx,
            ),
            KernelSpecification::Remote(remote_spec) => {
                RemoteRunningKernel::new(remote_spec, working_directory, view, window, cx)
            }
        };

        let pending_kernel = cx
            .spawn(async move |this, cx| {
                let kernel = kernel_task.await;

                match kernel {
                    Ok(kernel) => {
                        this.update(cx, |editor, cx| {
                            editor.kernel = Kernel::RunningKernel(kernel);
                            editor.clear_last_error(cx);
                            cx.notify();
                        })
                        .ok();
                    }
                    Err(err) => {
                        this.update(cx, |editor, cx| {
                            editor.kernel = Kernel::ErroredLaunch(err.to_string());
                            editor.set_last_error(format!("Failed to launch kernel: {err}"), cx);
                        })
                        .ok();
                    }
                }
            })
            .shared();

        self.kernel = Kernel::StartingKernel(pending_kernel);
        cx.notify();
    }

    // Note: Python environments are only detected as kernels if ipykernel is installed.
    // Users need to run `pip install ipykernel` (or `uv pip install ipykernel`) in their
    // virtual environment for it to appear in the kernel selector.
    // This happens because we have an ipykernel check inside the function python_env_kernel_specification in mod.rs L:121

    fn change_kernel(
        &mut self,
        spec: KernelSpecification,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            kernel.force_shutdown(window, cx).detach();
        }

        self.cancel_pending_executions("Kernel changed before execution finished.", cx);

        self.launch_kernel_with_spec(spec, window, cx);
    }

    fn restart_kernel(&mut self, _: &RestartKernel, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(spec) = self.kernel_specification.clone() {
            if let Kernel::RunningKernel(kernel) = &mut self.kernel {
                kernel.force_shutdown(window, cx).detach();
            }

            self.cancel_pending_executions("Kernel restarted before execution finished.", cx);
            self.kernel = Kernel::Restarting;
            cx.notify();

            self.launch_kernel_with_spec(spec, window, cx);
        }
    }

    fn interrupt_kernel(
        &mut self,
        _: &InterruptKernel,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Kernel::RunningKernel(kernel) = &self.kernel {
            let interrupt_request = runtimelib::InterruptRequest {};
            let message: JupyterMessage = interrupt_request.into();
            if let Err(error) = kernel.request_tx().try_send(message) {
                self.set_last_error(format!("Failed to interrupt kernel: {error}"), cx);
                return;
            }
            self.clear_last_error(cx);
            cx.notify();
        } else {
            self.set_last_error("Kernel is not running.", cx);
        }
    }

    fn request_cell_completions(
        &mut self,
        cell_id: CellId,
        completion_request: CompleteRequest,
        cx: &mut Context<Self>,
    ) -> Result<oneshot::Receiver<CompleteReply>> {
        if !matches!(self.cell_map.get(&cell_id), Some(Cell::Code(_))) {
            anyhow::bail!("completion requested for non-code cell");
        }

        let message: JupyterMessage = completion_request.into();
        let request_id = message.header.msg_id.clone();
        let (sender, receiver) = oneshot::channel();

        self.completion_requests.insert(
            request_id.clone(),
            PendingNotebookCompletionRequest { sender },
        );

        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            if let Err(error) = kernel.request_tx().try_send(message) {
                self.completion_requests.remove(&request_id);
                self.set_last_error(format!("Failed to request completions: {error}"), cx);
                anyhow::bail!("failed to request completions: {error}");
            }
            self.clear_last_error(cx);
            return Ok(receiver);
        }

        self.completion_requests.remove(&request_id);
        self.set_last_error("Kernel is not running.", cx);
        anyhow::bail!("kernel is not running");
    }

    fn execute_cell(&mut self, cell_id: CellId, cx: &mut Context<Self>) {
        let code = if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            let editor = cell.read(cx).editor().clone();
            let buffer = editor.read(cx).buffer().read(cx);
            buffer
                .as_singleton()
                .map(|b| b.read(cx).text())
                .unwrap_or_default()
        } else {
            return;
        };

        if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
            cell.update(cx, |cell, cx| {
                if cell.has_outputs() {
                    cell.clear_outputs();
                }
                cell.start_execution();
                cx.notify();
            });
        }

        let request = ExecuteRequest {
            code,
            ..Default::default()
        };
        let message: JupyterMessage = request.into();
        let msg_id = message.header.msg_id.clone();

        self.execution_requests.insert(msg_id, cell_id.clone());

        if let Kernel::RunningKernel(kernel) = &mut self.kernel {
            if let Err(error) = kernel.request_tx().try_send(message) {
                self.execution_requests
                    .retain(|_request_id, request_cell_id| request_cell_id != &cell_id);
                self.set_last_error(format!("Failed to execute cell: {error}"), cx);
                if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                    let error_message = format!("Failed to execute cell: {error}");
                    cell.update(cx, |cell, cx| {
                        cell.finish_execution();
                        cell.push_message_output(error_message);
                        cx.notify();
                    });
                }
                self.execute_next_queued_cell(cx);
                return;
            }
            self.clear_last_error(cx);
        } else {
            self.execution_requests
                .retain(|_request_id, request_cell_id| request_cell_id != &cell_id);
            self.set_last_error("Kernel is not running.", cx);
            if let Some(Cell::Code(cell)) = self.cell_map.get(&cell_id) {
                cell.update(cx, |cell, cx| {
                    cell.finish_execution();
                    cell.push_message_output("Kernel is not running.");
                    cx.notify();
                });
            }
            self.execute_next_queued_cell(cx);
        }
    }

    fn has_outputs(&self, _window: &mut Window, cx: &mut Context<Self>) -> bool {
        self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).has_outputs()
            } else {
                false
            }
        })
    }

    fn clear_outputs(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    cell.clear_outputs();
                    cx.notify();
                });
            }
        }
        cx.notify();
    }

    fn run_cells(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.run_all_queue = self
            .cell_order
            .iter()
            .filter(|cell_id| matches!(self.cell_map.get(*cell_id), Some(Cell::Code(_))))
            .cloned()
            .collect();
        self.run_all_in_progress = true;
        self.execute_next_queued_cell(cx);
    }

    fn execute_next_queued_cell(&mut self, cx: &mut Context<Self>) {
        if !self.run_all_in_progress {
            return;
        }

        let has_running_code_cell = self.cell_map.values().any(|cell| {
            if let Cell::Code(code_cell) = cell {
                code_cell.read(cx).is_executing()
            } else {
                false
            }
        });

        if has_running_code_cell {
            return;
        }

        if let Some(next_cell_id) = self.run_all_queue.pop_front() {
            self.execute_cell(next_cell_id, cx);
        } else {
            self.run_all_in_progress = false;
            cx.notify();
        }
    }

    fn cancel_pending_executions(&mut self, reason: &str, cx: &mut Context<Self>) {
        self.execution_requests.clear();
        self.completion_requests.clear();
        self.run_all_queue.clear();
        self.run_all_in_progress = false;

        for cell in self.cell_map.values() {
            if let Cell::Code(code_cell) = cell {
                code_cell.update(cx, |cell, cx| {
                    if cell.is_executing() {
                        cell.finish_execution();
                        cell.push_message_output(reason.to_string());
                    }
                    cx.notify();
                });
            }
        }
    }

    fn set_last_error(&mut self, message: impl Into<SharedString>, cx: &mut Context<Self>) {
        self.last_error = Some(message.into());
        cx.notify();
    }

    fn clear_last_error(&mut self, cx: &mut Context<Self>) {
        if self.last_error.take().is_some() {
            cx.notify();
        }
    }

    fn run_current_cell(&mut self, _: &Run, window: &mut Window, cx: &mut Context<Self>) {
        self.run_all_in_progress = false;
        self.run_all_queue.clear();
        if let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() {
            if let Some(cell) = self.cell_map.get(&cell_id) {
                match cell {
                    Cell::Code(_) => {
                        self.execute_cell(cell_id, cx);
                    }
                    Cell::Markdown(markdown_cell) => {
                        // for markdown, finish editing and move to next cell
                        let is_editing = markdown_cell.read(cx).is_editing();
                        if is_editing {
                            markdown_cell.update(cx, |cell, cx| {
                                cell.run(cx);
                            });
                            // move to the next cell
                            // Discussion can be done on this default implementation
                            self.move_to_next_cell(window, cx);
                        }
                    }
                    Cell::Raw(_) => {}
                }
            }
        }
    }

    // Discussion can be done on this default implementation
    /// Moves focus to the next cell, or creates a new code cell if at the end
    fn move_to_next_cell(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.cell_order.is_empty() && self.selected_cell_index < self.cell_order.len() - 1 {
            self.selected_cell_index += 1;
            // focus the new cell's editor
            if let Some(cell_id) = self.cell_order.get(self.selected_cell_index) {
                if let Some(cell) = self.cell_map.get(cell_id) {
                    match cell {
                        Cell::Code(code_cell) => {
                            let editor = code_cell.read(cx).editor();
                            window.focus(&editor.focus_handle(cx), cx);
                        }
                        Cell::Markdown(markdown_cell) => {
                            // Don't auto-enter edit mode for next markdown cell
                            // Just select it
                        }
                        Cell::Raw(_) => {}
                    }
                }
            }
            cx.notify();
        } else {
            // in the end, could optionally create a new cell
            // For now, just stay on the current cell
        }
    }

    fn open_notebook(&mut self, _: &OpenNotebook, _window: &mut Window, cx: &mut Context<Self>) {
        cx.notify();
    }

    fn move_cell_up(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.selected_cell_index > 0 {
            self.cell_order
                .swap(self.selected_cell_index, self.selected_cell_index - 1);
            self.selected_cell_index -= 1;
            cx.notify();
        }
    }

    fn move_cell_down(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if !self.cell_order.is_empty() && self.selected_cell_index < self.cell_order.len() - 1 {
            self.cell_order
                .swap(self.selected_cell_index, self.selected_cell_index + 1);
            self.selected_cell_index += 1;
            cx.notify();
        }
    }

    fn empty_cell_metadata() -> nbformat::v4::CellMetadata {
        nbformat::v4::CellMetadata {
            id: None,
            collapsed: None,
            scrolled: None,
            deletable: None,
            editable: None,
            format: None,
            name: None,
            tags: None,
            jupyter: None,
            execution: None,
            additional: std::collections::HashMap::new(),
        }
    }

    fn insert_cell(
        &mut self,
        insert_index: usize,
        cell_id: CellId,
        cell: Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let notebook_editor = cx.entity().downgrade();
        Self::subscribe_to_cell_events(&notebook_editor, &cell_id, &cell, cx);
        if let Some(history) = self.nav_history.clone() {
            let maybe_editor = match &cell {
                Cell::Code(code_cell) => Some(code_cell.read(cx).editor().clone()),
                Cell::Markdown(markdown_cell) => Some(markdown_cell.read(cx).editor().clone()),
                Cell::Raw(_) => None,
            };
            if let Some(editor) = maybe_editor {
                editor.update(cx, |editor, cx| {
                    Item::set_nav_history(editor, history, window, cx)
                });
            }
        }
        self.cell_order.insert(insert_index, cell_id.clone());
        self.cell_map.insert(cell_id, cell);
        self.selected_cell_index = insert_index;
        self.cell_list.reset(self.cell_order.len());
        cx.notify();
    }

    fn add_markdown_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let new_cell_id: CellId = Uuid::new_v4().into();
        let languages = self.languages.clone();
        let metadata = Self::empty_cell_metadata();

        let markdown_cell = Cell::Markdown(cx.new(|cx| {
            super::MarkdownCell::new(
                new_cell_id.clone(),
                metadata,
                None,
                String::new(),
                languages,
                window,
                cx,
            )
        }));

        let insert_index = if self.cell_order.is_empty() {
            0
        } else {
            self.selected_cell_index + 1
        };
        self.insert_cell(insert_index, new_cell_id, markdown_cell, window, cx);
    }

    fn add_code_block(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let new_cell_id: CellId = Uuid::new_v4().into();
        let notebook_language = self.notebook_language.clone();
        let metadata = Self::empty_cell_metadata();

        let code_cell = Cell::Code(cx.new(|cx| {
            super::CodeCell::new(
                new_cell_id.clone(),
                metadata,
                String::new(),
                notebook_language,
                window,
                cx,
            )
        }));

        let insert_index = if self.cell_order.is_empty() {
            0
        } else {
            self.selected_cell_index + 1
        };
        self.insert_cell(insert_index, new_cell_id, code_cell, window, cx);
    }

    fn duplicate_selected_cell(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };
        let Some(cell) = self.cell_map.get(&cell_id) else {
            return;
        };

        let mut new_cell = cell.to_nbformat_cell(cx);
        let duplicated_cell_id: CellId = Uuid::new_v4().into();
        match &mut new_cell {
            nbformat::v4::Cell::Code {
                id,
                execution_count,
                outputs,
                ..
            } => {
                *id = duplicated_cell_id.clone();
                *execution_count = None;
                outputs.clear();
            }
            nbformat::v4::Cell::Markdown { id, .. } | nbformat::v4::Cell::Raw { id, .. } => {
                *id = duplicated_cell_id.clone();
            }
        }

        let duplicated_cell = Cell::load(
            &new_cell,
            &self.languages,
            self.notebook_language.clone(),
            _window,
            cx,
        );
        let insert_index = self.selected_cell_index + 1;
        self.insert_cell(
            insert_index,
            duplicated_cell_id,
            duplicated_cell,
            _window,
            cx,
        );
    }

    fn delete_selected_cell(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(cell_id) = self.cell_order.get(self.selected_cell_index).cloned() else {
            return;
        };

        self.cell_order
            .retain(|existing_cell_id| existing_cell_id != &cell_id);
        self.cell_map.remove(&cell_id);

        if self.cell_order.is_empty() {
            self.selected_cell_index = 0;
        } else if self.selected_cell_index >= self.cell_order.len() {
            self.selected_cell_index = self.cell_order.len() - 1;
        }

        self.cell_list.reset(self.cell_order.len());
        cx.notify();
    }

    fn cell_count(&self) -> usize {
        self.cell_map.len()
    }

    fn selected_index(&self) -> usize {
        self.selected_cell_index
    }

    fn selected_editor(&self, cx: &App) -> Option<Entity<editor::Editor>> {
        let selected_cell_id = self.cell_order.get(self.selected_cell_index)?;
        match self.cell_map.get(selected_cell_id)? {
            Cell::Code(code_cell) => Some(code_cell.read(cx).editor().clone()),
            Cell::Markdown(markdown_cell) => Some(markdown_cell.read(cx).editor().clone()),
            Cell::Raw(_) => None,
        }
    }

    pub fn set_selected_index(
        &mut self,
        index: usize,
        jump_to_index: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        // let previous_index = self.selected_cell_index;
        self.selected_cell_index = index;
        let current_index = self.selected_cell_index;

        // in the future we may have some `on_cell_change` event that we want to fire here

        if jump_to_index {
            self.jump_to_cell(current_index, window, cx);
        }
    }

    pub fn select_next(
        &mut self,
        _: &menu::SelectNext,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == count - 1 {
                count - 1
            } else {
                index + 1
            };
            self.set_selected_index(ix, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_previous(
        &mut self,
        _: &menu::SelectPrevious,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            let index = self.selected_index();
            let ix = if index == 0 { 0 } else { index - 1 };
            self.set_selected_index(ix, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_first(
        &mut self,
        _: &menu::SelectFirst,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(0, true, window, cx);
            cx.notify();
        }
    }

    pub fn select_last(
        &mut self,
        _: &menu::SelectLast,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let count = self.cell_count();
        if count > 0 {
            self.set_selected_index(count - 1, true, window, cx);
            cx.notify();
        }
    }

    fn jump_to_cell(&mut self, index: usize, _window: &mut Window, _cx: &mut Context<Self>) {
        self.cell_list.scroll_to_reveal_item(index);
    }

    fn button_group(window: &mut Window, cx: &mut Context<Self>) -> Div {
        v_flex()
            .gap(DynamicSpacing::Base04.rems(cx))
            .items_center()
            .w(px(CONTROL_SIZE + 4.0))
            .overflow_hidden()
            .rounded(px(5.))
            .bg(cx.theme().colors().title_bar_background)
            .p_px()
            .border_1()
            .border_color(cx.theme().colors().border)
    }

    fn render_notebook_control(
        id: impl Into<SharedString>,
        icon: IconName,
        _window: &mut Window,
        _cx: &mut Context<Self>,
    ) -> IconButton {
        let id: ElementId = ElementId::Name(id.into());
        IconButton::new(id, icon).width(px(CONTROL_SIZE))
    }

    fn render_notebook_controls(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let has_outputs = self.has_outputs(window, cx);

        v_flex()
            .max_w(px(CONTROL_SIZE + 4.0))
            .items_center()
            .gap(DynamicSpacing::Base16.rems(cx))
            .justify_between()
            .flex_none()
            .h_full()
            .py(DynamicSpacing::Base12.px(cx))
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "run-all-cells",
                                    IconName::PlayFilled,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Execute all cells", &RunAll, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(RunAll), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "clear-all-outputs",
                                    IconName::ListX,
                                    window,
                                    cx,
                                )
                                .disabled(!has_outputs)
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Clear all outputs", &ClearOutputs, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(ClearOutputs), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-up",
                                    IconName::ArrowUp,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell up", &MoveCellUp, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellUp), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "move-cell-down",
                                    IconName::ArrowDown,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Move cell down", &MoveCellDown, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(MoveCellDown), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "new-markdown-cell",
                                    IconName::Plus,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add markdown block", &AddMarkdownBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddMarkdownBlock), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "new-code-cell",
                                    IconName::Code,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Add code block", &AddCodeBlock, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(AddCodeBlock), cx);
                                }),
                            ),
                    )
                    .child(
                        Self::button_group(window, cx)
                            .child(
                                Self::render_notebook_control(
                                    "duplicate-cell",
                                    IconName::Copy,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Duplicate cell", &DuplicateCell, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(DuplicateCell), cx);
                                }),
                            )
                            .child(
                                Self::render_notebook_control(
                                    "delete-cell",
                                    IconName::Trash,
                                    window,
                                    cx,
                                )
                                .tooltip(move |window, cx| {
                                    Tooltip::for_action("Delete cell", &DeleteCell, cx)
                                })
                                .on_click(|_, window, cx| {
                                    window.dispatch_action(Box::new(DeleteCell), cx);
                                }),
                            ),
                    ),
            )
            .child(
                v_flex()
                    .gap(DynamicSpacing::Base08.rems(cx))
                    .items_center()
                    .child(
                        Self::render_notebook_control("more-menu", IconName::Ellipsis, window, cx)
                            .tooltip(move |window, cx| (Tooltip::text("More options"))(window, cx)),
                    )
                    .child(Self::button_group(window, cx).child({
                        let kernel_status = self.kernel.status();
                        let (icon, icon_color) = match &kernel_status {
                            KernelStatus::Idle => (IconName::ReplNeutral, Color::Success),
                            KernelStatus::Busy => (IconName::ReplNeutral, Color::Warning),
                            KernelStatus::Starting => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Error => (IconName::ReplNeutral, Color::Error),
                            KernelStatus::ShuttingDown => (IconName::ReplNeutral, Color::Muted),
                            KernelStatus::Shutdown => (IconName::ReplNeutral, Color::Disabled),
                            KernelStatus::Restarting => (IconName::ReplNeutral, Color::Warning),
                        };
                        let kernel_name = self
                            .kernel_specification
                            .as_ref()
                            .map(|spec| spec.name().to_string())
                            .unwrap_or_else(|| "Select Kernel".to_string());
                        IconButton::new("repl", icon)
                            .icon_color(icon_color)
                            .tooltip(move |window, cx| {
                                Tooltip::text(format!(
                                    "{} ({}). Click to change kernel.",
                                    kernel_name,
                                    kernel_status.to_string()
                                ))(window, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.kernel_picker_handle.toggle(window, cx);
                            }))
                    })),
            )
    }

    fn render_kernel_status_bar(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let kernel_status = self.kernel.status();
        let kernel_name = self
            .kernel_specification
            .as_ref()
            .map(|spec| spec.name().to_string())
            .unwrap_or_else(|| "Select Kernel".to_string());

        let (status_icon, status_color) = match &kernel_status {
            KernelStatus::Idle => (IconName::Circle, Color::Success),
            KernelStatus::Busy => (IconName::ArrowCircle, Color::Warning),
            KernelStatus::Starting => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Error => (IconName::XCircle, Color::Error),
            KernelStatus::ShuttingDown => (IconName::ArrowCircle, Color::Muted),
            KernelStatus::Shutdown => (IconName::Circle, Color::Muted),
            KernelStatus::Restarting => (IconName::ArrowCircle, Color::Warning),
        };

        let is_spinning = matches!(
            kernel_status,
            KernelStatus::Busy
                | KernelStatus::Starting
                | KernelStatus::ShuttingDown
                | KernelStatus::Restarting
        );

        let status_icon_element = if is_spinning {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .with_rotate_animation(2)
                .into_any_element()
        } else {
            Icon::new(status_icon)
                .size(IconSize::Small)
                .color(status_color)
                .into_any_element()
        };

        let worktree_id = self.worktree_id;
        let kernel_picker_handle = self.kernel_picker_handle.clone();
        let view = cx.entity().downgrade();
        let last_error = self.last_error.clone();

        h_flex()
            .w_full()
            .px_3()
            .py_1()
            .gap_2()
            .items_center()
            .justify_between()
            .bg(cx.theme().colors().status_bar_background)
            .child(
                KernelSelector::new(
                    Box::new(move |spec: KernelSpecification, window, cx| {
                        if let Some(view) = view.upgrade() {
                            view.update(cx, |this, cx| {
                                this.change_kernel(spec, window, cx);
                            });
                        }
                    }),
                    worktree_id,
                    Button::new("kernel-selector", kernel_name.clone())
                        .label_size(LabelSize::Small)
                        .icon(status_icon)
                        .icon_size(IconSize::Small)
                        .icon_color(status_color)
                        .icon_position(IconPosition::Start),
                    Tooltip::text(format!(
                        "Kernel: {} ({}). Click to change.",
                        kernel_name,
                        kernel_status.to_string()
                    )),
                )
                .with_handle(kernel_picker_handle),
            )
            .child(
                h_flex()
                    .gap_1()
                    .child(
                        IconButton::new("restart-kernel", IconName::RotateCw)
                            .icon_size(IconSize::Small)
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Restart Kernel", &RestartKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.restart_kernel(&RestartKernel, window, cx);
                            })),
                    )
                    .child(
                        IconButton::new("interrupt-kernel", IconName::Stop)
                            .icon_size(IconSize::Small)
                            .disabled(!matches!(kernel_status, KernelStatus::Busy))
                            .tooltip(|window, cx| {
                                Tooltip::for_action("Interrupt Kernel", &InterruptKernel, cx)
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.interrupt_kernel(&InterruptKernel, window, cx);
                            })),
                    ),
            )
            .when_some(last_error, |this, error| {
                this.child(Label::new(error).size(LabelSize::Small).color(Color::Error))
            })
    }

    fn cell_list(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let view = cx.entity();
        list(self.cell_list.clone(), move |index, window, cx| {
            view.update(cx, |this, cx| {
                let cell_id = &this.cell_order[index];
                if let Some(cell) = this.cell_map.get(cell_id) {
                    this.render_cell(index, cell, window, cx).into_any_element()
                } else {
                    div().into_any_element()
                }
            })
        })
        .size_full()
    }

    fn cell_position(&self, index: usize) -> CellPosition {
        match index {
            0 => CellPosition::First,
            index if index == self.cell_count() - 1 => CellPosition::Last,
            _ => CellPosition::Middle,
        }
    }

    fn render_cell(
        &self,
        index: usize,
        cell: &Cell,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let cell_position = self.cell_position(index);

        let is_selected = index == self.selected_cell_index;

        match cell {
            Cell::Code(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Markdown(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
            Cell::Raw(cell) => {
                cell.update(cx, |cell, _cx| {
                    cell.set_selected(is_selected)
                        .set_cell_position(cell_position);
                });
                cell.clone().into_any_element()
            }
        }
    }
}

impl Render for NotebookEditor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .size_full()
            .key_context("NotebookEditor")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|this, &OpenNotebook, window, cx| {
                this.open_notebook(&OpenNotebook, window, cx)
            }))
            .on_action(
                cx.listener(|this, &ClearOutputs, window, cx| this.clear_outputs(window, cx)),
            )
            .on_action(
                cx.listener(|this, &Run, window, cx| this.run_current_cell(&Run, window, cx)),
            )
            .on_action(cx.listener(|this, &RunAll, window, cx| this.run_cells(window, cx)))
            .on_action(cx.listener(|this, &MoveCellUp, window, cx| this.move_cell_up(window, cx)))
            .on_action(
                cx.listener(|this, &MoveCellDown, window, cx| this.move_cell_down(window, cx)),
            )
            .on_action(cx.listener(|this, &AddMarkdownBlock, window, cx| {
                this.add_markdown_block(window, cx)
            }))
            .on_action(
                cx.listener(|this, &AddCodeBlock, window, cx| this.add_code_block(window, cx)),
            )
            .on_action(cx.listener(|this, &DuplicateCell, window, cx| {
                this.duplicate_selected_cell(window, cx)
            }))
            .on_action(
                cx.listener(|this, &DeleteCell, window, cx| this.delete_selected_cell(window, cx)),
            )
            .on_action(cx.listener(|this, _: &MoveUp, window, cx| {
                this.select_previous(&menu::SelectPrevious, window, cx);
                if let Some(cell_id) = this.cell_order.get(this.selected_cell_index) {
                    if let Some(cell) = this.cell_map.get(cell_id) {
                        match cell {
                            Cell::Code(cell) => {
                                let editor = cell.read(cx).editor().clone();
                                editor.update(cx, |editor, cx| {
                                    editor.move_to_end(&Default::default(), window, cx);
                                });
                                editor.focus_handle(cx).focus(window, cx);
                            }
                            Cell::Markdown(cell) => {
                                cell.update(cx, |cell, cx| {
                                    cell.set_editing(true);
                                    cx.notify();
                                });
                                let editor = cell.read(cx).editor().clone();
                                editor.update(cx, |editor, cx| {
                                    editor.move_to_end(&Default::default(), window, cx);
                                });
                                editor.focus_handle(cx).focus(window, cx);
                            }
                            _ => {}
                        }
                    }
                }
            }))
            .on_action(cx.listener(|this, _: &MoveDown, window, cx| {
                this.select_next(&menu::SelectNext, window, cx);
                if let Some(cell_id) = this.cell_order.get(this.selected_cell_index) {
                    if let Some(cell) = this.cell_map.get(cell_id) {
                        match cell {
                            Cell::Code(cell) => {
                                let editor = cell.read(cx).editor().clone();
                                editor.update(cx, |editor, cx| {
                                    editor.move_to_beginning(&Default::default(), window, cx);
                                });
                                editor.focus_handle(cx).focus(window, cx);
                            }
                            Cell::Markdown(cell) => {
                                cell.update(cx, |cell, cx| {
                                    cell.set_editing(true);
                                    cx.notify();
                                });
                                let editor = cell.read(cx).editor().clone();
                                editor.update(cx, |editor, cx| {
                                    editor.move_to_beginning(&Default::default(), window, cx);
                                });
                                editor.focus_handle(cx).focus(window, cx);
                            }
                            _ => {}
                        }
                    }
                }
            }))
            .on_action(
                cx.listener(|this, action, window, cx| this.restart_kernel(action, window, cx)),
            )
            .on_action(
                cx.listener(|this, action, window, cx| this.interrupt_kernel(action, window, cx)),
            )
            .child(
                h_flex()
                    .flex_1()
                    .w_full()
                    .h_full()
                    .gap_2()
                    .child(div().flex_1().h_full().child(self.cell_list(window, cx)))
                    .child(self.render_notebook_controls(window, cx)),
            )
            .child(self.render_kernel_status_bar(window, cx))
    }
}

impl Focusable for NotebookEditor {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

// Intended to be a NotebookBuffer
pub struct NotebookItem {
    path: PathBuf,
    project_path: ProjectPath,
    languages: Arc<LanguageRegistry>,
    // Raw notebook data
    notebook: nbformat::v4::Notebook,
    // Store our version of the notebook in memory (cell_order, cell_map)
    id: ProjectEntryId,
}

impl project::ProjectItem for NotebookItem {
    fn try_open(
        project: &Entity<Project>,
        path: &ProjectPath,
        cx: &mut App,
    ) -> Option<Task<anyhow::Result<Entity<Self>>>> {
        let path = path.clone();
        let project = project.clone();
        let fs = project.read(cx).fs().clone();
        let languages = project.read(cx).languages().clone();

        if path.path.extension().unwrap_or_default() == "ipynb" {
            Some(cx.spawn(async move |cx| {
                let abs_path = project
                    .read_with(cx, |project, cx| project.absolute_path(&path, cx))
                    .with_context(|| format!("finding the absolute path of {path:?}"))?;

                // todo: watch for changes to the file
                let file_content = fs.load(abs_path.as_path()).await?;

                let notebook = if file_content.trim().is_empty() {
                    nbformat::v4::Notebook {
                        nbformat: 4,
                        nbformat_minor: 5,
                        cells: vec![],
                        metadata: NotebookMetadata {
                            kernelspec: None,
                            language_info: None,
                            authors: None,
                            additional: std::collections::HashMap::new(),
                        },
                    }
                } else {
                    let notebook = match nbformat::parse_notebook(&file_content) {
                        Ok(nb) => nb,
                        Err(_) => {
                            // Pre-process to ensure IDs exist
                            let mut json: serde_json::Value = serde_json::from_str(&file_content)?;
                            if let Some(cells) =
                                json.get_mut("cells").and_then(|c| c.as_array_mut())
                            {
                                for cell in cells {
                                    if cell.get("id").is_none() {
                                        cell["id"] =
                                            serde_json::Value::String(Uuid::new_v4().to_string());
                                    }
                                }
                            }
                            let file_content = serde_json::to_string(&json)?;
                            nbformat::parse_notebook(&file_content)?
                        }
                    };

                    match notebook {
                        nbformat::Notebook::V4(notebook) => notebook,
                        // 4.1 - 4.4 are converted to 4.5
                        nbformat::Notebook::Legacy(legacy_notebook) => {
                            // TODO: Decide if we want to mutate the notebook by including Cell IDs
                            // and any other conversions

                            nbformat::upgrade_legacy_notebook(legacy_notebook)?
                        }
                    }
                };

                let id = project
                    .update(cx, |project, cx| {
                        project.entry_for_path(&path, cx).map(|entry| entry.id)
                    })
                    .context("Entry not found")?;

                Ok(cx.new(|_| NotebookItem {
                    path: abs_path,
                    project_path: path,
                    languages,
                    notebook,
                    id,
                }))
            }))
        } else {
            None
        }
    }

    fn entry_id(&self, _: &App) -> Option<ProjectEntryId> {
        Some(self.id)
    }

    fn project_path(&self, _: &App) -> Option<ProjectPath> {
        Some(self.project_path.clone())
    }

    fn is_dirty(&self) -> bool {
        // TODO: Track if notebook metadata or structure has changed
        false
    }
}

impl NotebookItem {
    pub fn language_name(&self) -> Option<String> {
        self.notebook
            .metadata
            .language_info
            .as_ref()
            .map(|l| l.name.clone())
            .or(self
                .notebook
                .metadata
                .kernelspec
                .as_ref()
                .and_then(|spec| spec.language.clone()))
    }

    pub fn notebook_language(&self) -> impl Future<Output = Option<Arc<Language>>> + use<> {
        let language_name = self.language_name();
        let languages = self.languages.clone();

        async move {
            if let Some(language_name) = language_name {
                languages.language_for_name(&language_name).await.ok()
            } else {
                None
            }
        }
    }
}

impl EventEmitter<()> for NotebookItem {}

impl EventEmitter<()> for NotebookEditor {}

// pub struct NotebookControls {
//     pane_focused: bool,
//     active_item: Option<Box<dyn ItemHandle>>,
//     // subscription: Option<Subscription>,
// }

// impl NotebookControls {
//     pub fn new() -> Self {
//         Self {
//             pane_focused: false,
//             active_item: Default::default(),
//             // subscription: Default::default(),
//         }
//     }
// }

// impl EventEmitter<ToolbarItemEvent> for NotebookControls {}

// impl Render for NotebookControls {
//     fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
//         div().child("notebook controls")
//     }
// }

// impl ToolbarItemView for NotebookControls {
//     fn set_active_pane_item(
//         &mut self,
//         active_pane_item: Option<&dyn workspace::ItemHandle>,
//         window: &mut Window, cx: &mut Context<Self>,
//     ) -> workspace::ToolbarItemLocation {
//         cx.notify();
//         self.active_item = None;

//         let Some(item) = active_pane_item else {
//             return ToolbarItemLocation::Hidden;
//         };

//         ToolbarItemLocation::PrimaryLeft
//     }

//     fn pane_focus_update(&mut self, pane_focused: bool, _window: &mut Window, _cx: &mut Context<Self>) {
//         self.pane_focused = pane_focused;
//     }
// }

impl Item for NotebookEditor {
    type Event = ();

    fn can_split(&self) -> bool {
        true
    }

    fn clone_on_split(
        &self,
        _workspace_id: Option<workspace::WorkspaceId>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Option<Entity<Self>>>
    where
        Self: Sized,
    {
        Task::ready(Some(cx.new(|cx| {
            Self::new(self.project.clone(), self.notebook_item.clone(), window, cx)
        })))
    }

    fn buffer_kind(&self, _: &App) -> workspace::item::ItemBufferKind {
        workspace::item::ItemBufferKind::Singleton
    }

    fn for_each_project_item(
        &self,
        cx: &App,
        f: &mut dyn FnMut(gpui::EntityId, &dyn project::ProjectItem),
    ) {
        f(self.notebook_item.entity_id(), self.notebook_item.read(cx))
    }

    fn tab_content_text(&self, _detail: usize, cx: &App) -> SharedString {
        self.notebook_item
            .read(cx)
            .project_path
            .path
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_default()
            .into()
    }

    fn tab_content(&self, params: TabContentParams, window: &Window, cx: &App) -> AnyElement {
        Label::new(self.tab_content_text(params.detail.unwrap_or(0), cx))
            .single_line()
            .color(params.text_color())
            .when(params.preview, |this| this.italic())
            .into_any_element()
    }

    fn tab_icon(&self, _window: &Window, _cx: &App) -> Option<Icon> {
        Some(IconName::Book.into())
    }

    fn show_toolbar(&self) -> bool {
        false
    }

    fn pixel_position_of_cursor(&self, cx: &App) -> Option<Point<Pixels>> {
        let editor = self.selected_editor(cx)?;
        editor.read(cx).pixel_position_of_cursor(cx)
    }

    fn as_searchable(
        &self,
        _this: &Entity<Self>,
        cx: &App,
    ) -> Option<Box<dyn SearchableItemHandle>> {
        self.selected_editor(cx)
            .map(|editor| Box::new(editor) as Box<dyn SearchableItemHandle>)
    }

    fn set_nav_history(
        &mut self,
        history: workspace::ItemNavHistory,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.nav_history = Some(history.clone());
        for cell in self.cell_map.values() {
            let maybe_editor = match cell {
                Cell::Code(code_cell) => Some(code_cell.read(cx).editor().clone()),
                Cell::Markdown(markdown_cell) => Some(markdown_cell.read(cx).editor().clone()),
                Cell::Raw(_) => None,
            };
            if let Some(editor) = maybe_editor {
                editor.update(cx, |editor, cx| {
                    Item::set_nav_history(editor, history.clone(), window, cx)
                });
            }
        }
    }

    fn can_save(&self, _cx: &App) -> bool {
        true
    }

    fn save(
        &mut self,
        _options: SaveOptions,
        project: Entity<Project>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let path = self.notebook_item.read(cx).path.clone();
        let fs = project.read(cx).fs().clone();

        cx.spawn(async move |this, cx| {
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            fs.atomic_write(path, json).await?;
            this.update(cx, |this, cx| this.mark_as_saved(cx)).ok();
            Ok(())
        })
    }

    fn save_as(
        &mut self,
        project: Entity<Project>,
        path: ProjectPath,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let notebook = self.to_notebook(cx);
        let fs = project.read(cx).fs().clone();

        let abs_path = project.read(cx).absolute_path(&path, cx);

        cx.spawn(async move |this, cx| {
            let abs_path = abs_path.context("Failed to get absolute path")?;
            let json =
                serde_json::to_string_pretty(&notebook).context("Failed to serialize notebook")?;
            fs.atomic_write(abs_path, json).await?;
            this.update(cx, |this, cx| this.mark_as_saved(cx)).ok();
            Ok(())
        })
    }

    fn reload(
        &mut self,
        project: Entity<Project>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<Result<()>> {
        let path = self.notebook_item.read(cx).path.clone();
        let fs = project.read(cx).fs().clone();
        let languages = self.languages.clone();
        let notebook_language = self.notebook_language.clone();

        cx.spawn_in(window, async move |this, cx| {
            let file_content = fs.load(&path).await?;

            let mut json: serde_json::Value = serde_json::from_str(&file_content)?;
            if let Some(cells) = json.get_mut("cells").and_then(|c| c.as_array_mut()) {
                for cell in cells {
                    if cell.get("id").is_none() {
                        cell["id"] = serde_json::Value::String(Uuid::new_v4().to_string());
                    }
                }
            }
            let file_content = serde_json::to_string(&json)?;

            let notebook = nbformat::parse_notebook(&file_content);
            let notebook = match notebook {
                Ok(nbformat::Notebook::V4(notebook)) => notebook,
                Ok(nbformat::Notebook::Legacy(legacy_notebook)) => {
                    nbformat::upgrade_legacy_notebook(legacy_notebook)?
                }
                Err(e) => {
                    anyhow::bail!("Failed to parse notebook: {:?}", e);
                }
            };

            this.update_in(cx, |this, window, cx| {
                let mut cell_order = vec![];
                let mut cell_map = HashMap::default();

                for cell in notebook.cells.iter() {
                    let cell_id = cell.id();
                    cell_order.push(cell_id.clone());
                    let cell_entity =
                        Cell::load(cell, &languages, notebook_language.clone(), window, cx);
                    let notebook_editor = cx.entity().downgrade();
                    Self::subscribe_to_cell_events(&notebook_editor, &cell_id, &cell_entity, cx);
                    cell_map.insert(cell_id.clone(), cell_entity);
                }

                this.cell_order = cell_order.clone();
                this.original_cell_order = cell_order;
                this.cell_map = cell_map;
                this.cell_list =
                    ListState::new(this.cell_order.len(), gpui::ListAlignment::Top, px(1000.));
                if let Some(history) = this.nav_history.clone() {
                    for cell in this.cell_map.values() {
                        let maybe_editor = match cell {
                            Cell::Code(code_cell) => Some(code_cell.read(cx).editor().clone()),
                            Cell::Markdown(markdown_cell) => {
                                Some(markdown_cell.read(cx).editor().clone())
                            }
                            Cell::Raw(_) => None,
                        };
                        if let Some(editor) = maybe_editor {
                            editor.update(cx, |editor, cx| {
                                Item::set_nav_history(editor, history.clone(), window, cx)
                            });
                        }
                    }
                }
                this.notebook_item.update(cx, |item, cx| {
                    item.notebook = notebook.clone();
                    cx.emit(());
                });
                this.notebook_metadata_dirty = false;
                cx.notify();
            })?;

            Ok(())
        })
    }

    fn is_dirty(&self, cx: &App) -> bool {
        self.has_structural_changes() || self.has_content_changes(cx) || self.has_metadata_changes()
    }
}

impl ProjectItem for NotebookEditor {
    type Item = NotebookItem;

    fn for_project_item(
        project: Entity<Project>,
        _pane: Option<&Pane>,
        item: Entity<Self::Item>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        Self::new(project, item, window, cx)
    }
}

impl KernelSession for NotebookEditor {
    fn route(&mut self, message: &JupyterMessage, window: &mut Window, cx: &mut Context<Self>) {
        // Handle kernel status updates (these are broadcast to all)
        if let JupyterMessageContent::Status(status) = &message.content {
            self.kernel.set_execution_state(&status.execution_state);
            if status.execution_state == runtimelib::ExecutionState::Idle {
                self.execute_next_queued_cell(cx);
            }
            cx.notify();
        }

        if let JupyterMessageContent::KernelInfoReply(reply) = &message.content {
            self.kernel.set_kernel_info(reply);

            if let Ok(language_info) = serde_json::from_value::<nbformat::v4::LanguageInfo>(
                serde_json::to_value(&reply.language_info).unwrap_or_default(),
            ) {
                self.notebook_item.update(cx, |item, cx| {
                    item.notebook.metadata.language_info = Some(language_info);
                    cx.emit(());
                });
                self.notebook_metadata_dirty = true;
            }
            cx.notify();
        }

        // Handle cell-specific messages
        if let Some(parent_header) = &message.parent_header {
            let parent_message_id = parent_header.msg_id.clone();

            if let JupyterMessageContent::CompleteReply(completion_reply) = &message.content {
                if let Some(pending_request) = self.completion_requests.remove(&parent_message_id) {
                    if pending_request
                        .sender
                        .send(completion_reply.clone())
                        .is_err()
                    {
                        self.set_last_error(
                            "Notebook completion request was dropped before reply arrived.",
                            cx,
                        );
                    }
                }
                return;
            }

            let cell_id = self.execution_requests.get(&parent_message_id).cloned();
            if let Some(cell_id) = cell_id.as_ref() {
                if let Some(Cell::Code(cell)) = self.cell_map.get(cell_id) {
                    cell.update(cx, |cell, cx| {
                        cell.handle_message(message, window, cx);
                    });
                }
            }

            if matches!(message.content, JupyterMessageContent::ExecuteReply(_)) {
                self.execution_requests.remove(&parent_message_id);
                self.execute_next_queued_cell(cx);
            }
        }
    }

    fn kernel_errored(&mut self, error_message: String, cx: &mut Context<Self>) {
        self.cancel_pending_executions("Kernel errored before execution finished.", cx);
        self.kernel = Kernel::ErroredLaunch(error_message.clone());
        self.set_last_error(format!("Kernel error: {error_message}"), cx);
    }
}
