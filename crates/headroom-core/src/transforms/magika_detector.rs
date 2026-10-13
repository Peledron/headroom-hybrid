//! Magika-based content detection (Stage 3d Tier 1).
//!
//! Wraps Google's [`magika`] crate — an ONNX-backed content classifier —
//! and maps its 200+ labels onto Headroom's existing
//! [`crate::transforms::content_detector::ContentType`] enum so the
//! ContentRouter dispatch (PR5) can stay enum-stable.
//!
//! # Design
//!
//! - **Singleton session, loaded off the caller's thread.** Magika model
//!   loading is the expensive part (one-time ONNX init, ~50 ms cold). We do it
//!   exactly once per process, on a background thread. The `Session` requires
//!   `&mut self` for inference, so the singleton wraps it in a `Mutex` — fine
//!   for our throughput; if benchmarks show contention later we'll pool.
//!
//! - **No caller ever waits on another caller.** Detection takes the session
//!   with `try_lock` and gives up on the load with a plain `get`, so both
//!   contended paths return [`MagikaDetectorError::Busy`] rather than parking.
//!   This matters because the Python binding releases the GIL and its caller
//!   runs under a watchdog that abandons the thread on timeout. An abandoned
//!   thread inside the model keeps whatever it holds forever, so any blocking
//!   wait here turns one slow inference into a dead detector for the life of
//!   the process. Use [`wait_until_ready`] to warm the model instead.
//!
//! - **Loud failures.** If the model fails to load or inference fails,
//!   `magika_detect` returns `Err`. The ContentRouter (PR5) decides
//!   whether to fall back to Tier 2 (`unidiff-rs`) or surface to the
//!   caller. We deliberately do **not** silently return `PlainText` on
//!   error — that's the kind of silent fallback the audit doc forbids.
//!
//! - **Mapping table is explicit.** Every magika label we care about
//!   has an explicit case in [`map_magika_label`]; everything else
//!   falls into [`ContentType::PlainText`]. This is a code-route
//!   decision, not a regex — adding a new mapping is one line, and
//!   readers can audit the dispatch in one screen.
//!
//! - **No router rewiring here.** PR3 lands the detector + tests
//!   only. PR5 flips the ContentRouter to call us instead of the
//!   regex-based [`crate::transforms::content_detector`].
//!
//! - **CPU compatibility.** The precompiled ONNX Runtime binary shipped by
//!   `ort-sys` may contain AVX2-family instructions on x86/x86_64. Where
//!   AVX2 is unavailable on those targets, the session init returns an
//!   error early instead of crashing with SIGILL; the detection chain then
//!   falls through to Tier 2 and Tier 3 normally.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, OnceLock, TryLockError};
use std::time::Duration;

use magika::Session;
use thiserror::Error;
use tracing;

use crate::transforms::content_detector::ContentType;

/// Check whether the CPU can run the precompiled ONNX Runtime binary
/// that magika depends on.
///
/// On x86/x86_64 without AVX2, the `onnxruntime` shared library shipped
/// by `ort-sys` can contain AVX2-family instructions that will SIGILL.
/// We detect this up front so the magika session init can fail gracefully
/// instead of crashing.
///
/// On non-x86 targets, this x86-specific AVX2 gate is not applied.
///
/// Delegates to the shared [`crate::onnx_cpu`] guard so magika and the
/// embedding scorer agree on a single CPU-support source of truth.
pub(crate) fn magika_onnx_runtime_supported_by_cpu() -> bool {
    crate::onnx_cpu::onnx_runtime_supported_by_cpu()
}

/// Check whether this process can safely initialize Magika's ONNX session.
///
/// This is stricter than the CPU check. On dynamic-ORT platforms, the runtime
/// loader must be pinned before `Session::new()` runs; otherwise Windows can
/// resolve the OS-provided `System32\onnxruntime.dll` and hang inside ORT
/// initialization. Python callers get the pin from `headroom._ort`; direct Rust
/// binaries/tests need this fail-fast guard.
pub(crate) fn magika_runtime_available_for_session_init() -> Result<(), String> {
    if !magika_onnx_runtime_supported_by_cpu() {
        return Err(
            "Magika ONNX Runtime backend requires AVX2 on this platform; \
             falling back to non-Magika detection"
                .to_string(),
        );
    }

    dynamic_ort_loader_ready()
}

static DYNAMIC_ORT_INIT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// Ensure the ONNX Runtime shared library is resolved and committed to
/// `ort` before any `ort` API is touched.
///
/// With `ort-load-dynamic` (every platform, see Cargo.toml) this MUST
/// run before any code path that can construct an `ort` session
/// (magika, fastembed): if the dylib cannot be loaded, `ort`
/// 2.0.0-rc.12 deadlocks inside its API-setup error path (recursive
/// `OnceLock` init), and the stuck thread then wedges process exit in
/// `ort`'s `dl_fini` environment teardown (#1715 CI hang).
pub(crate) fn dynamic_ort_loader_ready() -> Result<(), String> {
    DYNAMIC_ORT_INIT
        .get_or_init(initialize_dynamic_ort)
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

fn initialize_dynamic_ort() -> Result<PathBuf, String> {
    let explicit = std::env::var("ORT_DYLIB_PATH")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if let Some(path) = explicit {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!(
                "ORT_DYLIB_PATH points to a missing ONNX Runtime library: {}",
                path.display()
            ));
        }
        init_ort_from_path(&path)?;
        return Ok(path);
    }

    let mut errors = Vec::new();
    let candidates = discover_onnxruntime_libraries();
    for path in &candidates {
        match init_ort_from_path(path) {
            Ok(()) => {
                tracing::info!(
                    ort_dylib_path = %path.display(),
                    "initialized ONNX Runtime for Magika from discovered onnxruntime package"
                );
                return Ok(path.clone());
            }
            Err(error) => errors.push(format!("{}: {error}", path.display())),
        }
    }

    if candidates.is_empty() {
        Err(
            "no pip onnxruntime native library was found for Magika dynamic ONNX Runtime loading; \
             install headroom-ai[proxy], install onnxruntime, or set ORT_DYLIB_PATH"
                .to_string(),
        )
    } else {
        Err(format!(
            "failed to initialize ONNX Runtime for Magika from discovered libraries: {}",
            errors.join("; ")
        ))
    }
}

/// Layout of ONNX Runtime's `OrtApiBase`, the one struct in its C ABI
/// that is reachable without going through `ort`.
///
/// Only the two members we need are declared. Both have been at the
/// front of the struct since ONNX Runtime 1.0 and the C API treats the
/// layout as frozen, so a shorter definition reads the same fields any
/// version would hand back.
#[repr(C)]
struct OrtApiBaseProbe {
    get_api: unsafe extern "C" fn(u32) -> *const core::ffi::c_void,
    get_version_string: unsafe extern "C" fn() -> *const core::ffi::c_char,
}

/// Read the version string out of an ONNX Runtime shared library.
///
/// Loads the library, calls `OrtGetApiBase()->GetVersionString()`, and
/// leaves the mapping in place. We never unload: ONNX Runtime installs
/// static initializers and exit handlers, and `dlclose`-ing it is the
/// same teardown hazard that wedges process exit (ort #1715). Keeping
/// the mapping also means the `dlopen` `ort` does next is a refcount
/// bump on an object we already proved loadable.
#[allow(unsafe_code)]
fn onnxruntime_version_string(path: &Path) -> Result<String, String> {
    // SAFETY: `OrtGetApiBase` is ONNX Runtime's documented entry point.
    // It takes no arguments, cannot fail, and returns a pointer to a
    // static struct owned by the library. `GetVersionString` likewise
    // returns a static NUL-terminated string. The library outlives the
    // borrow because we forget the handle instead of dropping it.
    unsafe {
        let library = libloading::Library::new(path)
            .map_err(|error| format!("failed to load `{}`: {error}", path.display()))?;

        let result = (|| {
            let base_getter: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBaseProbe> =
                library.get(b"OrtGetApiBase").map_err(|error| {
                    format!(
                        "`{}` does not export OrtGetApiBase: {error}",
                        path.display()
                    )
                })?;

            let base = base_getter();
            if base.is_null() {
                return Err(format!("`{}` returned a null OrtApiBase", path.display()));
            }

            let version = ((*base).get_version_string)();
            if version.is_null() {
                return Err(format!("`{}` reported a null version", path.display()));
            }

            Ok(CStr::from_ptr(version).to_string_lossy().into_owned())
        })();

        core::mem::forget(library);
        result
    }
}

/// Reject an ONNX Runtime library that this build of `ort` cannot use.
///
/// `ort` compares the library's minor version against the API level it
/// was built for and errors out on anything older. That error is not
/// survivable: `load_dylib_from_path` marks its global library
/// `OnceLock` completed *without writing a value*, then `setup_api`
/// panics on the `Err` it got back, and every later `ort` call reads
/// the slot that was never written. The process is left with a
/// permanently wedged detector, which is exactly the hang we saw with
/// onnxruntime 1.23.2 against an `ort` compiled for 1.24.
///
/// So the version check happens here, before `ort` sees the path. A
/// library that fails it is skipped with a plain `Err`, which lets the
/// discovery loop keep trying other candidates.
fn ort_can_use_library(path: &Path) -> Result<(), String> {
    let version = onnxruntime_version_string(path)?;

    // Parse exactly the way `ort` does, so our verdict and its verdict
    // can never disagree: second dot-separated field, anything
    // unparseable counts as 0 and therefore too old.
    let minor = version
        .split('.')
        .nth(1)
        .map_or(0, |field| field.parse::<u32>().unwrap_or(0));

    if minor < ort::MINOR_VERSION {
        return Err(format!(
            "ONNX Runtime at `{}` is version `{version}`, but this build of ort needs \
             1.{}.x or newer; install `onnxruntime>=1.{}`",
            path.display(),
            ort::MINOR_VERSION,
            ort::MINOR_VERSION
        ));
    }

    Ok(())
}

fn init_ort_from_path(path: &Path) -> Result<(), String> {
    ort_can_use_library(path)?;

    let builder = ort::init_from(path).map_err(|error| {
        format!(
            "failed to load ONNX Runtime from `{}`: {error}",
            path.display()
        )
    })?;
    let _committed = builder.commit();
    Ok(())
}

fn discover_onnxruntime_libraries() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    for var in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Some(root) = env_path(var) {
            roots.push(root);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd.join(".venv"));
        roots.push(cwd.join("venv"));
    }

    if let Some(user_profile) = env_path("USERPROFILE") {
        roots.extend(versioned_children(
            user_profile
                .join(".pyenv")
                .join("pyenv-win")
                .join("versions"),
        ));
        roots.extend(versioned_children(
            user_profile
                .join("AppData")
                .join("Local")
                .join("Programs")
                .join("Python"),
        ));
        roots.extend(versioned_children(
            user_profile.join("AppData").join("Roaming").join("Python"),
        ));
    }

    if let Some(home) = env_path("HOME") {
        roots.extend(versioned_children(home.join(".pyenv").join("versions")));
        roots.push(home.join(".local"));
    }

    let mut candidates = Vec::new();
    for root in roots {
        candidates.extend(onnxruntime_candidates_under(&root));
    }
    dedup_existing_files(candidates)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn versioned_children(root: PathBuf) -> Vec<PathBuf> {
    let mut children = std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    children.sort_by(|a, b| b.cmp(a));
    children
}

fn onnxruntime_candidates_under(root: &Path) -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        vec![
            root.join("Lib")
                .join("site-packages")
                .join("onnxruntime")
                .join("capi")
                .join("onnxruntime.dll"),
            root.join("site-packages")
                .join("onnxruntime")
                .join("capi")
                .join("onnxruntime.dll"),
        ]
    }

    #[cfg(not(target_os = "windows"))]
    {
        let mut candidates = Vec::new();
        for site_packages in python_site_packages_dirs(root) {
            let capi = site_packages.join("onnxruntime").join("capi");
            candidates.extend(onnxruntime_dylibs_in(&capi));
        }
        candidates
    }
}

#[cfg(not(target_os = "windows"))]
fn python_site_packages_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![root.join("lib").join("site-packages")];
    let lib = root.join("lib");
    dirs.extend(
        std::fs::read_dir(lib)
            .ok()
            .into_iter()
            .flat_map(|entries| entries.filter_map(Result::ok))
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("python"))
            })
            .map(|path| path.join("site-packages")),
    );
    dirs
}

#[cfg(not(target_os = "windows"))]
fn onnxruntime_dylibs_in(capi: &Path) -> Vec<PathBuf> {
    let mut dylibs = std::fs::read_dir(capi)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libonnxruntime")
                        && (name.ends_with(".dylib") || name.contains(".so"))
                })
        })
        .collect::<Vec<_>>();
    dylibs.sort();
    dylibs
}

fn dedup_existing_files(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_file() && !out.iter().any(|seen| seen == &path) {
            out.push(path);
        }
    }
    out
}

/// Errors from the magika detector. Wraps the underlying `magika::Error`
/// so callers can match on whether init or inference broke without
/// pulling magika types into their imports.
#[derive(Debug, Error)]
pub enum MagikaDetectorError {
    /// One-time session initialization failed (model load, ONNX init).
    /// Once we hit this, every subsequent call also fails — there is no
    /// retry path here. The router should surface and stop.
    #[error("magika session init failed: {0}")]
    Init(String),

    /// Inference call failed for this input. Usually transient; future
    /// calls may succeed. The error message is the magika-side text;
    /// we don't try to wrap it.
    #[error("magika inference failed: {0}")]
    Inference(String),

    /// Singleton lock was poisoned (a previous holder panicked while
    /// holding it). The detector is unusable until the process
    /// restarts. We don't auto-recover — a panicked detector means
    /// something is corrupt and continuing would mask it.
    #[error("magika session lock poisoned")]
    Poisoned,

    /// The session is not loaded yet, or another caller holds it. Not a
    /// failure: the chain drops to the unidiff/regex tiers for this one call
    /// and the next call may well succeed.
    ///
    /// This variant is what makes the detector safe to call from a thread the
    /// embedder is willing to abandon. Both waits it replaces were unbounded,
    /// so a caller that hung inside the model took the whole process's
    /// detector with it. See [`wait_until_ready`].
    #[error("magika session not ready")]
    Busy,
}

/// One-process singleton holding the magika session. Populated only by the
/// background initializer in [`ensure_init_started`], never by a detection
/// caller.
///
/// `Mutex<Result<Session, ...>>` rather than `Result<Mutex<Session>>`
/// so init failure is recorded once and replayed cheaply on every
/// subsequent call (no re-attempting the load — if the model file is
/// missing or ort can't init, retrying just wastes cycles).
static MAGIKA_SESSION: OnceLock<Mutex<Result<Session, String>>> = OnceLock::new();

/// Latch saying the background initializer has been kicked off. Deliberately
/// an atomic rather than a `OnceLock` init closure: `get_or_init` parks every
/// other caller for as long as the closure runs, which is the wait this module
/// exists to remove.
static INIT_STARTED: AtomicBool = AtomicBool::new(false);

/// Set once `MAGIKA_SESSION` is populated, so [`wait_until_ready`] can block
/// without polling. Detection never touches these.
static INIT_DONE: Mutex<bool> = Mutex::new(false);
static INIT_DONE_CV: Condvar = Condvar::new();

/// Record the init outcome and release anyone waiting in [`wait_until_ready`].
fn publish_session(result: Result<Session, String>) {
    // Only the single initializer thread reaches here, so `set` cannot lose a
    // race. Ignoring the error keeps the first outcome authoritative anyway.
    let _ = MAGIKA_SESSION.set(Mutex::new(result));
    let mut done = INIT_DONE.lock().unwrap_or_else(|e| e.into_inner());
    *done = true;
    INIT_DONE_CV.notify_all();
}

/// Start loading the session in the background, at most once per process.
///
/// Returns immediately. Everything that can hang — the dynamic ORT loader
/// probe and `Session::new()` alike — runs on threads no caller waits on.
fn ensure_init_started() {
    if INIT_STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    if let Err(e) = std::thread::Builder::new()
        .name("magika-init".into())
        .spawn(run_session_init)
    {
        tracing::warn!("magika init thread spawn failed: {e}");
        publish_session(Err(format!("magika init thread spawn failed: {e}")));
    }
}

/// Supervise the load: bound it, then publish an `Ok` or an `Err` either way.
///
/// The probe runs inside the supervised worker, not before it. On dynamic-ORT
/// platforms `dynamic_ort_loader_ready` can itself park inside ort's recursive
/// `OnceLock` error path (#1715), so leaving it outside the timeout would let
/// the one thing this timeout is for escape it.
fn run_session_init() {
    let timeout = magika_init_timeout();
    let (tx, rx) = mpsc::channel();
    // The orphaned worker on timeout is left to finish on its own; its
    // eventual `send` lands on a dropped receiver (harmless) and the `Session`
    // is then dropped.
    let spawned = std::thread::Builder::new()
        .name("magika-session-new".into())
        .spawn(move || {
            let outcome = match magika_runtime_available_for_session_init() {
                Err(error) => Err(error),
                Ok(()) => Session::new().map_err(|e| e.to_string()),
            };
            let _ = tx.send(outcome);
        });
    if let Err(e) = spawned {
        tracing::warn!("magika session thread spawn failed: {e}");
        publish_session(Err(format!("magika session thread spawn failed: {e}")));
        return;
    }

    match rx.recv_timeout(timeout) {
        Ok(res) => publish_session(res),
        Err(_) => {
            let ort_dylib = std::env::var("ORT_DYLIB_PATH").ok();
            tracing::warn!(
                timeout_secs = timeout.as_secs(),
                ort_dylib_path = ort_dylib.as_deref(),
                "magika ONNX session init timed out; detection falls back to \
                 non-ML tiers for this process. On Windows an unset \
                 ORT_DYLIB_PATH usually means the WinML System32 \
                 onnxruntime.dll was picked up (deadlocks ort init)."
            );
            publish_session(Err(format!(
                "magika session init exceeded {}s timeout; \
                 using non-ML detection tiers",
                timeout.as_secs()
            )));
        }
    }
}

/// Block until the session has loaded (or failed to), up to `timeout`.
///
/// Detection itself never calls this. It exists for two callers who can afford
/// to wait and would otherwise silently lose the ML tier: a process that warms
/// the detector at startup, and tests. Returns whether the outcome is known,
/// not whether it was a success.
pub fn wait_until_ready(timeout: Duration) -> bool {
    ensure_init_started();
    let done = INIT_DONE.lock().unwrap_or_else(|e| e.into_inner());
    let (done, _) = INIT_DONE_CV
        .wait_timeout_while(done, timeout, |ready| !*ready)
        .unwrap_or_else(|e| e.into_inner());
    *done
}

/// Default cap on magika ONNX session init.
///
/// On some platforms `Session::new()` can hang indefinitely instead of
/// returning an error. Root-caused on Windows: with `ort-load-dynamic`
/// (Windows-gated in `Cargo.toml`), the bare `LoadLibrary("onnxruntime.dll")`
/// search resolves to `C:\Windows\System32\onnxruntime.dll` — the Windows ML
/// OS component (1.17.x on Win11 24H2+) — and initializing an ort 2.x
/// session against it deadlocks at 0% CPU rather than erroring. A hang —
/// unlike an `Err` — is not caught by the tiered fallback in
/// [`crate::transforms::detection`], so it stalls the entire compression
/// pipeline until the proxy's own 30s+ timeout fires on every request.
///
/// The real fix is `headroom/_ort.py`, which pins `ORT_DYLIB_PATH` to the
/// pip-installed `onnxruntime` DLL before this crate can load ort. This
/// timeout remains as the safety net for unpinned embedders of the crate.
/// Override with `HEADROOM_MAGIKA_INIT_TIMEOUT_SECS`.
const MAGIKA_INIT_TIMEOUT_SECS_DEFAULT: u64 = 5;

fn magika_init_timeout() -> Duration {
    let secs = std::env::var("HEADROOM_MAGIKA_INIT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(MAGIKA_INIT_TIMEOUT_SECS_DEFAULT);
    Duration::from_secs(secs)
}

/// Classify `content` and return the mapped Headroom [`ContentType`].
///
/// Empty input shortcuts to [`ContentType::PlainText`] without touching
/// the model — saves the round trip on every empty tool result.
///
/// **Never blocks on another caller.** If the session is still loading, or a
/// different thread is inside the model, this returns
/// [`MagikaDetectorError::Busy`] and the chain drops a tier for this one call.
/// The earlier version waited in both places, which is how a single stuck
/// inference took the detector out for the rest of the process: the stuck
/// thread kept the session mutex, and every later call parked behind it with
/// no timeout and no escape. Callers that need the ML tier warm should call
/// [`wait_until_ready`] once at startup instead of paying for it here.
pub fn magika_detect(content: &str) -> Result<ContentType, MagikaDetectorError> {
    if content.is_empty() {
        return Ok(ContentType::PlainText);
    }

    ensure_init_started();
    let Some(mutex) = MAGIKA_SESSION.get() else {
        return Err(MagikaDetectorError::Busy);
    };
    let mut guard = match mutex.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::WouldBlock) => return Err(MagikaDetectorError::Busy),
        Err(TryLockError::Poisoned(_)) => return Err(MagikaDetectorError::Poisoned),
    };
    let session = guard
        .as_mut()
        .map_err(|e| MagikaDetectorError::Init(e.clone()))?;

    let bytes = content.as_bytes();
    let file_type = session
        .identify_content_sync(bytes)
        .map_err(|e| MagikaDetectorError::Inference(e.to_string()))?;

    Ok(map_magika_label(file_type.info().label))
}

/// Map a magika label string to Headroom's [`ContentType`] enum.
///
/// **Why explicit cases instead of `group == "code"`:** magika's
/// `group` field is a coarse bucket ("code", "text", "binary",
/// "executable", ...). Some entries we want — like `markdown`,
/// `txt`, `latex` — are in the `text` group along with formats we
/// route differently. So we case on the label directly: clear, one
/// match arm per decision, no group-vs-label semantic confusion.
///
/// **Unmapped labels return [`ContentType::PlainText`]**, the safest
/// default — passthrough at the router level rather than misroute to
/// a wrong compressor. PR5 will refine this for `SearchResults` /
/// `BuildOutput` (which magika has no equivalent for).
pub fn map_magika_label(label: &str) -> ContentType {
    match label {
        // ── JSON ───────────────────────────────────────────────────
        // PR5 will refine this with the existing `is_json_array_of_dicts`
        // check — magika says "this is JSON" but doesn't tell us if it's
        // an array of records vs. a single object. For PR3 the mapping
        // exists; the refinement is a router concern.
        "json" | "jsonl" => ContentType::JsonArray,

        // ── Diffs ──────────────────────────────────────────────────
        "diff" => ContentType::GitDiff,

        // ── HTML ───────────────────────────────────────────────────
        "html" | "xml" => ContentType::Html,

        // ── Source code ────────────────────────────────────────────
        // The big "code" group from magika. We list the labels we
        // actually expect to see in tool outputs / pasted code in
        // proxy traffic. Anything else in the code group falls
        // through to PlainText — better passthrough than misroute.
        "rust" | "python" | "javascript" | "typescript" | "go" | "java" | "c" | "cpp" | "cs"
        | "php" | "ruby" | "swift" | "kotlin" | "scala" | "haskell" | "lua" | "dart" | "perl"
        | "shell" | "powershell" | "batch" | "sql" | "css" | "vue" | "groovy" | "clojure"
        | "asm" | "cmake" | "dockerfile" | "makefile" | "yaml" | "toml" | "ini" | "hcl"
        | "jinja" => ContentType::SourceCode,

        // ── Plain text-ish ─────────────────────────────────────────
        // markdown, rst, latex, log-style, txt, empty/unknown all
        // route as plain text. The router won't try to compress these
        // with a code-aware compressor.
        "markdown" | "rst" | "latex" | "txt" | "empty" | "unknown" | "undefined" => {
            ContentType::PlainText
        }

        // ── Default: passthrough ───────────────────────────────────
        _ => ContentType::PlainText,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────
//
// These tests are integration-y — they hit the real magika model, which
// loads ONNX on first call (~50 ms cold). Total wall-clock for the full
// suite is dominated by that one-time load, so we keep cases compact.
//
// Detection is probabilistic; we assert against `ContentType` enum
// values rather than confidence scores or labels directly. If magika's
// model version changes (`MODEL_NAME` in their crate), individual
// label assignments may shift but our `match` arms are wide enough to
// stay stable.

#[cfg(test)]
mod tests {
    use super::*;

    /// Wait out the two things production callers refuse to wait for.
    ///
    /// `magika_detect` now declines rather than queues, so a test that asserts
    /// a label has to warm the model itself and retry past a peer holding the
    /// session. Cargo runs these in parallel, so the peer case is real.
    fn detect_for_test(content: &str) -> Result<ContentType, MagikaDetectorError> {
        assert!(
            wait_until_ready(Duration::from_secs(60)),
            "magika session init never settled"
        );
        for _ in 0..200 {
            match magika_detect(content) {
                Err(MagikaDetectorError::Busy) => std::thread::sleep(Duration::from_millis(10)),
                other => return other,
            }
        }
        panic!("magika session stayed busy for 2s; a caller is holding it")
    }

    fn assert_detect(content: &str, expected: ContentType, hint: &str) {
        if let Err(init_reason) = magika_runtime_available_for_session_init() {
            // On hosts where Magika cannot safely initialize, assert graceful
            // degradation rather than panicking or hanging.
            match detect_for_test(content) {
                Err(MagikaDetectorError::Init(msg)) => {
                    assert!(
                        msg == init_reason,
                        "{hint}: expected init error {init_reason:?}, got: {msg:?}"
                    );
                }
                other => panic!("{hint}: expected Magika init error, got {other:?}"),
            }
        } else {
            match detect_for_test(content) {
                Ok(got) => {
                    assert_eq!(got, expected, "{hint}: expected {expected:?}, got {got:?}")
                }
                Err(e) => panic!("{hint}: detection failed: {e}"),
            }
        }
    }

    #[test]
    fn empty_input_is_plain_text_without_model_call() {
        // The shortcut path — should not touch the model.
        let result = magika_detect("").unwrap();
        assert_eq!(result, ContentType::PlainText);
    }

    #[test]
    fn detects_json() {
        assert_detect(
            r#"{"name": "Alice", "age": 30, "tags": ["a", "b"]}"#,
            ContentType::JsonArray,
            "single-object JSON",
        );
    }

    #[test]
    fn detects_json_array() {
        let payload = r#"[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 3, "v": "c"}]"#;
        assert_detect(payload, ContentType::JsonArray, "array-of-records JSON");
    }

    #[test]
    fn detects_python_source() {
        let src = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Tree:
    def __init__(self, value):
        self.value = value
        self.children = []
"#;
        assert_detect(src, ContentType::SourceCode, "python class+def");
    }

    #[test]
    fn detects_rust_source() {
        let src = r#"
use std::collections::HashMap;

pub struct Counter {
    counts: HashMap<String, u32>,
}

impl Counter {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }
}
"#;
        assert_detect(src, ContentType::SourceCode, "rust struct+impl");
    }

    #[test]
    fn detects_javascript_source() {
        let src = r#"
const fetchUser = async (id) => {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error('Not found');
    return response.json();
};
"#;
        assert_detect(src, ContentType::SourceCode, "JS arrow + async");
    }

    #[test]
    fn detects_unified_diff() {
        let diff = r#"diff --git a/foo.py b/foo.py
index abc123..def456 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def hello():
+    print("new line")
     return "world"
"#;
        assert_detect(diff, ContentType::GitDiff, "git unified diff");
    }

    #[test]
    fn detects_markdown_as_plain_text() {
        // Markdown isn't routed to a code compressor — it goes to
        // plain text. This is by design; markdown compression has its
        // own path that isn't hooked up yet.
        let md = "# Hello\n\nThis is **bold** and *italic*.\n\n- Item 1\n- Item 2\n";
        assert_detect(md, ContentType::PlainText, "markdown");
    }

    #[test]
    fn detects_plain_text() {
        let prose = "The quick brown fox jumps over the lazy dog. \
                     This is just regular English prose with no \
                     special structure.";
        assert_detect(prose, ContentType::PlainText, "english prose");
    }

    #[test]
    fn detects_html() {
        let html =
            "<!DOCTYPE html><html><head><title>x</title></head><body><h1>Hi</h1></body></html>";
        assert_detect(html, ContentType::Html, "minimal HTML page");
    }

    #[test]
    fn detects_yaml_as_source_code() {
        let yaml = "name: my-app\nversion: 1.0\ndependencies:\n  - foo\n  - bar\n";
        assert_detect(yaml, ContentType::SourceCode, "YAML config");
    }

    #[test]
    fn detects_shell_script_as_source_code() {
        let sh = "#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  echo \"$f\"\ndone\n";
        assert_detect(sh, ContentType::SourceCode, "bash script with shebang");
    }

    #[test]
    fn detects_sql_as_source_code() {
        let sql = "SELECT u.id, u.name, COUNT(o.id) AS order_count \
                   FROM users u LEFT JOIN orders o ON u.id = o.user_id \
                   WHERE u.active = TRUE GROUP BY u.id, u.name;";
        assert_detect(sql, ContentType::SourceCode, "SQL query");
    }

    #[test]
    fn singleton_session_is_reused_across_calls() {
        // Two back-to-back calls should reuse the same session
        // (or same cached error). When the Magika runtime is available the
        // session is Ok and repeated calls succeed; otherwise the session is
        // Err and repeated calls return the same Err.
        if let Err(init_reason) = magika_runtime_available_for_session_init() {
            let r1 = detect_for_test("hello world");
            let r2 = detect_for_test("def f(): pass");
            let r3 = detect_for_test(r#"{"a":1}"#);
            for r in [&r1, &r2, &r3] {
                match r {
                    Err(MagikaDetectorError::Init(msg)) => {
                        assert_eq!(msg, &init_reason);
                    }
                    other => panic!("expected Magika init error, got {other:?}"),
                }
            }
        } else {
            // On available hosts the session loads once and all calls
            // succeed. Wall-clock asymmetry (cold ~50 ms, warm
            // <1 ms) confirms reuse.
            detect_for_test("hello world").unwrap();
            detect_for_test("def f(): pass").unwrap();
            detect_for_test(r#"{"a":1}"#).unwrap();
        }
    }

    /// The regression that motivated the non-blocking rewrite.
    ///
    /// A caller stuck inside the model holds the session mutex. Before, every
    /// later call parked on that mutex with no timeout, so one stuck inference
    /// cost the process its detector permanently. The contract now is that a
    /// held session is reported, not waited on, and that the detector recovers
    /// the moment the holder lets go.
    #[test]
    fn a_held_session_is_declined_not_waited_on() {
        assert!(
            wait_until_ready(Duration::from_secs(60)),
            "magika session init never settled"
        );
        let mutex = MAGIKA_SESSION.get().expect("session published");

        let held = mutex.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        let during = magika_detect("hello world");
        let waited = started.elapsed();
        drop(held);

        assert!(
            matches!(during, Err(MagikaDetectorError::Busy)),
            "expected Busy while the session was held, got {during:?}"
        );
        assert!(
            waited < Duration::from_secs(1),
            "declining took {waited:?}; it should not have waited at all"
        );

        // Recovery: the holder is gone, so the next call goes through.
        assert!(
            !matches!(
                detect_for_test("hello world"),
                Err(MagikaDetectorError::Busy)
            ),
            "detector stayed unusable after the holder released it"
        );
    }

    #[test]
    fn waiting_for_readiness_settles_within_the_init_budget() {
        // Bounded either way: the initializer publishes an Ok or an Err, so a
        // host without a usable ONNX runtime settles just as fast as one with.
        assert!(wait_until_ready(Duration::from_secs(60)));
        assert!(MAGIKA_SESSION.get().is_some());
    }

    #[test]
    fn unmapped_labels_route_to_plain_text() {
        // Direct test of the mapping table — covers labels we
        // explicitly didn't enumerate. Future magika versions may
        // add new labels and we want unknown-but-real labels to
        // safely passthrough rather than misroute.
        assert_eq!(map_magika_label("ace"), ContentType::PlainText);
        assert_eq!(map_magika_label("flac"), ContentType::PlainText);
        assert_eq!(map_magika_label("3gp"), ContentType::PlainText);
        assert_eq!(
            map_magika_label("garbage_unseen_label"),
            ContentType::PlainText
        );
    }

    #[test]
    fn known_label_table_round_trips() {
        // Cheap sanity that the mapping arms compile and behave.
        // No magika session needed — pure table lookup.
        assert_eq!(map_magika_label("json"), ContentType::JsonArray);
        assert_eq!(map_magika_label("jsonl"), ContentType::JsonArray);
        assert_eq!(map_magika_label("diff"), ContentType::GitDiff);
        assert_eq!(map_magika_label("html"), ContentType::Html);
        assert_eq!(map_magika_label("rust"), ContentType::SourceCode);
        assert_eq!(map_magika_label("python"), ContentType::SourceCode);
        assert_eq!(map_magika_label("yaml"), ContentType::SourceCode);
        assert_eq!(map_magika_label("markdown"), ContentType::PlainText);
        assert_eq!(map_magika_label("txt"), ContentType::PlainText);
        assert_eq!(map_magika_label("empty"), ContentType::PlainText);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_onnxruntime_candidate_matches_pip_layout() {
        let root = std::env::temp_dir().join(format!(
            "headroom-ort-discovery-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let dll = root
            .join("Lib")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi")
            .join("onnxruntime.dll");
        std::fs::create_dir_all(dll.parent().unwrap()).unwrap();
        std::fs::write(&dll, b"not a real dll").unwrap();

        let candidates = dedup_existing_files(onnxruntime_candidates_under(&root));
        assert_eq!(candidates, vec![dll]);

        let _ = std::fs::remove_dir_all(root);
    }
}
