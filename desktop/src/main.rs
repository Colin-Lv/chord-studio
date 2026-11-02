#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{env, fs, io, mem, path::{Path, PathBuf}, process::{Child, Command, Stdio},
          ptr, thread, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use std::os::windows::{ffi::OsStrExt, io::AsRawHandle, process::CommandExt};
use serde_json::Value;
use tauri::Manager;
use windows_sys::Win32::{Foundation::*, System::{JobObjects::*, Threading::*}, UI::WindowsAndMessaging::*};
use windows_sys::Win32::System::Diagnostics::ToolHelp::*;

fn wide(text: &str) -> Vec<u16> { std::ffi::OsStr::new(text).encode_wide().chain(Some(0)).collect() }
fn fail(message: &str) { unsafe { MessageBoxW(ptr::null_mut(), wide(message).as_ptr(), wide("Chord Studio").as_ptr(), MB_OK | MB_ICONERROR); } }

struct OwnedHandle(HANDLE);
unsafe impl Send for OwnedHandle {}
unsafe impl Sync for OwnedHandle {}
impl Drop for OwnedHandle { fn drop(&mut self) { unsafe { if !self.0.is_null() { CloseHandle(self.0); } } } }

fn job() -> io::Result<OwnedHandle> {
    unsafe {
        let handle = OwnedHandle(CreateJobObjectW(ptr::null(), ptr::null()));
        if handle.0.is_null() { return Err(io::Error::last_os_error()); }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(handle.0, JobObjectExtendedLimitInformation,
            &limits as *const _ as _, mem::size_of_val(&limits) as u32) == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(handle)
    }
}

fn resume(child: &Child) -> io::Result<()> {
    unsafe {
        let raw = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if raw == INVALID_HANDLE_VALUE { return Err(io::Error::last_os_error()); }
        let snapshot = OwnedHandle(raw);
        let mut entry: THREADENTRY32 = mem::zeroed(); entry.dwSize = mem::size_of_val(&entry) as u32;
        let mut found = Thread32First(snapshot.0, &mut entry);
        while found != 0 {
            if entry.th32OwnerProcessID == child.id() {
                let thread = OwnedHandle(OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID));
                if thread.0.is_null() || ResumeThread(thread.0) == u32::MAX { return Err(io::Error::last_os_error()); }
                return Ok(());
            }
            found = Thread32Next(snapshot.0, &mut entry);
        }
        Err(io::Error::other("启动线程不存在"))
    }
}

struct Backend {
    scope: OwnedHandle,
    child: Child,
    ready: PathBuf,
    gate: PathBuf,
    url: tauri::Url,
    data: PathBuf,
}
impl Drop for Backend {
    fn drop(&mut self) {
        unsafe { TerminateJobObject(self.scope.0, 0); }
        let _ = self.child.wait();
        // Wait for descendants (including Python redirectors) before releasing ownership.
        let end = Instant::now() + Duration::from_secs(15);
        while Instant::now() < end {
            unsafe {
                let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = mem::zeroed();
                if QueryInformationJobObject(self.scope.0, JobObjectBasicAccountingInformation,
                    &mut info as *mut _ as _, mem::size_of_val(&info) as u32, ptr::null_mut()) == 0 || info.ActiveProcesses == 0 { break; }
            }
            thread::sleep(Duration::from_millis(25));
        }
        let _ = fs::remove_file(&self.ready);
        let _ = fs::remove_file(&self.gate);
    }
}

fn json(path: &Path) -> Result<Value, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("无法读取 {}：{e}", path.display()))?;
    serde_json::from_str(content.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())
}
fn resolve(home: &Path, value: &str) -> PathBuf { let p = PathBuf::from(value); if p.is_absolute() {p} else {home.join(p)} }

fn start(home: &Path, data: PathBuf, python: PathBuf, backend: PathBuf) -> Result<Backend, String> {
    fs::create_dir_all(&data).map_err(|e| e.to_string())?;
    if !python.is_file() { return Err("运行环境尚未配置完成，请运行安装程序。".into()); }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
    let ready = data.join(format!(".session-{}-{stamp}.json", std::process::id()));
    let gate = data.join(format!(".launch-{}-{stamp}.go", std::process::id()));
    let log = fs::OpenOptions::new().create(true).append(true).open(data.join("backend.log")).map_err(|e| e.to_string())?;
    let scope = job().map_err(|e| e.to_string())?;
    let mut child = Command::new(python).args(["-I", "-B"]).arg(backend.join("desktop_backend.py"))
        .arg("--data-dir").arg(&data).arg("--ready-file").arg(&ready).arg("--go-file").arg(&gate)
        .arg("--owner-pid").arg(std::process::id().to_string()).current_dir(home)
        .env("PYTHONUTF8", "1").env("PYTHONNOUSERSITE", "1").env_remove("PYTHONPATH").env_remove("PYTHONHOME")
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?)).stderr(Stdio::from(log))
        .creation_flags(CREATE_NO_WINDOW | CREATE_SUSPENDED).spawn().map_err(|e| e.to_string())?;
    unsafe {
        if AssignProcessToJobObject(scope.0, child.as_raw_handle() as HANDLE) == 0 {
            let error = io::Error::last_os_error(); let _ = child.kill(); let _ = child.wait();
            return Err(error.to_string());
        }
    }
    // Backend cannot start workers before assignment to the desktop's Job Object.
    let mut result = Backend {scope, child, ready, gate, data, url: "http://127.0.0.1/".parse().unwrap()};
    fs::write(&result.gate, b"go").map_err(|e| e.to_string())?;
    resume(&result.child).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if result.ready.is_file() {
            let session = json(&result.ready)?;
            let url: tauri::Url = session["url"].as_str().ok_or("服务地址缺失")?.parse().map_err(|_| "服务地址无效")?;
            if url.scheme() != "http" || url.host_str() != Some("127.0.0.1") || session["owner_pid"].as_u64() != Some(std::process::id() as u64) {
                return Err("本地服务返回了不匹配的启动信息。".into());
            }
            fs::copy(&result.ready, result.data.join("active-session.json")).map_err(|e| e.to_string())?;
            result.url = url; return Ok(result);
        }
        if result.child.try_wait().map_err(|e| e.to_string())?.is_some() { return Err(format!("本机服务启动失败，项目已保留。\n日志：{}", result.data.join("backend.log").display())); }
        if Instant::now() >= deadline { return Err("本机服务启动超时，项目已保留。请查看 backend.log。".into()); }
        thread::sleep(Duration::from_millis(50));
    }
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args_os().collect();
    let smoke = args.iter().any(|arg| arg == "--smoke-test");
    let ui_smoke = args.iter().any(|arg| arg == "--ui-smoke-test");
    let home = if let Some(index) = args.iter().position(|a| a == "--home") {
        PathBuf::from(args.get(index+1).ok_or("--home 缺少目录")?)
    } else { env::current_exe().map_err(|e| e.to_string())?.parent().unwrap().to_path_buf() };
    let install = json(&home.join("installation.json"))?;
    let data = resolve(&home, install["data_dir"].as_str().ok_or("安装配置缺少数据目录")?);
    // One owner per data directory. Opening another window must not start another queue.
    let name = format!("Local\\ChordStudio-{}", data.to_string_lossy().replace(['\\','/',':'], "_").to_lowercase());
    let instance;
    unsafe {
        instance = OwnedHandle(CreateMutexW(ptr::null(), 0, wide(&name).as_ptr()));
        if instance.0.is_null() { return Err(io::Error::last_os_error().to_string()); }
        if GetLastError() == ERROR_ALREADY_EXISTS {
            if smoke { return Err("应用已在运行。".into()); }
            let window = FindWindowW(ptr::null(), wide("Chord Studio").as_ptr());
            if !window.is_null() { ShowWindow(window, SW_RESTORE); SetForegroundWindow(window); }
            return Ok(());
        }
    }
    let backend = start(&home, data, resolve(&home,install["web_python"].as_str().ok_or("运行环境配置缺失")?),
                        resolve(&home, install["backend"].as_str().unwrap_or("backend")))?;
    if smoke {
        let pid = backend.child.id(); drop(backend);
        fs::write(home.join("launcher-smoke.json"), format!("{{\"started\":true,\"owner_cleanup_completed\":true,\"child_pid\":{pid}}}" )).map_err(|e| e.to_string())?;
        return Ok(());
    }
    let url = backend.url.clone(); let origin = url.origin();
    let webview_data = backend.data.join("webview");
    let ui_report = home.join("launcher-ui-smoke.json");
    if ui_smoke { let _ = fs::remove_file(&ui_report); }
    tauri::Builder::default().setup(move |app| {
        if ui_smoke {
            let handle = app.handle().clone();
            thread::spawn(move || {thread::sleep(Duration::from_secs(60));handle.exit(2);});
        }
        tauri::WebviewWindowBuilder::new(app, "main", tauri::WebviewUrl::External(url))
            .title("Chord Studio").inner_size(1440.0, 960.0).min_inner_size(900.0, 640.0)
            .visible(!ui_smoke)
            .data_directory(webview_data).on_navigation(move |target| target.origin() == origin)
            .on_page_load(move |window, payload| {
                if ui_smoke && payload.event() == tauri::webview::PageLoadEvent::Finished && payload.url().path() == "/" {
                    let _ = fs::write(&ui_report, "{\"webview2_loaded\":true,\"local_page_loaded\":true}");
                    window.app_handle().exit(0);
                }
            })
            .build()?;
        Ok(())
    }).run(tauri::generate_context!()).map_err(|e| e.to_string())?;
    drop(backend); drop(instance); Ok(())
}

fn main() {
    if let Err(error) = run() {
        if env::args().any(|a| a == "--smoke-test" || a == "--ui-smoke-test") { eprintln!("{error}"); } else { fail(&error); }
        std::process::exit(1);
    }
}
