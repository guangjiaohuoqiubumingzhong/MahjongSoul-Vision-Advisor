#![windows_subsystem = "windows"]

use std::{env, os::windows::process::CommandExt, process::{Command, exit}};

#[link(name = "user32")]
unsafe extern "system" {
    fn MessageBoxW(hwnd: *mut core::ffi::c_void, text: *const u16, title: *const u16, kind: u32) -> i32;
}

fn main() {
    let result = (|| -> std::io::Result<std::process::ExitStatus> {
        let executable = env::current_exe()?;
        let runtime = executable.parent().unwrap().join("_internal");
        Command::new(runtime.join("pythonw.exe"))
            .arg("-B")
            .arg(runtime.join("launch_portable.pyw"))
            .args(env::args_os().skip(1))
            .current_dir(&runtime)
            .creation_flags(0x08000000)
            .status()
    })();
    match result {
        Ok(status) => exit(status.code().unwrap_or(1)),
        Err(error) => {
            let text: Vec<u16> = format!("无法启动 Majan，请完整解压便携包并保留 _internal 文件夹。\n\n{}", error)
                .encode_utf16().chain(Some(0)).collect();
            let title: Vec<u16> = "Majan 无法启动".encode_utf16().chain(Some(0)).collect();
            unsafe { MessageBoxW(core::ptr::null_mut(), text.as_ptr(), title.as_ptr(), 0x10); }
            exit(1);
        }
    }
}
