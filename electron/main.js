const { app, BrowserWindow, dialog, ipcMain } = require('electron')
const { spawn, execSync } = require('child_process')
const path = require('path')
const http = require('http')
const fs = require('fs')

// ── Path resolution ──────────────────────────────────────────────────────────
// In dev:       projectRoot = /path/to/DegradedPhotoDetection/
// In packaged:  projectRoot = Contents/Resources/  (extraResources land here)
const isDev = !app.isPackaged
const projectRoot = isDev
    ? path.join(__dirname, '..')
    : process.resourcesPath

// Flag file written after first successful pip install so we never run it again
const depsFlag = path.join(app.getPath('userData'), '.deps_installed')

let mainWindow = null
let splashWindow = null
let pythonProcess = null

// ── Splash / loading window ──────────────────────────────────────────────────
function createSplash(message) {
    splashWindow = new BrowserWindow({
        width: 480,
        height: 260,
        frame: false,
        alwaysOnTop: true,
        resizable: false,
        transparent: true,
        webPreferences: { contextIsolation: true }
    })

    // Inline HTML so there are no external file dependencies
    const html = `
    <html>
    <body style="margin:0;background:#1A1A2E;border-radius:12px;
                 font-family:-apple-system,sans-serif;color:#fff;
                 display:flex;flex-direction:column;align-items:center;
                 justify-content:center;height:100vh;user-select:none;">
      <div style="font-size:22px;font-weight:700;color:#E8AA14;margin-bottom:12px;">
        Degraded Photo Detection
      </div>
      <div id="msg" style="font-size:14px;color:#A0B0C8;margin-bottom:24px;">${message}</div>
      <div style="width:300px;height:6px;background:#0F3A60;border-radius:3px;overflow:hidden;">
        <div id="bar" style="height:100%;width:30%;background:#E8AA14;
             border-radius:3px;animation:slide 1.4s ease-in-out infinite;"></div>
      </div>
      <style>
        @keyframes slide {
          0%   { transform: translateX(-100%); width: 40%; }
          50%  { transform: translateX(200%);  width: 60%; }
          100% { transform: translateX(500%);  width: 40%; }
        }
      </style>
    </body>
    </html>`

    splashWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html))
}

function updateSplash(message) {
    if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.webContents.executeJavaScript(
            `document.getElementById('msg').innerText = ${JSON.stringify(message)}`
        )
    }
}

function closeSplash() {
    if (splashWindow && !splashWindow.isDestroyed()) {
        splashWindow.close()
        splashWindow = null
    }
}

const IS_WIN = process.platform === 'win32'

// ── Find Python interpreter ──────────────────────────────────────────────────
function findPython() {
    // 'py' is the Windows Python launcher — always in PATH even when python/python3 aren't
    const candidates = IS_WIN
        ? ['py', 'python', 'python3']
        : ['python3', '/usr/bin/python3', '/usr/local/bin/python3', 'python']

    for (const p of candidates) {
        try {
            const out = execSync(`"${p}" --version`, {
                encoding: 'utf8',
                stdio: 'pipe',
                shell: IS_WIN   // required on Windows to find executables in PATH
            })
            // Confirm it's a real Python — the Windows Store stub exits 0 but prints nothing
            if (out && out.includes('Python')) return p
        } catch { continue }
    }
    return null
}

// ── Dependency installation ──────────────────────────────────────────────────
function installRequirements(python) {
    return new Promise((resolve, reject) => {
        // Even if the flag exists, verify uvicorn is actually importable.
        // If not, the previous install was incomplete — delete flag and reinstall.
        if (fs.existsSync(depsFlag)) {
            try {
                execSync(`"${python}" -c "import uvicorn"`, {
                    stdio: 'ignore',
                    shell: IS_WIN
                })
                resolve()   // uvicorn is importable — deps are healthy
                return
            } catch {
                fs.unlinkSync(depsFlag)   // stale flag — fall through to reinstall
            }
        }

        updateSplash('Installing Python dependencies (first launch only)…')

        const reqFile = path.join(projectRoot, 'requirements.txt')

        // python -m pip ensures we install into the exact Python that runs the app
        const install = spawn(python, ['-m', 'pip', 'install', '-r', reqFile], {
            cwd: projectRoot,
            shell: IS_WIN
        })

        install.on('close', code => {
            if (code === 0) {
                fs.writeFileSync(depsFlag, new Date().toISOString())
                resolve()
            } else {
                reject(new Error(`pip install failed (exit code ${code}).\nMake sure Python 3 is installed from https://python.org`))
            }
        })

        install.on('error', () => {
            reject(new Error('Python not found.\nPlease install Python 3 from https://python.org and relaunch.'))
        })
    })
}

// ── FastAPI server ───────────────────────────────────────────────────────────
function startPythonServer(python) {
    updateSplash('Starting AI backend…')

    // python -m uvicorn works regardless of where pip installed the binary
    pythonProcess = spawn(
        python,
        ['-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '8000'],
        {
            cwd: projectRoot,
            stdio: isDev ? 'inherit' : 'ignore',
            shell: IS_WIN
        }
    )

    pythonProcess.on('error', err => {
        dialog.showErrorBox('Backend Error', `Failed to start server:\n${err.message}`)
        app.quit()
    })

    pythonProcess.on('exit', (code) => {
        if (mainWindow && !mainWindow.isDestroyed() && code !== 0) {
            dialog.showErrorBox('Backend Crashed', `Server exited unexpectedly (code ${code})`)
        }
    })
}

// ── Wait for FastAPI to respond ──────────────────────────────────────────────
function waitForServer(retries = 40) {
    return new Promise((resolve, reject) => {
        const attempt = (n) => {
            http.get('http://127.0.0.1:8000', res => {
                resolve()
            }).on('error', () => {
                if (n <= 0) {
                    reject(new Error('Server did not start in time.'))
                } else {
                    setTimeout(() => attempt(n - 1), 500)
                }
            })
        }
        attempt(retries)
    })
}

// ── Main window ──────────────────────────────────────────────────────────────
function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 1024,
        minHeight: 700,
        title: 'Degraded Photo Detection',
        show: false,   // show only after load to avoid flash
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false
        }
    })

    mainWindow.loadURL('http://127.0.0.1:8000')

    mainWindow.once('ready-to-show', () => {
        closeSplash()
        mainWindow.show()
    })

    mainWindow.on('closed', () => { mainWindow = null })

    // Remove the default menu bar (optional — gives a cleaner look)
    mainWindow.setMenuBarVisibility(false)
}

// ── App lifecycle ────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
    createSplash('Starting up…')

    try {
        const python = findPython()
        if (!python) {
            throw new Error('Python 3 not found.\nPlease install it from https://python.org and relaunch the app.')
        }

        await installRequirements(python)
        startPythonServer(python)
        updateSplash('Waiting for AI backend to be ready…')
        await waitForServer(40)   // up to 20 seconds
        createMainWindow()
    } catch (err) {
        closeSplash()
        dialog.showErrorBox('Startup Failed', err.message)
        app.quit()
    }
})

// Cross-platform folder picker — works on macOS, Windows, and Linux
ipcMain.handle('browse-folder', async () => {
    const result = await dialog.showOpenDialog(mainWindow, {
        title: 'Select Gallery Folder to Scan',
        properties: ['openDirectory']
    })
    return result.canceled ? '' : result.filePaths[0]
})

app.on('window-all-closed', () => {
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM')
        pythonProcess = null
    }
    app.quit()
})

app.on('before-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill('SIGTERM')
        pythonProcess = null
    }
})
