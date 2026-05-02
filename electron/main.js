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

// Initialized inside app.whenReady() — app.getPath() is only valid after app is ready
let depsFlag   = null
let venvDir    = null
let venvPython = null

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

// ── Find system Python (used only to create the venv) ───────────────────────
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
                shell: IS_WIN
            })
            // Reject Windows Store stub — it exits 0 but prints nothing
            if (out && out.includes('Python')) return p
        } catch { continue }
    }
    return null
}

// ── Create isolated venv in userData ────────────────────────────────────────
function setupVenv(systemPython) {
    return new Promise((resolve, reject) => {
        if (fs.existsSync(venvPython)) {
            resolve()   // venv already exists
            return
        }
        updateSplash('Creating Python environment…')
        const proc = spawn(systemPython, ['-m', 'venv', venvDir], { shell: IS_WIN })
        proc.on('close', code => {
            if (code === 0) resolve()
            else reject(new Error(`Failed to create Python environment (code ${code})`))
        })
        proc.on('error', () => {
            reject(new Error('Could not create Python environment.\nPlease install Python 3 from https://python.org'))
        })
    })
}

// ── Install dependencies into the venv ──────────────────────────────────────
function installRequirements() {
    return new Promise((resolve, reject) => {
        // Verify uvicorn is importable; if not, wipe the flag and reinstall
        if (fs.existsSync(depsFlag)) {
            try {
                execSync(`"${venvPython}" -c "import uvicorn"`, {
                    stdio: 'ignore',
                    shell: IS_WIN
                })
                resolve()   // deps are healthy
                return
            } catch {
                fs.unlinkSync(depsFlag)   // stale flag — reinstall
            }
        }

        updateSplash('Installing Python dependencies (first launch only)…')

        const reqFile = path.join(projectRoot, 'requirements.txt')

        // Installing inside a venv never hits PEP 668 or permission errors
        const install = spawn(venvPython, ['-m', 'pip', 'install', '-r', reqFile], {
            cwd: projectRoot,
            shell: IS_WIN
        })

        install.on('close', code => {
            if (code === 0) {
                fs.writeFileSync(depsFlag, new Date().toISOString())
                resolve()
            } else {
                reject(new Error(`Dependency install failed (code ${code}).\nPlease check your internet connection and relaunch.`))
            }
        })

        install.on('error', err => {
            reject(new Error(`Install error: ${err.message}`))
        })
    })
}

// ── FastAPI server ───────────────────────────────────────────────────────────
function startPythonServer() {
    updateSplash('Starting AI backend…')

    // Write server logs to userData so crashes are diagnosable
    const logPath = path.join(app.getPath('userData'), 'server.log')
    const logStream = fs.createWriteStream(logPath, { flags: 'a' })

    pythonProcess = spawn(
        venvPython,
        ['-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '8000'],
        {
            cwd: projectRoot,
            stdio: isDev ? 'inherit' : ['ignore', logStream, logStream],
            shell: IS_WIN
        }
    )

    pythonProcess.on('error', err => {
        dialog.showErrorBox('Backend Error', `Failed to start server:\n${err.message}\n\nLog: ${logPath}`)
        app.quit()
    })

    pythonProcess.on('exit', (code) => {
        if (mainWindow && !mainWindow.isDestroyed() && code !== 0) {
            dialog.showErrorBox('Backend Crashed',
                `Server exited unexpectedly (code ${code})\n\nCheck log for details:\n${logPath}`)
        }
    })
}

// ── Wait for FastAPI to respond ──────────────────────────────────────────────
// TensorFlow cold-start can take 30–60 s on first run — allow up to 2 minutes
function waitForServer(retries = 240) {
    return new Promise((resolve, reject) => {
        const attempt = (n) => {
            // Update splash message as time passes so the user isn't confused
            const elapsed = (240 - n) * 0.5
            if (elapsed > 10 && elapsed <= 10.5)
                updateSplash('Loading AI models… (first launch takes up to 2 minutes)')
            if (elapsed > 60 && elapsed <= 60.5)
                updateSplash('Almost ready — loading TensorFlow weights…')

            http.get('http://127.0.0.1:8000', () => {
                resolve()
            }).on('error', () => {
                if (n <= 0) {
                    const logPath = path.join(app.getPath('userData'), 'server.log')
                    reject(new Error(
                        `Server did not start within 2 minutes.\n\nCheck the log for errors:\n${logPath}`
                    ))
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
    // app.getPath() is only safe to call after app is ready
    const userData = app.getPath('userData')
    depsFlag   = path.join(userData, '.deps_installed')
    venvDir    = path.join(userData, 'app-venv')
    venvPython = IS_WIN
        ? path.join(venvDir, 'Scripts', 'python.exe')
        : path.join(venvDir, 'bin', 'python3')

    createSplash('Starting up…')

    try {
        const systemPython = findPython()
        if (!systemPython) {
            throw new Error('Python 3 not found.\nPlease install it from https://python.org and relaunch the app.')
        }

        await setupVenv(systemPython)
        await installRequirements()
        startPythonServer()
        updateSplash('Waiting for AI backend to be ready…')
        await waitForServer(240)  // up to 2 minutes — TensorFlow cold-start is slow
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
