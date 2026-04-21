import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [photos, setPhotos]             = useState([]);
  const [loading, setLoading]           = useState(true);
  const [targetFolder, setTargetFolder] = useState('');
  const [isScanning, setIsScanning]     = useState(false);
  const [isBrowsing, setIsBrowsing]     = useState(false);
  const [photosVisible, setPhotosVisible] = useState(false);

  const [enhancedPhotos, setEnhancedPhotos] = useState({});
  const [enhancing, setEnhancing]           = useState(new Set());
  const [showEnhanced, setShowEnhanced]     = useState({});

  useEffect(() => {
    fetch('http://localhost:8000/api/degraded-photos')
      .then(res => res.json())
      .then(data => {
        setPhotos(data);
        setLoading(false);
        if (data.length > 0) setTimeout(() => setPhotosVisible(true), 50);
      })
      .catch(() => setLoading(false));
  }, []);

  const handleBrowse = async () => {
    setIsBrowsing(true);
    try {
      const res  = await fetch('http://localhost:8000/api/browse');
      const data = await res.json();
      if (data.folder_path) setTargetFolder(data.folder_path);
    } catch {
      alert('Failed to open native file browser.');
    } finally {
      setIsBrowsing(false);
    }
  };

  const handleScan = async () => {
    if (!targetFolder.trim()) { alert('Please enter a valid folder path.'); return; }
    setIsScanning(true);
    setPhotos([]);
    setPhotosVisible(false);
    setEnhancedPhotos({});
    setShowEnhanced({});
    try {
      const res  = await fetch('http://localhost:8000/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: targetFolder }),
      });
      const data = await res.json();
      if (res.ok) {
        setPhotos(data);
        setTimeout(() => setPhotosVisible(true), 50);
      } else {
        alert(`Scan failed: ${data.detail}`);
      }
    } catch {
      alert('Failed to connect to the AI Backend.');
    } finally {
      setIsScanning(false);
    }
  };

  const handleEnhance = async (photo) => {
    setEnhancing(prev => new Set(prev).add(photo.id));
    try {
      const res  = await fetch('http://localhost:8000/api/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: photo.path, reason: photo.reason }),
      });
      const data = await res.json();
      if (res.ok) {
        setEnhancedPhotos(prev => ({ ...prev, [photo.id]: data.enhanced_path }));
        setShowEnhanced(prev => ({ ...prev, [photo.id]: true }));
      } else {
        alert(`Enhancement failed: ${data.detail}`);
      }
    } catch {
      alert('Failed to connect to the AI Backend.');
    } finally {
      setEnhancing(prev => { const n = new Set(prev); n.delete(photo.id); return n; });
    }
  };

  if (loading) {
    return (
      <div className="splash">
        <div className="splash-orb" />
        <div className="splash-content">
          <div className="splash-icon">🔍</div>
          <p className="splash-text">Loading Gallery<span className="dots"><span>.</span><span>.</span><span>.</span></span></p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Animated background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-glow" />
        <div className="header-badge">AI-Powered</div>
        <h1 className="app-title">
          <span className="title-icon">🖼</span>
          Photo Quality<br />
          <span className="title-gradient">Inspector</span>
        </h1>
        <p className="app-subtitle">
          Detect &amp; restore degraded photos using MobileNetV2 + ESRGAN
        </p>

        {/* ── Scan Controller ── */}
        <div className="scan-controller">
          <button
            onClick={handleBrowse}
            disabled={isScanning || isBrowsing}
            className="browse-button"
          >
            {isBrowsing ? <span className="btn-spinner" /> : '📁'}
            <span>{isBrowsing ? 'Opening…' : 'Browse'}</span>
          </button>

          <div className="input-wrapper">
            <span className="input-icon">📂</span>
            <input
              type="text"
              placeholder="Paste or browse for folder path…"
              value={targetFolder}
              onChange={e => setTargetFolder(e.target.value)}
              disabled={isScanning}
              className="folder-input"
            />
          </div>

          <button
            onClick={handleScan}
            disabled={isScanning}
            className="scan-button"
          >
            {isScanning
              ? <><span className="btn-spinner" /><span>Scanning…</span></>
              : <><span>⚡</span><span>Scan Gallery</span></>
            }
          </button>
        </div>

        {!isScanning && photos.length > 0 && (
          <div className="stats-bar">
            <span className="stat-chip stat-chip-red">
              <span className="stat-dot" />
              {photos.length} flagged
            </span>
            <span className="stat-chip stat-chip-green">
              <span className="stat-dot stat-dot-green" />
              {Object.keys(enhancedPhotos).length} enhanced
            </span>
          </div>
        )}
      </header>

      {/* ── Body ── */}
      {isScanning ? (
        <div className="scanning-state">
          <div className="scan-animation">
            <div className="scan-ring scan-ring-1" />
            <div className="scan-ring scan-ring-2" />
            <div className="scan-ring scan-ring-3" />
            <div className="scan-core">🤖</div>
          </div>
          <h2 className="scan-title">Analyzing with AI</h2>
          <p className="scan-subtitle">MobileNetV2 + Laplacian variance gate scanning your gallery…</p>
          <div className="progress-bar"><div className="progress-fill" /></div>
        </div>
      ) : photos.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">✨</div>
          <h2 className="empty-title">All Clear!</h2>
          <p className="empty-subtitle">No degraded photos found — your gallery is pristine.</p>
        </div>
      ) : (
        <div className={`photo-grid ${photosVisible ? 'grid-visible' : ''}`}>
          {photos.map((photo, index) => {
            const isEnhancing      = enhancing.has(photo.id);
            const enhancedPath     = enhancedPhotos[photo.id];
            const isViewingEnhanced = showEnhanced[photo.id] && enhancedPath;
            const isBlur           = photo.reason.includes('Blur');

            const imageSrc = isViewingEnhanced
              ? `http://localhost:8000/api/image?path=${encodeURIComponent(enhancedPath)}`
              : `http://localhost:8000/api/image?path=${encodeURIComponent(photo.path)}`;

            return (
              <div
                key={photo.id}
                className={`photo-card${enhancedPath ? ' has-enhanced' : ''}`}
                style={{ animationDelay: `${index * 60}ms` }}
              >
                {/* Image */}
                <div className="image-container">
                  <img src={imageSrc} alt={photo.reason} loading="lazy" />

                  {/* Degradation type tag */}
                  <div className={`image-tag ${isBlur ? 'tag-blur' : 'tag-cnn'}`}>
                    {isBlur ? '💧 Blur' : '🔍 CNN'}
                  </div>

                  {isViewingEnhanced && (
                    <div className="enhanced-overlay-badge">
                      <span>✨</span> Enhanced
                    </div>
                  )}

                  {isEnhancing && (
                    <div className="image-enhancing-overlay">
                      <div className="enhancing-orb" />
                      <span>Enhancing…</span>
                    </div>
                  )}
                </div>

                {/* Details */}
                <div className="card-details">
                  <p className="file-path" title={photo.path}>
                    {photo.path.split('/').pop()}
                  </p>
                  <p className="reason-text">{photo.reason}</p>

                  <div className="card-actions">
                    {!enhancedPath && (
                      <button
                        className="enhance-button"
                        onClick={() => handleEnhance(photo)}
                        disabled={isEnhancing}
                      >
                        {isEnhancing
                          ? <><span className="btn-spinner" />Enhancing…</>
                          : <>✨ Enhance</>
                        }
                      </button>
                    )}

                    {enhancedPath && (
                      <div className="view-toggle">
                        <button
                          className={`toggle-btn${!isViewingEnhanced ? ' active' : ''}`}
                          onClick={() => setShowEnhanced(p => ({ ...p, [photo.id]: false }))}
                        >
                          Original
                        </button>
                        <button
                          className={`toggle-btn${isViewingEnhanced ? ' active' : ''}`}
                          onClick={() => setShowEnhanced(p => ({ ...p, [photo.id]: true }))}
                        >
                          ✨ Enhanced
                        </button>
                      </div>
                    )}

                    {enhancedPath && (
                      <a
                        href={`http://localhost:8000/api/image?path=${encodeURIComponent(enhancedPath)}`}
                        download
                        className="download-link"
                      >
                        ↓ Save
                      </a>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default App;
