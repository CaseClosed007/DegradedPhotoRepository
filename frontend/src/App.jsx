import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [photos, setPhotos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [targetFolder, setTargetFolder] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [isBrowsing, setIsBrowsing] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/degraded-photos')
      .then(res => res.json())
      .then(data => {
        setPhotos(data);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch photos:", err);
        setLoading(false);
      });
  }, []);

  // NEW: Function to trigger the native Python folder picker
  const handleBrowse = async () => {
    setIsBrowsing(true);
    try {
      const res = await fetch('http://localhost:8000/api/browse');
      const data = await res.json();
      
      if (data.folder_path) {
        setTargetFolder(data.folder_path); // Auto-fill the input box!
      }
    } catch (err) {
      console.error(err);
      alert("Failed to open native file browser.");
    } finally {
      setIsBrowsing(false);
    }
  };

  const handleScan = async () => {
    if (!targetFolder.trim()) {
      alert("Please enter a valid folder path.");
      return;
    }

    setIsScanning(true);
    setPhotos([]); 

    try {
      const res = await fetch('http://localhost:8000/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: targetFolder })
      });

      const data = await res.json();
      
      if (res.ok) {
        setPhotos(data);
      } else {
        alert(`Scan failed: ${data.detail}`);
      }
    } catch (err) {
      console.error(err);
      alert("Failed to connect to the AI Backend.");
    } finally {
      setIsScanning(false);
    }
  };

  if (loading) return <div className="loading">Loading Gallery...</div>;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Degraded Photo Review</h1>
        
        {/* UPDATED: The Scan Controller with Browse Button */}
        <div className="scan-controller">
          <button 
            onClick={handleBrowse} 
            disabled={isScanning || isBrowsing}
            className="browse-button"
            title="Open native folder picker"
          >
            📁 Browse
          </button>
          
          <input 
            type="text" 
            placeholder="Paste or browse for absolute folder path..." 
            value={targetFolder}
            onChange={(e) => setTargetFolder(e.target.value)}
            disabled={isScanning}
            className="folder-input"
          />
          
          <button 
            onClick={handleScan} 
            disabled={isScanning}
            className="scan-button"
          >
            {isScanning ? "Scanning..." : "Scan Gallery"}
          </button>
        </div>
        
        {!isScanning && <p>Found {photos.length} photos flagged for deletion.</p>}
      </header>

      {/* ... (Keep your existing grid rendering code down here) ... */}
      {isScanning ? (
        <div className="loading">Analyzing images with MobileNetV2... This may take a moment.</div>
      ) : photos.length === 0 ? (
        <div className="empty-state">No degraded photos found! Your drive is clean.</div>
      ) : (
        <div className="photo-grid">
          {photos.map((photo) => (
            <div key={photo.id} className="photo-card">
              <div className="image-container">
                <img 
                  src={`http://localhost:8000/api/image?path=${encodeURIComponent(photo.path)}`} 
                  alt={photo.reason} 
                  loading="lazy"
                />
              </div>
              <div className="card-details">
                <span className="reason-badge">{photo.reason}</span>
                <p className="file-path" title={photo.path}>{photo.path.split('/').pop()}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;