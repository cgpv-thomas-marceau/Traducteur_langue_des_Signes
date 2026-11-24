import { useState, useRef } from 'react';
import WebcamView from '../WebcamView';
import SignPredictor from '../SignPredictor';
import './App.css';

export default function App() {
  const [isHandDetected, setIsHandDetected] = useState(false);
  const landmarksRef = useRef(null);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Reconnaissance de la Langue des Signes</h1>
        <p>Preuve de concept — Edge ML avec MediaPipe & TensorFlow.js</p>
      </header>

      <main className="main-content">
        <section className="video-card">
          <WebcamView
            onHandUpdate={setIsHandDetected}
            landmarksRef={landmarksRef}
          />
          <div style={{ marginTop: '12px' }}>
            Statut :{' '}
            <span className={`status-badge ${isHandDetected ? 'active' : ''}`}>
              {isHandDetected ? 'Main détectée' : 'Aucune main'}
            </span>
          </div>
        </section>

        <section className="control-panel">
          <SignPredictor
            landmarksRef={landmarksRef}
            isHandDetected={isHandDetected}
          />
        </section>
      </main>
    </div>
  );
}