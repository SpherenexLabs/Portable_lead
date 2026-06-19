import { useState, useEffect, useRef, useCallback } from 'react';
import { ref, onValue } from 'firebase/database';
import { database } from './firebase';
import { predictLeadContamination, getModelInfo } from './api';
import SensorCard from './components/SensorCard';
import PredictionCard from './components/PredictionCard';
import TrendChart from './components/TrendChart';
import AnalysisPanel from './components/AnalysisPanel';
import ModelInfo from './components/ModelInfo';
import './styles/App.css';

const MAX_HISTORY = 20;

function App() {
  const [sensors, setSensors] = useState({ TDS: 0, TDS_ADC: 0, TURB: 0, TURB_ADC: 0 });
  const [prediction, setPrediction] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [history, setHistory] = useState([]);
  const [firebaseStatus, setFirebaseStatus] = useState('Connecting...');
  const [backendStatus, setBackendStatus] = useState('Unknown');
  const [predicting, setPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const debounceRef = useRef(null);

  // Fetch model metadata once on load
  useEffect(() => {
    getModelInfo()
      .then((info) => {
        setModelInfo(info);
        setBackendStatus('Connected');
      })
      .catch(() => {
        setBackendStatus('Offline');
        setModelInfo({ error: 'Backend offline. Run: python app.py' });
      });
  }, []);

  // Call ML predict endpoint whenever sensor values change
  const callPredict = useCallback(async (data) => {
    setPredicting(true);
    setErrorMsg(null);
    try {
      const result = await predictLeadContamination(data);
      setPrediction(result);
      setBackendStatus('Connected');
    } catch (err) {
      setErrorMsg(`Prediction error: ${err.message}`);
      setBackendStatus('Error');
    } finally {
      setPredicting(false);
    }
  }, []);

  // Subscribe to Firebase /Portable_lead
  useEffect(() => {
    const dbRef = ref(database, 'Portable_lead');

    const unsub = onValue(
      dbRef,
      (snapshot) => {
        const raw = snapshot.val();
        if (!raw) {
          setFirebaseStatus('Connected (No Data)');
          return;
        }

        setFirebaseStatus('Connected');

        const newSensors = {
          TDS: Number(raw.TDS) || 0,
          TDS_ADC: Number(raw.TDS_ADC) || 0,
          TURB: Number(raw.TURB) || 0,
          TURB_ADC: Number(raw.TURB_ADC) || 0,
        };

        setSensors(newSensors);
        setLastUpdated(new Date());

        // Append to rolling history for trend chart
        setHistory((prev) => {
          const entry = {
            time: new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            TDS: newSensors.TDS,
            TURB: newSensors.TURB,
          };
          return [...prev, entry].slice(-MAX_HISTORY);
        });

        // Debounce predict calls (500 ms after last Firebase update)
        if (debounceRef.current) clearTimeout(debounceRef.current);
        debounceRef.current = setTimeout(() => callPredict(newSensors), 500);
      },
      (err) => {
        setFirebaseStatus('Disconnected');
        setErrorMsg(`Firebase: ${err.message}`);
      }
    );

    return () => {
      unsub();
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [callPredict]);

  const isAlert =
    prediction && (prediction.status === 'Risky' || prediction.status === 'Not Drinkable');

  const alertClass = prediction?.status === 'Not Drinkable' ? 'alert-danger' : 'alert-warning';

  return (
    <div className="app-wrapper">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="header-logo">💧</span>
            <div>
              <h1 className="header-title">Portable Lead Detection in Water</h1>
              <p className="header-sub">IoT + ML Based Water Quality Monitoring System</p>
            </div>
          </div>

          <div className="header-badges">
            <span className={`badge ${firebaseStatus === 'Connected' ? 'badge-green' : 'badge-red'}`}>
              <span className="badge-dot" />
              Firebase: {firebaseStatus}
            </span>
            <span className={`badge ${backendStatus === 'Connected' ? 'badge-green' : 'badge-red'}`}>
              <span className="badge-dot" />
              Backend: {backendStatus}
            </span>
          </div>
        </div>
      </header>

      <main className="app-main">
        {/* ── Alert Banner ── */}
        {isAlert && prediction && (
          <div className={`alert-banner ${alertClass}`}>
            <span className="alert-icon">⚠️</span>
            <div>
              <strong>Water Quality Alert — {prediction.status}</strong>
              <p>{prediction.analysis?.split('.')[0]}.</p>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {errorMsg && (
          <div className="error-bar">
            <span>⚠️ {errorMsg}</span>
            <button className="error-dismiss" onClick={() => setErrorMsg(null)}>✕</button>
          </div>
        )}

        {/* ── Sensor Cards ── */}
        <section className="section">
          <div className="section-head">
            <h2 className="section-title">Live Sensor Readings</h2>
            {lastUpdated && (
              <span className="section-meta">Updated: {lastUpdated.toLocaleTimeString()}</span>
            )}
          </div>
          <div className="sensor-grid">
            <SensorCard label="TDS" value={sensors.TDS} unit="ppm" icon="🧪" colorClass="card-blue" />
            <SensorCard label="TDS ADC" value={sensors.TDS_ADC} unit="raw" icon="📊" colorClass="card-purple" />
            <SensorCard label="Turbidity" value={sensors.TURB} unit="NTU" icon="🌊" colorClass="card-teal" />
            <SensorCard label="Turbidity ADC" value={sensors.TURB_ADC} unit="raw" icon="📈" colorClass="card-indigo" />
          </div>
        </section>

        {/* ── ML Prediction ── */}
        <section className="section">
          <h2 className="section-title">ML Predicted Lead Risk</h2>
          <p className="section-note">
            Result powered by trained ML model — not hardcoded sensor thresholds
          </p>
          <PredictionCard prediction={prediction} loading={predicting} />
        </section>

        {/* ── Trend Chart ── */}
        <section className="section">
          <h2 className="section-title">Real-time Sensor Trends</h2>
          <TrendChart history={history} />
        </section>

        {/* ── Analysis + Suggestions ── */}
        {prediction && (
          <section className="section">
            <h2 className="section-title">ML Analysis & Recommendations</h2>
            <p className="section-note">
              Analysis text and suggestions are generated by the Flask backend based on ML output
            </p>
            <AnalysisPanel prediction={prediction} />
          </section>
        )}

        {/* ── Model Info ── */}
        <section className="section">
          <h2 className="section-title">ML Model Information</h2>
          <ModelInfo modelInfo={modelInfo} />
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <p>
          ⚠️ <strong>Disclaimer:</strong> This system shows <em>ML Predicted Lead Risk</em>, not
          laboratory-certified lead detection. Always verify with certified water testing for
          health-critical decisions.
        </p>
      </footer>
    </div>
  );
}

export default App;
