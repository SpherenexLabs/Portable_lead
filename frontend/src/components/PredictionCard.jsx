/* ML prediction result card — all values come from backend ML model output */
function PredictionCard({ prediction, loading }) {
  // Full spinner only on very first load (no result yet)
  if (loading && !prediction) {
    return (
      <div className="prediction-card card-loading">
        <div className="spinner"></div>
        <p className="loading-text">Analyzing with ML model...</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="prediction-card card-empty">
        <span className="empty-icon">🔬</span>
        <p>Waiting for sensor data and ML prediction...</p>
        <p className="hint-text">Make sure the Flask backend is running.</p>
      </div>
    );
  }

  // Color class driven by ML-predicted status (not raw sensor values)
  const statusColorMap = {
    Drinkable: 'status-safe',
    Risky: 'status-risky',
    'Not Drinkable': 'status-unsafe',
  };
  const safetyColorMap = {
    Safe: 'tag-safe',
    'Moderate Risk': 'tag-risky',
    Unsafe: 'tag-unsafe',
  };
  const statusIcon = {
    Drinkable: '✅',
    Risky: '⚠️',
    'Not Drinkable': '❌',
  };

  const statusClass = statusColorMap[prediction.status] || 'status-safe';
  const safetyClass = safetyColorMap[prediction.safety_level] || 'tag-safe';

  return (
    <div className={`prediction-card ${statusClass}`}>
      {/* Small live-updating badge — shows only while a background fetch is running */}
      {loading && (
        <div className="pred-updating-badge">
          <span className="pred-updating-dot" />
          Updating...
        </div>
      )}

      <div className="prediction-grid">
        {/* Left: percentage gauge */}
        <div className="pred-left">
          <p className="pred-main-label">ML Predicted Lead Risk</p>
          <div className="percentage-display">
            <span className="percentage-number">{prediction.lead_percentage.toFixed(1)}</span>
            <span className="percentage-pct">%</span>
          </div>
          <div className="risk-bar-bg">
            <div
              className={`risk-bar-fill ${statusClass}`}
              style={{ width: `${Math.min(prediction.lead_percentage, 100)}%` }}
            />
          </div>
          <div className="risk-bar-labels">
            <span>0%</span>
            <span>Safe</span>
            <span>Risk</span>
            <span>100%</span>
          </div>
        </div>

        {/* Right: detail items */}
        <div className="pred-right">
          <div className={`status-chip ${statusClass}`}>
            {statusIcon[prediction.status]} {prediction.status}
          </div>

          <div className="pred-detail-list">
            <div className="pred-detail-item">
              <span className="pred-detail-label">Safety Level</span>
              <span className={`pred-detail-value tag ${safetyClass}`}>
                {prediction.safety_level}
              </span>
            </div>
            <div className="pred-detail-item">
              <span className="pred-detail-label">ML Confidence</span>
              <span className="pred-detail-value">{prediction.confidence.toFixed(1)}%</span>
            </div>
            <div className="pred-detail-item">
              <span className="pred-detail-label">Model Based</span>
              <span className="pred-detail-value">
                {prediction.model_based ? '✓ Yes' : '✗ No'}
              </span>
            </div>
            <div className="pred-detail-item">
              <span className="pred-detail-label">Input TDS</span>
              <span className="pred-detail-value">{prediction.input_values?.TDS?.toFixed(1)} ppm</span>
            </div>
            <div className="pred-detail-item">
              <span className="pred-detail-label">Input Turb.</span>
              <span className="pred-detail-value">{prediction.input_values?.TURB?.toFixed(2)} NTU</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PredictionCard;
