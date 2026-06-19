/* ML model metadata panel */
function ModelInfo({ modelInfo }) {
  if (!modelInfo) {
    return (
      <div className="model-info-card model-info-empty">
        <span className="empty-icon">🤖</span>
        <p>Model info loading... ensure the Flask backend is running.</p>
        <code className="cmd-hint">python app.py</code>
      </div>
    );
  }

  if (modelInfo.error) {
    return (
      <div className="model-info-card model-info-warn">
        <p>⚠️ {modelInfo.error}</p>
        <code className="cmd-hint">cd backend &amp;&amp; python train_model.py</code>
      </div>
    );
  }

  const liveFeatureSet = new Set(modelInfo.live_features || []);
  const trainedAt = modelInfo.trained_at
    ? new Date(modelInfo.trained_at).toLocaleString()
    : 'N/A';

  return (
    <div className="model-info-card">
      <div className="model-info-grid">
        <InfoItem label="Model" value={modelInfo.estimator} />
        <InfoItem label="Task" value={modelInfo.model_type} />
        <InfoItem label="Dataset" value={modelInfo.dataset} />
        <InfoItem label="Kaggle Slug" value={modelInfo.kaggle_slug} />
        <InfoItem
          label={modelInfo.metric_name || 'Score'}
          value={
            modelInfo.score != null
              ? `${(modelInfo.score * 100).toFixed(1)}%`
              : 'N/A'
          }
          highlight
        />
        <InfoItem label="Estimators" value={modelInfo.n_estimators ?? 'N/A'} />
        <InfoItem label="Total Features" value={modelInfo.n_features ?? 'N/A'} />
        <InfoItem label="Trained At" value={trainedAt} wide />
      </div>

      {/* Feature tags */}
      <div className="feature-section">
        <p className="feature-section-label">Training Features</p>
        <div className="feature-tags">
          {(modelInfo.features_used || []).map((f, i) => (
            <span
              key={i}
              className={`feature-tag ${liveFeatureSet.has(f) ? 'tag-live' : 'tag-imputed'}`}
              title={liveFeatureSet.has(f) ? 'Available from hardware' : 'Imputed by ML pipeline'}
            >
              {f}
            </span>
          ))}
        </div>
        <div className="feature-legend">
          <span className="tag-live feature-tag">■ Live from hardware</span>
          <span className="tag-imputed feature-tag">■ ML imputed (not in hardware)</span>
        </div>
      </div>

      {/* Warning for missing hardware features */}
      {modelInfo.missing_live_features && modelInfo.missing_live_features.length > 0 && (
        <div className="hardware-warn">
          <strong>⚠️ Not available from current hardware:</strong>{' '}
          {modelInfo.missing_live_features.join(', ')}
          <br />
          <small>
            The ML pipeline uses median imputation for these features.
            Prediction accuracy may improve when pH, temperature sensors are added.
          </small>
        </div>
      )}
    </div>
  );
}

function InfoItem({ label, value, wide, highlight }) {
  return (
    <div className={`info-item ${wide ? 'info-wide' : ''}`}>
      <span className="info-label">{label}</span>
      <span className={`info-value ${highlight ? 'info-highlight' : ''}`}>{value}</span>
    </div>
  );
}

export default ModelInfo;
