/* Analysis and suggestions — all text sourced from ML backend response */
function AnalysisPanel({ prediction }) {
  if (!prediction) return null;

  const statusColorMap = {
    Drinkable: 'status-safe',
    Risky: 'status-risky',
    'Not Drinkable': 'status-unsafe',
  };
  const statusClass = statusColorMap[prediction.status] || 'status-safe';

  return (
    <div className="analysis-panel">
      {/* ML Analysis block */}
      <div className="analysis-card">
        <div className="analysis-header">
          <span className="analysis-icon">🔬</span>
          <h3 className="analysis-title">ML Analysis</h3>
        </div>
        <p className="analysis-text">{prediction.analysis}</p>
      </div>

      {/* Suggestions from backend */}
      <div className="suggestions-card">
        <div className="analysis-header">
          <span className="analysis-icon">💡</span>
          <h3 className="analysis-title">ML Recommendations</h3>
        </div>
        {prediction.suggestions && prediction.suggestions.length > 0 ? (
          <ul className="suggestions-list">
            {prediction.suggestions.map((s, i) => (
              <li key={i} className={`suggestion-item ${statusClass}`}>
                <span className="suggestion-arrow">›</span>
                <span>{s}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="analysis-text">No recommendations available.</p>
        )}
      </div>
    </div>
  );
}

export default AnalysisPanel;
