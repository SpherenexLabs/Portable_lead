/* Live sensor reading card */
function SensorCard({ label, value, unit, icon, colorClass }) {
  const displayValue =
    value === null || value === undefined
      ? '—'
      : typeof value === 'number'
      ? value.toFixed(2)
      : value;

  return (
    <div className={`sensor-card ${colorClass}`}>
      <div className="sensor-card-header">
        <span className="sensor-icon">{icon}</span>
        <span className="sensor-label">{label}</span>
      </div>
      <div className="sensor-value-row">
        <span className="sensor-value">{displayValue}</span>
        <span className="sensor-unit">{unit}</span>
      </div>
    </div>
  );
}

export default SensorCard;
