import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function TrendChart({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="chart-card chart-empty">
        <span className="empty-icon">📈</span>
        <p>Waiting for sensor data to build trend...</p>
      </div>
    );
  }

  return (
    <div className="chart-card">
      <div className="chart-block">
        <h3 className="chart-subtitle">TDS (ppm) — Last {history.length} Readings</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8eaf0" />
            <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#6b7280' }} />
            <YAxis tick={{ fontSize: 11, fill: '#6b7280' }} />
            <Tooltip
              contentStyle={{ borderRadius: '8px', border: '1px solid #e0e0e0' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="TDS"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
              name="TDS (ppm)"
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-block">
        <h3 className="chart-subtitle">Turbidity (NTU) — Last {history.length} Readings</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={history} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e8eaf0" />
            <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#6b7280' }} />
            <YAxis tick={{ fontSize: 11, fill: '#6b7280' }} />
            <Tooltip
              contentStyle={{ borderRadius: '8px', border: '1px solid #e0e0e0' }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="TURB"
              stroke="#059669"
              strokeWidth={2}
              dot={false}
              name="Turbidity (NTU)"
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default TrendChart;
