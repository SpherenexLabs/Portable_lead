import axios from 'axios';

const API_BASE = 'http://localhost:5000';

// Send live sensor values to Flask backend for ML prediction
export const predictLeadContamination = async (sensorData) => {
  const response = await axios.post(`${API_BASE}/api/predict`, sensorData, {
    timeout: 10000,
  });
  return response.data;
};

// Get ML model info (dataset, accuracy, features)
export const getModelInfo = async () => {
  const response = await axios.get(`${API_BASE}/api/model-info`, {
    timeout: 5000,
  });
  return response.data;
};

// Trigger model re-training (optional)
export const triggerTraining = async () => {
  const response = await axios.post(`${API_BASE}/api/train`, {}, {
    timeout: 300000,
  });
  return response.data;
};

// Health check
export const checkBackendHealth = async () => {
  const response = await axios.get(`${API_BASE}/`, { timeout: 3000 });
  return response.data;
};
