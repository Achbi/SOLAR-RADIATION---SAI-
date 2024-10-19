"use client"
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Sa = () => {
  const [formData, setFormData] = useState({
    temperature: '',
    pressure: '',
    humidity: '',
    windDirection: '',
    speed: '',
    month: '',
    day: '',
    hour: '',
    minute: '',
    riseHour: '',
    riseMinute: '',
    setHour: '',
    setMinute: ''
  });

  const [predictions, setPredictions] = useState(null);

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.id]: e.target.value });
  };

  const predictRadiation = (e) => {
    e.preventDefault();
    // Simulating API call with fake predictions
    const fakePredictions = {
      xgboost: Math.random() * 1000,
      neuralNetwork: Math.random() * 1000
    };
    setPredictions(fakePredictions);
  };

  const chartData = predictions ? [
    { name: 'XGBoost', prediction: predictions.xgboost },
    { name: 'Neural Network', prediction: predictions.neuralNetwork }
  ] : [];

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4 bg-cover bg-center bg-no-repeat"
      style={{
        backgroundImage: `url('https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDJ8fHNvbGFyfGVufDB8fHx8MTY4MTEzNTI4Ng&ixlib=rb-4.0.3&q=80&w=1080')`
      }}
    >
      <div className="bg-black bg-opacity-70 backdrop-filter backdrop-blur-sm rounded-xl shadow-xl p-8 w-full max-w-4xl">
        <h1 className="text-4xl font-bold text-yellow-400 text-center mb-8">Solar Radiation Predictor</h1>
        <form onSubmit={predictRadiation} className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.keys(formData).map((key) => (
            <div key={key} className="flex flex-col">
              <label htmlFor={key} className="text-yellow-200 font-semibold mb-1 capitalize">
                {key.replace(/([A-Z])/g, ' $1').trim()}:
              </label>
              <input
                type="number"
                id={key}
                value={formData[key]}
                onChange={handleInputChange}
                className="bg-gray-800 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-yellow-400"
                required
              />
            </div>
          ))}
          <button
            type="submit"
            className="col-span-full bg-yellow-500 hover:bg-yellow-600 text-gray-900 font-bold py-3 px-6 rounded-full transition duration-300 ease-in-out transform hover:scale-105 mt-4"
          >
            Predict Solar Radiation
          </button>
        </form>

        {predictions && (
          <div className="mt-8 bg-gray-800 bg-opacity-50 rounded-lg p-6">
            <h2 className="text-2xl font-bold text-yellow-400 mb-4">Predictions</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              <div className="bg-green-600 bg-opacity-70 p-4 rounded-lg">
                <p className="text-yellow-200 font-semibold">XGBoost Prediction:</p>
                <p className="text-3xl font-bold text-white">{predictions.xgboost.toFixed(2)} W/m²</p>
              </div>
              <div className="bg-blue-600 bg-opacity-70 p-4 rounded-lg">
                <p className="text-yellow-200 font-semibold">Neural Network Prediction:</p>
                <p className="text-3xl font-bold text-white">{predictions.neuralNetwork.toFixed(2)} W/m²</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffd700" />
                <XAxis dataKey="name" stroke="#ffd700" />
                <YAxis stroke="#ffd700" />
                <Tooltip contentStyle={{ backgroundColor: 'rgba(0, 0, 0, 0.8)', color: '#ffd700', border: '1px solid #ffd700' }} />
                <Legend />
                <Line type="linear" dataKey="prediction" stroke="#ffd700" strokeWidth={2} dot={{ fill: '#ffd700', r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default Sa;
