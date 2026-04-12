import { useState } from 'react';
import FinanceForm from './components/FinanceForm';
import ResultCard from './components/ResultCard';

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const calculateStatus = async (formData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to connect to the backend API. Make sure it is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Financial Predictor AI</h1>
        <p>Discover your personal financial vulnerability status powered by Random Forest</p>
      </header>

      <main className="dashboard">
        <section className="glass-panel">
          <FinanceForm onSubmit={calculateStatus} disabled={loading} />
        </section>
        
        <section className="glass-panel" style={{ '--animation-delay': '0.2s', animationDelay: '0.2s' }}>
          {loading ? (
            <div className="loading">Analyzing your profile...</div>
          ) : error ? (
            <div className="empty-state">
              <span className="icon" style={{color: 'var(--accent-red)'}}>⚠️</span>
              <p>{error}</p>
            </div>
          ) : result ? (
            <ResultCard result={result} />
          ) : (
            <div className="empty-state">
              <span className="icon">📊</span>
              <h2>Ready to Analyze</h2>
              <p>Fill out the comprehensive form to get your customized financial health check.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
