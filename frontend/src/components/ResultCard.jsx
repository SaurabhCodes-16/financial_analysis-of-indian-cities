const ResultCard = ({ result }) => {
  const { status, metrics } = result;

  // Formatting currency
  const formatValue = (val) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR' }).format(val);

  return (
    <div className="result-card">
      <h2 style={{ fontSize: '2rem', marginBottom: '1rem', color: 'var(--text-primary)' }}>Your Financial Status</h2>
      
      <div className={`status-badge status-${status.replace(' ', '-')}`}>
        {status}
      </div>

      <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', lineHeight: '1.6' }}>
        {status === 'Stable' && "Excellent! Your financial habits are healthy. Keep maintaining your savings rate and expense management."}
        {status === 'At Risk' && "Warning mode ON. You have a narrow margin for error. Consider reducing discretionary expenses like entertainment and eating out."}
        {status === 'Vulnerable' && "Critical Alert! Your expenses outpace your means to sustain them in emergencies. Immediate restructuring of your budget is mandated."}
      </p>

      <div className="metrics-grid">
        <div className="metric-box">
          <span className="label">Total Expenses</span>
          <span className="value" style={{ color: 'var(--accent-red)' }}>
            {formatValue(metrics.Total_Expenses)}
          </span>
        </div>
        <div className="metric-box">
          <span className="label">Disposable Income</span>
          <span className="value" style={{ color: 'var(--accent-green)' }}>
            {formatValue(metrics.Disposable_Income)}
          </span>
        </div>
        <div className="metric-box">
          <span className="label">Savings Gap</span>
          <span className="value" style={{ color: metrics.Savings_Gap > 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
            {formatValue(metrics.Savings_Gap)}
          </span>
        </div>
        <div className="metric-box">
          <span className="label">Expense Ratio</span>
          <span className="value" style={{ color: 'var(--accent-blue)' }}>
            {(metrics.Expense_Ratio * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {result.shap_data && result.shap_data.length > 0 && (
        <div className="shap-section">
          <h3>Why are you {status}?</h3>
          <p className="subtitle">Driven by our SHAP AI interpretability engine, these are your top contributing factors.</p>
          <div className="shap-bar-container">
            {result.shap_data.map((item, index) => {
              // Calculate width based on max relative value if needed, 
              // for simplicity we cap at some reasonable scaling or just map straight to %.
              // SHAP values can vary, let's normalize to a width percentage.
              const maxShap = Math.max(...result.shap_data.map(d => Math.abs(d.value)));
              const widthRatio = maxShap === 0 ? 0 : Math.abs(item.value) / maxShap * 100;
              
              // Positive SHAP for the predicted class usually implies it pushed the prediction to this class.
              const isPositivePush = item.value > 0;
              let barColor = status === 'Stable' ? 'var(--accent-green)' : 'var(--accent-red)';
              // If it's a negative force reducing the chance of this class
              if (!isPositivePush) {
                barColor = 'var(--text-secondary)';
              }

              return (
                <div key={index} className="shap-bar-row">
                  <div className="shap-label-wrapper">
                    <span className="shap-feature-name">{item.feature.replace(/_/g, ' ')}</span>
                    <span className="shap-value">{item.value > 0 ? '+' : ''}{(item.value * 100).toFixed(2)} pts</span>
                  </div>
                  <div className="shap-bar-track">
                    <div 
                      className="shap-bar-fill" 
                      style={{ 
                        width: `${widthRatio}%`, 
                        backgroundColor: barColor,
                        marginLeft: item.value < 0 ? 'auto' : '0', // Just a visual trick if we wanted bi-directional, but doing absolute max is fine.
                      }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultCard;
