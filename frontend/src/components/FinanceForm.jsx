import { useState } from 'react';

const FinanceForm = ({ onSubmit, disabled }) => {
  const [formData, setFormData] = useState({
    Income: '',
    Age: '',
    Dependents: '',
    City_Tier: 'Tier 1',
    Occupation: 'Employed',
    Rent: '',
    Loan_Repayment: '',
    Insurance: '',
    Groceries: '',
    Transport: '',
    Eating_Out: '',
    Entertainment: '',
    Utilities: '',
    Healthcare: '',
    Education: '',
    Miscellaneous: '',
    Desired_Savings_Percentage: '',
    Potential_Savings_Groceries: '',
    Potential_Savings_Eating_Out: '',
    Potential_Savings_Entertainment: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-section">
        <h3>1. Demographics</h3>
        <div className="input-grid">
          <div className="input-group">
            <label>Age</label>
            <input type="number" name="Age" value={formData.Age} onChange={handleChange} required min="18" max="100"/>
          </div>
          <div className="input-group">
            <label>Dependents</label>
            <input type="number" name="Dependents" value={formData.Dependents} onChange={handleChange} required min="0" max="20"/>
          </div>
          <div className="input-group">
            <label>City Tier</label>
            <select name="City_Tier" value={formData.City_Tier} onChange={handleChange}>
              <option value="Tier 1">Tier 1 (Metro)</option>
              <option value="Tier 2">Tier 2</option>
              <option value="Tier 3">Tier 3</option>
            </select>
          </div>
          <div className="input-group">
            <label>Occupation</label>
            <select name="Occupation" value={formData.Occupation} onChange={handleChange}>
              <option value="Employed">Salaried / Employed</option>
              <option value="Self-Employed">Self-Employed / Business</option>
              <option value="Student">Student</option>
              <option value="Retired">Retired</option>
            </select>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>2. Income & Targets</h3>
        <div className="input-grid">
          <div className="input-group">
            <label>Monthly Income (₹)</label>
            <input type="number" name="Income" value={formData.Income} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Desired Savings (%)</label>
            <input type="number" name="Desired_Savings_Percentage" value={formData.Desired_Savings_Percentage} onChange={handleChange} required min="0" max="100" />
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>3. Monthly Expenses (₹)</h3>
        <div className="input-grid">
          <div className="input-group">
            <label>Rent / EMI</label>
            <input type="number" name="Rent" value={formData.Rent} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Personal Loan Repayment</label>
            <input type="number" name="Loan_Repayment" value={formData.Loan_Repayment} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Groceries</label>
            <input type="number" name="Groceries" value={formData.Groceries} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Transport</label>
            <input type="number" name="Transport" value={formData.Transport} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Eating Out</label>
            <input type="number" name="Eating_Out" value={formData.Eating_Out} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Entertainment</label>
            <input type="number" name="Entertainment" value={formData.Entertainment} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Utilities</label>
            <input type="number" name="Utilities" value={formData.Utilities} onChange={handleChange} required />
          </div>
          <div className="input-group">
            <label>Healthcare</label>
            <input type="number" name="Healthcare" value={formData.Healthcare} onChange={handleChange} required />
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>4. Potential Savings Goals (₹)</h3>
        <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem'}}>
          How much do you realistically think you can cut down on these?
        </p>
        <div className="input-grid">
          <div className="input-group">
            <label>Potential cuts in Groceries</label>
            <input type="number" name="Potential_Savings_Groceries" value={formData.Potential_Savings_Groceries} onChange={handleChange} placeholder="0" />
          </div>
          <div className="input-group">
            <label>Potential cuts in Eating Out</label>
            <input type="number" name="Potential_Savings_Eating_Out" value={formData.Potential_Savings_Eating_Out} onChange={handleChange} placeholder="0" />
          </div>
          <div className="input-group">
            <label>Potential cuts in Entertainment</label>
            <input type="number" name="Potential_Savings_Entertainment" value={formData.Potential_Savings_Entertainment} onChange={handleChange} placeholder="0" />
          </div>
        </div>
      </div>

      <button type="submit" className="submit-btn" disabled={disabled}>
        {disabled ? 'Processing...' : 'Calculate Status'}
      </button>
    </form>
  );
};

export default FinanceForm;
