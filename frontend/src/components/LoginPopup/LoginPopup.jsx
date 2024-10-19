import React, { useState, useContext } from 'react';
import './LoginPopup.css';
import { AuthContext } from '../../context/AuthContext';

const LoginPopup = ({ setShowLogin }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [formData, setFormData] = useState({ email: '', password: '', name: '', country: '' });
    const [error, setError] = useState('');
    const { logIn } = useContext(AuthContext);

    const handleToggle = () => {
        setIsLogin(!isLogin);
        setError('');
    };

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        const url = isLogin 
            ? 'http://127.0.0.1:5000/api/auth/login' 
            : 'http://127.0.0.1:5000/api/auth/register';
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                credentials: 'omit',
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            
            if (data.success) {
                // For login, we get token and user data
                if (isLogin) {
                    logIn(data.user, data.token);
                    setShowLogin(false);
                } else {
                    // For register, automatically log in
                    const loginResponse = await fetch('http://127.0.0.1:5000/api/auth/login', {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        },
                        credentials: 'omit',
                        body: JSON.stringify({
                            email: formData.email,
                            password: formData.password
                        })
                    });

                    const loginData = await loginResponse.json();
                    if (loginData.success) {
                        logIn(loginData.user, loginData.token);
                        setShowLogin(false);
                    }
                }
            } else {
                setError(data.message || 'Authentication failed');
            }
        } catch (err) {
            setError(err.message || 'Failed to connect to the server');
            console.error('Error:', err);
        }
    };
  return (
    <div className="login-popup">
      <form className="login-popup-container" onSubmit={handleSubmit}>
        <div className="login-popup-title">
          <h2>{isLogin ? 'Login' : 'Register'}</h2>
          <i onClick={() => setShowLogin(false)} className='bx bx-x'></i>
        </div>
        <div className="login-popup-inputs">
          {!isLogin && (
            <input type="text" name='name' placeholder='Your Name' required onChange={handleChange} />
          )}
          <input type="email" name="email" placeholder='Email' required onChange={handleChange} />
          <input type="password" name="password" placeholder='Password' required onChange={handleChange} />
          {!isLogin && (
            <select name="country" onChange={handleChange}>
              <option value="">Select Country</option>
              <option value="India">India</option>
              <option value="US">US</option>
              <option value="UK">UK</option>
              <option value="Europe">Europe</option>
            </select>
          )}
        </div>
        <button type='submit'>
          {isLogin ? 'Login' : 'Create Account'}
        </button>

        <p>
          {isLogin ? 'Don’t have an account?' : 'Already have an account?'}
          <span onClick={handleToggle}>
            {isLogin ? ' Register Here' : ' Login Here'}
          </span>
        </p>
      </form>
    </div>
  );
};

export default LoginPopup;
