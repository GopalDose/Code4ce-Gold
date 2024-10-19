import { createContext, useEffect, useState } from "react";

// Create context
export const AuthContext = createContext(null);

// Provide context to the component tree
export const AuthProvider = ({ children }) => {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [ category, setCategory] = useState("All");
    const [language, setLanguage] = useState("en"); // New state for language
    const [country, setCountry] = useState('');
    // Check for token on initial load
    useEffect(() => {
        const checkAuth = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    // Verify token and get user data
                    const response = await fetch('http://127.0.0.1:5000/api/auth/user/profile', {
                        headers: {
                            'Authorization': `Bearer ${token}`,
                            'Content-Type': 'application/json'
                        }
                    });

                    if (response.ok) {
                        const data = await response.json();
                        setUser(data.user);
                        setIsLoggedIn(true);
                    } else {
                        // If token is invalid, clear it
                        localStorage.removeItem('token');
                        setIsLoggedIn(false);
                        setUser(null);
                    }
                } catch (error) {
                    console.error('Auth check failed:', error);
                    localStorage.removeItem('token');
                    setIsLoggedIn(false);
                    setUser(null);
                }
            }
            setIsLoading(false);
        };

        checkAuth();
    }, []);

    const logIn = (userData, token) => {
        localStorage.setItem('token', token);
        setUser(userData);
        setIsLoggedIn(true);
    };

    const logOut = () => {
        localStorage.removeItem('token');
        setUser(null);
        setIsLoggedIn(false);
    };

    // Get the current auth token
    const getToken = () => localStorage.getItem('token');

    if (isLoading) {
        return <div>Loading...</div>; // Or your loading component
    }

    return (
        <AuthContext.Provider value={{ 
            isLoggedIn, 
            logIn, 
            logOut, 
            user, 
            getToken ,
            category,
            setCategory,
            language, 
            setLanguage,
            country,
            setCountry
        }}>
            {children}
        </AuthContext.Provider>
    );
};