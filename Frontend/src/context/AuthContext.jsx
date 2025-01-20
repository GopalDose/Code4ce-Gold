import { createContext, useEffect, useState } from "react";
export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [ category, setCategory] = useState("All");
    const [country, setCountry] = useState('Other');
    
    useEffect(()=>{
        console.log(country)
        console.log(category)
    }, [country, category]) 

    return (
        <AuthContext.Provider value={{ 
            category,
            setCategory,
            country,
            setCountry
        }}> 
            {children} 
        </AuthContext.Provider>
    );
};