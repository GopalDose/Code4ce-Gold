import React, { useContext } from 'react';
import './Header.css';
import { AuthContext } from '../../context/AuthContext'; // Adjust the path as needed
import logo from '../../assets/logo.png';

const Header = ({setShowLogin, showLogin}) => {
    const { isLoggedIn, logIn, logOut } = useContext(AuthContext);
    
    const logout = () => {
        logOut();
    }

    return (
        <header>
            <div className="brand">
                {/* <img src={logo} alt="" /> */}
                <a href="/">WarCast</a>
            </div>
            <nav className="navbar">
                <ul>
                    <li>
                        <a href="">Home</a>
                    </li>
                    <li>
                        <a href="">Contact</a>
                    </li>
                    {isLoggedIn ? (
                        <li>
                            <i className='bx bx-user-circle'></i>
                            <div className="dropdown">
                                <i className='bx bxs-up-arrow'></i>
                                <ul>
                                    <li onClick={logout}>Logout</li>
                                </ul>
                            </div>
                        </li>
                    ) : (
                        <li>
                            <button onClick={() => setShowLogin(true)}>Login</button> 
                        </li>
                    )}
                </ul>
            </nav>
        </header>
    );
}

export default Header;
