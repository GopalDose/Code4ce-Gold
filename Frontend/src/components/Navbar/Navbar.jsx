import React from 'react'
import './Navbar.css'

const Navbar = () => {
  return (
    <header>
        <div className="brand">
            WarCast
        </div>
        <div className="nav">
            <ul>
                <li>
                    <a href="">Home</a>
                </li>

                <li>
                    <a href="">Contact Us</a>
                </li>
                <li>
                    <a href="" className='loginbtn'>Login</a>
                </li>
            </ul>
        </div>
    </header>
  )
}

export default Navbar