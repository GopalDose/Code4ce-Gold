import React from 'react'
import './Home.css'
import image from '../../assets/india.jpg'
import Stock from '../Stock/Stock'
import LatestNews from '../LatestNews/LatestNews'
import Hero from '../Hero/Hero'

const Home = () => {
    return (
        <>
        <Hero />
        <div className="home">
            <LatestNews />
            <div className="stockContainer">
                <Stock />
            </div>
        </div>
        </>
    )
}

export default Home