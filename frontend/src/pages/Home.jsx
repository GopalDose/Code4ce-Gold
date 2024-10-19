import React, { useState } from 'react'
import LatestNews from '../components/LatestNews/LatestNews'
import Stock from '../components/Stock/Stock'
import Hero from '../components/Hero/Hero'

const Home = () => {
  const [homeState, setHomeState] = useState("Latest")
  return (
    <>
    <Hero/>
    <div className="hometab">
    <LatestNews homeState={homeState} setHomeState={setHomeState}/>
    <Stock/>
    </div>
    </>
  )
}

export default Home