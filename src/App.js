import './App.css';
import React, { useState, useRef, useEffect } from 'react'

function App() {

  const Ref = useRef(null);

  const [timer, setTimer] = useState('00:00:00');

  const getTimeRemaining = (e) => {
    const total = Date.parse(e) - Date.parse(new Date());
    const seconds = Math.floor((total / 1000) % 60);
    const minutes = Math.floor((total / 1000 / 60) % 60);
    const hours = Math.floor((total / 1000 / 60 / 60) % 24);
    return {
        total, hours, minutes, seconds
    };
}

const startTimer = (e) => {
  let { total, hours, minutes, seconds } 
              = getTimeRemaining(e);
  if (total >= 0) {
      if ( (getTimeRemaining(e).seconds - 5) == 8 ) {
        console.log("Thirteen seconds remaining")
      }
      // update the timer
      // check if less than 10 then we need to 
      // add '0' at the beginning of the variable
      setTimer(
          (hours > 9 ? hours : '0' + hours) + ':' +
          (minutes > 9 ? minutes : '0' + minutes) + ':'
          + (seconds > 9 ? seconds : '0' + seconds)
      )
  }
}


const clearTimer = (e) => {

  // If you adjust it you should also need to
  // adjust the Endtime formula we are about
  // to code next    
  setTimer('00:00:15');

  // If you try to remove this line the 
  // updating of timer Variable will be
  // after 1000ms or 1sec
  if (Ref.current) clearInterval(Ref.current);
  const id = setInterval(() => {
      startTimer(e);
  }, 1000)
  Ref.current = id;
}

const getDeadTime = () => {
  let deadline = new Date();

  // This is where you need to adjust if 
  // you entend to add more time
  deadline.setSeconds(deadline.getSeconds() + 15);
  // deadline.getMinutes(deadline.getMinutes() + 30);
  return deadline;
}

useEffect(() => {
  clearTimer(getDeadTime());
}, []);


const onClickReset = () => {
  clearTimer(getDeadTime());
}


  return (
    <div className="App">
      <header className="App-header">
        <h2>
          Maplestory Verus Hilla Timer
        </h2>
        <h2>{timer}</h2>
        <div>
          <input type="radio" value="thresh1" name="healthThreshold" /> More than 66%
          <input type="radio" value="thresh2" name="healthThreshold" /> Less than 66% More than 33%
          <input type="radio" value="thresh3" name="healthThreshold" /> Less than 33%
        </div>
        <button onClick={onClickReset}>Reset</button>
      </header>
    </div>
  );
}

export default App;
