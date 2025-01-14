var elDiceOne       = document.getElementById('dice1');
var elDiceTwo       = document.getElementById('dice2');
// var elComeOut       = document.getElementById('roll');

// elComeOut.onclick   = function () {rollDice();};

function rollDice(roll) {

//   console.log('Dices received:', roll);
//   var diceOne   = Math.floor((Math.random() * 6) + 1);
//   var diceTwo   = Math.floor((Math.random() * 6) + 1);

// //   const [diceOne, diceTwo] = roll;
  diceOne = roll[0]
  diceTwo = roll[1]
  console.log('Rolling dice:', diceOne, diceTwo);
  const dice_mix = {
    1: 1,
    2: 6,
    3: 4,
    4: 5,
    5: 2,
    6: 3
  };
 

  for (var i = 1; i <= 6; i++) {
    elDiceOne.classList.remove('show-' + i);
    if (dice_mix[diceOne] === i) {
      elDiceOne.classList.add('show-' + i);
    }
  }

  for (var k = 1; k <= 6; k++) {
    elDiceTwo.classList.remove('show-' + k);
    if (dice_mix[diceTwo] === k) {
      elDiceTwo.classList.add('show-' + k);
    }
  } 
//   setTimeout(rollDice, 1000);
}
