var elDiceOne       = document.getElementById('dice1');
var elDiceTwo       = document.getElementById('dice2');
var elComeOut       = document.getElementById('roll');

// elComeOut.onclick   = function () {rollDice();};

function rollDice(roll) {

//   console.log('Dices received:', roll);
//   var diceOne   = Math.floor((Math.random() * 6) + 1);
//   var diceTwo   = Math.floor((Math.random() * 6) + 1);

//   const [diceOne, diceTwo] = roll;
  diceOne = roll[0]
  diceTwo = roll[1]
  console.log('Rolling dice:', diceOne, diceTwo);
 
  console.log(diceOne + ' ' + diceTwo);

  for (var i = 1; i <= 6; i++) {
    elDiceOne.classList.remove('show-' + i);
    if (diceOne === i) {
      elDiceOne.classList.add('show-' + i);
    }
  }

  for (var k = 1; k <= 6; k++) {
    elDiceTwo.classList.remove('show-' + k);
    if (diceTwo === k) {
      elDiceTwo.classList.add('show-' + k);
    }
  } 
//   setTimeout(rollDice, 1000);
}
