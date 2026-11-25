// Coin Dungeon: collect every coin in a generated dungeon while patrolling guards and new hunters get in the way.
function build(canvas, seed = 1) {
  const g = GameKit.game({ canvas, seed, title: 'Coin Dungeon' });
  g.level('rooms', { w: 30, h: 20 });
  g.player();
  g.spawn('coin', 12);
  g.spawn('heart', 2);
  g.spawn('enemy', 4, { behaviour: 'patrol' });
  g.goal('collect', 'coin');
  g.rule({ every: 20, then: g => g.spawn('enemy', 1, { behaviour: 'chase', speed: 60 }) });
  return g;
}
