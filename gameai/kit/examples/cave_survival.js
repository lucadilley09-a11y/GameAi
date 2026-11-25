// Cave Survival: shoot (space) your way through a cave; defeat every enemy and the boss before time runs out.
function build(canvas, seed = 1) {
  const g = GameKit.game({ canvas, seed, title: 'Cave Survival', lives: 3 });
  g.level('cave', { w: 32, h: 22 });
  g.player({ canShoot: true, speed: 180 });
  g.spawn('enemy', 5, { behaviour: 'wander', shootEvery: 2.5 });
  g.spawn('boss', 1, { behaviour: 'circle', shootEvery: 1.5 });
  g.timer(120);
  g.goal('defeat');
  g.hud(['score', 'lives', 'time', 'goal']);
  g.rule({ when: 'defeat', n: 3, then: g => g.spawn('heart', 1) });
  return g;
}
